"""RAG pipelines (retrieve -> generate) and a factory that builds them from config."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.agent.graph import AgenticRetriever
from src.config import resolve
from src.generation.generator import AnswerGenerator
from src.generation.llm import OllamaClient
from src.ingest.store import ChunkStore
from src.retrievers.base import BaseRetriever, RetrievedChunk

ALL_PIPELINES = ("bm25", "vector", "hybrid", "agentic")


@dataclass
class PipelineResult:
    pipeline: str
    question: str
    answer: str
    contexts: list[RetrievedChunk]
    timings: dict
    tokens: dict
    agent: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"pipeline": self.pipeline, "question": self.question, "answer": self.answer,
                "contexts": [c.to_dict() for c in self.contexts], "timings": self.timings,
                "tokens": self.tokens, "agent": self.agent}


class RAGPipeline:
    def __init__(self, name: str, retriever: BaseRetriever, generator: AnswerGenerator | None, top_k: int = 5):
        self.name, self.retriever, self.generator, self.top_k = name, retriever, generator, top_k

    def run(self, question: str, generate: bool = True) -> PipelineResult:
        start = time.perf_counter()
        agent_info: dict = {}
        if isinstance(self.retriever, AgenticRetriever):
            docs, agent_info = self.retriever.search_with_trace(question, self.top_k)
        else:
            docs = self.retriever.search(question, self.top_k)
        retrieval_s = time.perf_counter() - start

        answer, gen_s, p_tok, c_tok = "", 0.0, 0, 0
        if generate and self.generator:
            resp = self.generator.generate(question, docs)
            answer, gen_s, p_tok, c_tok = resp.text, resp.latency_s, resp.prompt_tokens, resp.completion_tokens

        tokens = {"prompt": p_tok + (agent_info.get("prompt_tokens") or 0),
                  "completion": c_tok + (agent_info.get("completion_tokens") or 0),
                  "llm_calls": (1 if generate and self.generator else 0) + (agent_info.get("llm_calls") or 0)}
        timings = {"retrieval_s": retrieval_s, "generation_s": gen_s,
                   "total_s": time.perf_counter() - start}
        return PipelineResult(self.name, question, answer, docs, timings, tokens, agent_info)


def domain_of(cfg: dict) -> str:
    """Corpus description used in prompts (set by src.corpus.apply_corpus)."""
    return cfg.get("domain") or cfg["data"].get("domain", "the provided documents")


def make_llm(cfg: dict, judge: bool = False) -> OllamaClient:
    c = cfg["llm"]
    return OllamaClient(base_url=c["base_url"], model=c["judge_model"] if judge else c["model"],
                        temperature=c.get("temperature", 0.0), num_ctx=c.get("num_ctx", 8192),
                        timeout_s=c.get("timeout_s", 180), max_tokens=c.get("max_tokens", 768),
                        json_max_tokens=c.get("json_max_tokens", 512))


def build_retrievers(cfg: dict, names: list[str], llm: OllamaClient | None = None) -> dict[str, BaseRetriever]:
    # Imported lazily so BM25-only use does not load torch.
    from src.retrievers.bm25 import BM25Retriever
    from src.retrievers.hybrid import HybridRetriever

    store = ChunkStore.load(resolve(cfg["data"]["chunks_path"]))
    r_cfg, e_cfg = cfg["retrieval"], cfg["embeddings"]
    out: dict[str, BaseRetriever] = {}
    need_vector = any(n in names for n in ("vector", "hybrid", "agentic"))

    out["bm25"] = BM25Retriever(store, **r_cfg.get("bm25", {}))
    if need_vector:
        from src.retrievers.vector import VectorRetriever
        out["vector"] = VectorRetriever(store, e_cfg["model"], resolve(e_cfg["index_dir"]),
                                        e_cfg.get("query_prefix", ""), e_cfg.get("batch_size", 32))
        out["hybrid"] = HybridRetriever(out["bm25"], out["vector"], **r_cfg.get("hybrid", {}))
    if "agentic" in names:
        out["agentic"] = AgenticRetriever(
            llm or make_llm(cfg), {k: out[k] for k in ("bm25", "vector", "hybrid")},
            top_k=r_cfg["top_k"], max_attempts=cfg["agent"]["max_attempts"],
            rrf_k=r_cfg.get("hybrid", {}).get("rrf_k", 60), domain=domain_of(cfg))
    return {n: out[n] for n in names}


def build_pipelines(cfg: dict, names: list[str] | None = None, llm: OllamaClient | None = None,
                    with_generator: bool = True) -> dict[str, RAGPipeline]:
    names = list(names or cfg["eval"]["pipelines"])
    unknown = set(names) - set(ALL_PIPELINES)
    if unknown:
        raise ValueError(f"Unknown pipelines: {unknown}. Choose from {ALL_PIPELINES}")
    llm = llm or make_llm(cfg)
    retrievers = build_retrievers(cfg, names, llm)
    generator = AnswerGenerator(llm, domain_of(cfg)) if with_generator else None
    return {n: RAGPipeline(n, retrievers[n], generator, cfg["retrieval"]["top_k"]) for n in names}
