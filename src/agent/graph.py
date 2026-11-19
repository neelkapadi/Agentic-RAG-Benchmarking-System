"""Agentic retrieval as a LangGraph state machine (v2).

    route ──► retrieve ──► grade ──(accepted or out of attempts)──► finalize ──► END
                 ▲           │
                 └─ expand ◄─┘ (rejected)

- route:    LLM picks a *weighting* of BM25 and vector search: keyword / semantic / balanced
            (default balanced). Every strategy is a weighted hybrid, so a single weak retriever
            can never be chosen on its own.
- retrieve: BM25 + vector candidates fused with weighted Reciprocal Rank Fusion. Also records
            whether BM25 and vector *agree* on the top chunk (consensus).
- grade:    consensus → accept without an LLM call. Otherwise the LLM checks whether any passage
            is *relevant* to the question (partial answers count; full passages shown).
- expand:   instead of paraphrasing, the LLM writes a hypothetical answer passage + keywords
            (HyDE + query expansion), which brings in vocabulary the question lacked.
- finalize: the accepted attempt, or a fusion that weights the first attempt double, so a
            failed retry can never push a good first result far down.

v1 → v2 changes came from the pilot benchmark: the v1 grader rejected 9 of the 11 first
retrievals that already contained the gold chunk, rewrites were near-paraphrases, the router
often chose plain vector search, and fusing all attempts lowered good rankings.
"""
from __future__ import annotations

import operator
import re
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from src.generation.llm import as_bool
from src.generation.prompts import (EXPAND_SYSTEM, EXPAND_USER, GRADER_SYSTEM, GRADER_USER,
                                    ROUTER_SYSTEM, ROUTER_USER, fill)
from src.retrievers.base import BaseRetriever, RetrievedChunk
from src.retrievers.hybrid import rrf_fuse

# strategy -> (bm25 weight, vector weight) for weighted RRF
STRATEGY_WEIGHTS = {"keyword": (2.0, 1.0), "semantic": (1.0, 2.0), "balanced": (1.0, 1.0)}
STRATEGIES = tuple(STRATEGY_WEIGHTS)
CODE_HINT_RE = re.compile(r"`|\b\w+_\w+\b|\w+\(\)|\b[A-Z][a-z]+[A-Z]\w*\b|--\w+|\b[1-5]\d\d\b")
CONCEPTUAL_RE = re.compile(r"^(why|how (do|can|should)|what is the (difference|best way|purpose)|when should)", re.I)


class AgentState(TypedDict, total=False):
    question: str
    query: str
    strategy: str
    docs: list[RetrievedChunk]
    history: Annotated[list[list[RetrievedChunk]], operator.add]
    tried: Annotated[list[str], operator.add]
    attempts: int
    consensus: bool
    sufficient: bool
    trace: Annotated[list[dict], operator.add]
    llm_calls: Annotated[int, operator.add]
    prompt_tokens: Annotated[int, operator.add]
    completion_tokens: Annotated[int, operator.add]


def heuristic_route(query: str) -> str:
    if CODE_HINT_RE.search(query):
        return "keyword"
    if CONCEPTUAL_RE.search(query.strip()):
        return "semantic"
    return "balanced"


def has_consensus(bm25_ids: list[str], vector_ids: list[str], depth: int = 3) -> bool:
    """BM25 and vector search independently point at the same top chunk."""
    if not bm25_ids or not vector_ids:
        return False
    if bm25_ids[0] == vector_ids[0]:
        return True
    return bm25_ids[0] in vector_ids[:depth] and vector_ids[0] in bm25_ids[:depth]


def _usage(resp) -> dict:
    return {"llm_calls": 1, "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens}


def build_agent(llm, retrievers: dict[str, BaseRetriever], top_k: int = 5, max_attempts: int = 2,
                rrf_k: int = 60, candidates: int = 20, grade_passages: int = 3,
                passage_chars: int = 1500, use_consensus: bool = True,
                domain: str = "the provided documents"):
    bm25, vector = retrievers["bm25"], retrievers["vector"]
    router_system = fill(ROUTER_SYSTEM, domain=domain)
    grader_system = fill(GRADER_SYSTEM, domain=domain)
    expand_system = fill(EXPAND_SYSTEM, domain=domain)

    def route(state: AgentState) -> dict:
        data, resp = llm.chat_json([
            {"role": "system", "content": router_system},
            {"role": "user", "content": fill(ROUTER_USER, query=state["question"])},
        ])
        strategy = str(data.get("strategy", "")).strip().lower()
        reason = data.get("reason", "")
        if strategy not in STRATEGY_WEIGHTS:
            strategy, reason = heuristic_route(state["question"]), "fallback: rule-based router"
        return {"strategy": strategy, **_usage(resp),
                "trace": [{"step": "route", "strategy": strategy, "reason": reason,
                           "latency_s": round(resp.latency_s, 3)}]}

    def retrieve(state: AgentState) -> dict:
        start = time.perf_counter()
        n = max(top_k, candidates)
        b, v = bm25.search(state["query"], n), vector.search(state["query"], n)
        docs = rrf_fuse([b, v], k=rrf_k, weights=list(STRATEGY_WEIGHTS[state["strategy"]]))[:top_k]
        consensus = has_consensus([d.chunk_id for d in b], [d.chunk_id for d in v])
        return {"docs": docs, "history": [docs], "tried": [state["strategy"]], "consensus": consensus,
                "trace": [{"step": "retrieve", "strategy": state["strategy"], "query": state["query"][:300],
                           "chunk_ids": [d.chunk_id for d in docs], "consensus": consensus,
                           "latency_s": round(time.perf_counter() - start, 3)}]}

    def grade(state: AgentState) -> dict:
        attempts = state.get("attempts", 0) + 1
        if not state["docs"]:
            return {"sufficient": False, "attempts": attempts,
                    "trace": [{"step": "grade", "sufficient": False, "reason": "no passages", "by": "rule"}]}
        if use_consensus and state.get("consensus"):
            return {"sufficient": True, "attempts": attempts,
                    "trace": [{"step": "grade", "sufficient": True, "by": "consensus",
                               "reason": "BM25 and vector search agree on the top chunk"}]}
        passages = "\n\n".join(f"[{i}] {d.text[:passage_chars]}"
                               for i, d in enumerate(state["docs"][:grade_passages], 1))
        data, resp = llm.chat_json([
            {"role": "system", "content": grader_system},
            {"role": "user", "content": fill(GRADER_USER, question=state["question"], passages=passages)},
        ])
        # Lenient default: an unparseable verdict should not trigger a useless retry.
        sufficient = as_bool(data["relevant"]) if "relevant" in data else True
        return {"sufficient": sufficient, "attempts": attempts, **_usage(resp),
                "trace": [{"step": "grade", "sufficient": sufficient, "by": "llm",
                           "reason": data.get("reason", ""), "latency_s": round(resp.latency_s, 3)}]}

    def expand(state: AgentState) -> dict:
        data, resp = llm.chat_json([
            {"role": "system", "content": expand_system},
            {"role": "user", "content": fill(EXPAND_USER, question=state["question"])},
        ])
        keywords = data.get("keywords") or []
        if isinstance(keywords, str):
            keywords = [keywords]
        passage = str(data.get("passage") or "").strip()
        query = " ".join([state["question"], " ".join(map(str, keywords)), passage]).strip()
        # A retry with a different weighting than the first attempt, on the expanded query.
        next_strategy = "balanced" if state["strategy"] != "balanced" else "semantic"
        return {"query": query, "strategy": next_strategy, **_usage(resp),
                "trace": [{"step": "expand", "keywords": keywords[:10], "passage": passage[:300],
                           "next_strategy": next_strategy, "latency_s": round(resp.latency_s, 3)}]}

    def finalize(state: AgentState) -> dict:
        if state.get("sufficient"):
            return {"trace": [{"step": "finalize", "mode": "accepted", "attempts": state["attempts"]}]}
        history = state.get("history", [])
        weights = [2.0] + [1.0] * (len(history) - 1)  # keep the first attempt's ranking dominant
        docs = rrf_fuse(history, k=rrf_k, weights=weights)[:top_k]
        return {"docs": docs, "trace": [{"step": "finalize", "mode": "fused_first_weighted",
                                         "attempts": state["attempts"]}]}

    def decide(state: AgentState) -> str:
        if state.get("sufficient") or state.get("attempts", 0) >= max_attempts:
            return "finalize"
        return "expand"

    g = StateGraph(AgentState)
    for name, fn in (("route", route), ("retrieve", retrieve), ("grade", grade),
                     ("expand", expand), ("finalize", finalize)):
        g.add_node(name, fn)
    g.set_entry_point("route")
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", decide, {"expand": "expand", "finalize": "finalize"})
    g.add_edge("expand", "retrieve")
    g.add_edge("finalize", END)
    return g.compile()


class AgenticRetriever(BaseRetriever):
    """Wraps the LangGraph agent behind the common retriever interface."""
    name = "agentic"

    def __init__(self, llm, retrievers: dict[str, BaseRetriever], top_k: int = 5,
                 max_attempts: int = 2, rrf_k: int = 60, domain: str = "the provided documents",
                 candidates: int = 20, use_consensus: bool = True):
        self.top_k = top_k
        self.graph = build_agent(llm, retrievers, top_k=top_k, max_attempts=max_attempts, rrf_k=rrf_k,
                                 candidates=candidates, use_consensus=use_consensus, domain=domain)

    def search_with_trace(self, query: str, k: int | None = None) -> tuple[list[RetrievedChunk], dict]:
        state = self.graph.invoke({"question": query, "query": query, "attempts": 0,
                                   "llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
                                  {"recursion_limit": 50})
        docs = state.get("docs", [])[: k or self.top_k]
        info = {key: state.get(key) for key in
                ("trace", "attempts", "sufficient", "llm_calls", "prompt_tokens", "completion_tokens")}
        info["strategies"] = state.get("tried", [])
        return docs, info

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        return self.search_with_trace(query, k)[0]

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()
