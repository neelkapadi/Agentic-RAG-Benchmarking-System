"""Unit tests that run without Ollama or embedding models."""
from pathlib import Path

import pytest

from src.agent.graph import AgenticRetriever, heuristic_route
from src.evaluation.retrieval_metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from src.generation.llm import LLMResponse
from src.ingest.chunker import chunk_document, split_blocks, split_sections
from src.ingest.loader import Document, clean_markdown
from src.retrievers.base import BaseRetriever, RetrievedChunk
from src.retrievers.bm25 import tokenize
from src.retrievers.hybrid import rrf_fuse


def rc(cid: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, doc_id=cid.split("::")[0], text=f"text {cid}", score=0.0)


# ---------- ingest ----------

def test_clean_markdown_inlines_code_and_strips_syntax(tmp_path: Path):
    (tmp_path / "docs_src" / "x").mkdir(parents=True)
    (tmp_path / "docs_src" / "x" / "tutorial001.py").write_text("print('hi')  # <b>keep</b>")
    md = "# Title { #title }\n\n/// tip\n\nUse <dfn title='t'>this</dfn>.\n\n///\n\n{* ../../docs_src/x/tutorial001.py hl[1] *}"
    out = clean_markdown(md, tmp_path)
    assert "{ #title }" not in out and "<dfn" not in out and "///" not in out
    assert "**Tip:**" in out
    assert "```python\nprint('hi')  # <b>keep</b>\n```" in out  # HTML inside code untouched


def test_sections_ignore_hashes_inside_code():
    text = "# A\n\nintro\n\n## B\n\n```python\n# not a heading\n```\n\nbody"
    sections = split_sections(text)
    assert [s[0] for s in sections] == ["A", "A > B"]
    assert "# not a heading" in sections[1][1]


def test_blocks_keep_code_fences_whole():
    blocks = split_blocks("para one\n\n```py\na = 1\n\nb = 2\n```\n\npara two")
    assert blocks == ["para one", "```py\na = 1\n\nb = 2\n```", "para two"]


def test_chunk_document_respects_max_chars():
    body = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(20))
    chunks = chunk_document(Document("t.md", "T", f"# T\n\n{body}"), max_chars=800, overlap_chars=100, min_chars=10)
    assert len(chunks) > 1
    assert all(len(c.text) < 800 + 50 for c in chunks)
    assert chunks[0].chunk_id == "t.md::000" and chunks[0].text.startswith("T\n\n")


def test_tokenize_keeps_and_splits_identifiers():
    toks = tokenize("What does response_model do?")
    assert "response_model" in toks and "response" in toks and "model" in toks and "what" not in toks


# ---------- retrieval ----------

def test_rrf_prefers_items_ranked_high_in_both_lists():
    fused = rrf_fuse([[rc("a"), rc("b"), rc("c")], [rc("b"), rc("d"), rc("a")]], k=60)
    assert [r.chunk_id for r in fused][:2] == ["b", "a"]


def test_metrics():
    retrieved, gold = ["x", "g", "y"], {"g"}
    assert mrr(retrieved, gold) == 0.5
    assert recall_at_k(retrieved, gold, 1) == 0.0 and recall_at_k(retrieved, gold, 3) == 1.0
    assert precision_at_k(retrieved, gold, 2) == 0.5
    assert ndcg_at_k(["g"], gold, 5) == 1.0
    assert 0 < ndcg_at_k(retrieved, gold, 3) < 1


@pytest.mark.parametrize("q,expected", [
    ("What does `response_model` do?", "bm25"),
    ("How do I return a 404 error?", "bm25"),
    ("Why should I use async functions?", "vector"),
    ("background tasks after sending a response", "hybrid"),
])
def test_heuristic_router(q, expected):
    assert heuristic_route(q) == expected


# ---------- agent (fake LLM + fake retrievers) ----------

class FakeRetriever(BaseRetriever):
    def __init__(self, name):
        self.name = name

    def search(self, query, k=5):
        return [rc(f"{self.name}.md::{i:03d}") for i in range(k)]


class FakeLLM:
    """Router always says bm25; grader rejects the first attempt and accepts the second."""
    def __init__(self):
        self.grades = 0

    def chat_json(self, messages):
        system = messages[0]["content"]
        if "route questions" in system:
            return {"strategy": "bm25", "reason": "has identifier"}, LLMResponse("", 10, 5, 0.01)
        if "whether retrieved passages" in system:
            self.grades += 1
            return {"sufficient": self.grades >= 2, "reason": "missing details"}, LLMResponse("", 10, 5, 0.01)
        return {"query": "rewritten query"}, LLMResponse("", 10, 5, 0.01)


def test_agent_retries_then_accepts():
    retrievers = {n: FakeRetriever(n) for n in ("bm25", "vector", "hybrid")}
    agent = AgenticRetriever(FakeLLM(), retrievers, top_k=3, max_attempts=3)
    docs, info = agent.search_with_trace("What does response_model do?")
    steps = [t["step"] for t in info["trace"]]
    assert steps == ["route", "retrieve", "grade", "rewrite", "route", "retrieve", "grade", "finalize"]
    assert info["attempts"] == 2 and info["sufficient"] is True
    assert info["llm_calls"] == 5 and len(docs) == 3


def test_agent_fuses_when_never_sufficient():
    class NeverLLM(FakeLLM):
        def chat_json(self, messages):
            data, r = super().chat_json(messages)
            if "sufficient" in data:
                data["sufficient"] = False
            return data, r

    agent = AgenticRetriever(NeverLLM(), {n: FakeRetriever(n) for n in ("bm25", "vector", "hybrid")},
                             top_k=3, max_attempts=2)
    docs, info = agent.search_with_trace("q")
    assert info["attempts"] == 2 and info["trace"][-1]["mode"] == "fused_all_attempts"
    assert len(docs) == 3
