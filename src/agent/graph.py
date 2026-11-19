"""Agentic retrieval as a LangGraph state machine.

    route ──► retrieve ──► grade ──(sufficient or out of attempts)──► finalize ──► END
      ▲                      │
      └────── rewrite ◄──────┘ (not sufficient)

- route:    LLM picks bm25 / vector / hybrid for the current query (rule-based fallback).
- retrieve: run the chosen retriever.
- grade:    LLM checks whether the passages can answer the ORIGINAL question.
- rewrite:  LLM rewrites the query using the grader's reason, then routes again.
- finalize: use the passing attempt, or RRF-fuse every attempt if none passed.
"""
from __future__ import annotations

import operator
import re
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from src.generation.llm import as_bool
from src.generation.prompts import (GRADER_SYSTEM, GRADER_USER, REWRITER_SYSTEM, REWRITER_USER,
                                    ROUTER_SYSTEM, ROUTER_USER, fill)
from src.retrievers.base import BaseRetriever, RetrievedChunk
from src.retrievers.hybrid import rrf_fuse

STRATEGIES = ("bm25", "vector", "hybrid")
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
    sufficient: bool
    grade_reason: str
    trace: Annotated[list[dict], operator.add]
    llm_calls: Annotated[int, operator.add]
    prompt_tokens: Annotated[int, operator.add]
    completion_tokens: Annotated[int, operator.add]


def heuristic_route(query: str) -> str:
    if CODE_HINT_RE.search(query):
        return "bm25"
    if CONCEPTUAL_RE.search(query.strip()):
        return "vector"
    return "hybrid"


def _usage(resp) -> dict:
    return {"llm_calls": 1, "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens}


def build_agent(llm, retrievers: dict[str, BaseRetriever], top_k: int = 5,
                max_attempts: int = 3, rrf_k: int = 60, snippet_chars: int = 700,
                domain: str = "the provided documents"):
    router_system = fill(ROUTER_SYSTEM, domain=domain)
    grader_system = fill(GRADER_SYSTEM, domain=domain)
    rewriter_system = fill(REWRITER_SYSTEM, domain=domain)

    def route(state: AgentState) -> dict:
        tried = state.get("tried", [])
        tried_note = f"Already tried (with poor results): {', '.join(tried)}." if tried else ""
        data, resp = llm.chat_json([
            {"role": "system", "content": router_system},
            {"role": "user", "content": fill(ROUTER_USER, query=state["query"], tried=tried_note)},
        ])
        strategy = str(data.get("strategy", "")).strip().lower()
        reason = data.get("reason", "")
        if strategy not in retrievers:
            strategy, reason = heuristic_route(state["query"]), "fallback: rule-based router"
        return {"strategy": strategy, "tried": [strategy], **_usage(resp),
                "trace": [{"step": "route", "query": state["query"], "strategy": strategy,
                           "reason": reason, "latency_s": round(resp.latency_s, 3)}]}

    def retrieve(state: AgentState) -> dict:
        start = time.perf_counter()
        docs = retrievers[state["strategy"]].search(state["query"], k=top_k)
        return {"docs": docs, "history": [docs],
                "trace": [{"step": "retrieve", "strategy": state["strategy"],
                           "chunk_ids": [d.chunk_id for d in docs],
                           "latency_s": round(time.perf_counter() - start, 3)}]}

    def grade(state: AgentState) -> dict:
        passages = "\n\n".join(f"[{i}] {d.text[:snippet_chars]}" for i, d in enumerate(state["docs"], 1))
        data, resp = llm.chat_json([
            {"role": "system", "content": grader_system},
            {"role": "user", "content": fill(GRADER_USER, question=state["question"],
                                             passages=passages or "(none)")},
        ])
        sufficient = bool(state["docs"]) and as_bool(data.get("sufficient", False))
        reason = data.get("reason", "")
        return {"sufficient": sufficient, "grade_reason": reason,
                "attempts": state.get("attempts", 0) + 1, **_usage(resp),
                "trace": [{"step": "grade", "sufficient": sufficient, "reason": reason,
                           "latency_s": round(resp.latency_s, 3)}]}

    def rewrite(state: AgentState) -> dict:
        data, resp = llm.chat_json([
            {"role": "system", "content": rewriter_system},
            {"role": "user", "content": fill(
                REWRITER_USER, question=state["question"], query=state["query"],
                reason=state.get("grade_reason", ""))},
        ])
        new_query = str(data.get("query") or "").strip() or state["question"]
        return {"query": new_query, **_usage(resp),
                "trace": [{"step": "rewrite", "query": new_query, "latency_s": round(resp.latency_s, 3)}]}

    def finalize(state: AgentState) -> dict:
        if state.get("sufficient"):
            return {"trace": [{"step": "finalize", "mode": "accepted", "attempts": state["attempts"]}]}
        docs = rrf_fuse(state.get("history", []), k=rrf_k)[:top_k]
        return {"docs": docs,
                "trace": [{"step": "finalize", "mode": "fused_all_attempts", "attempts": state["attempts"]}]}

    def decide(state: AgentState) -> str:
        if state.get("sufficient") or state.get("attempts", 0) >= max_attempts:
            return "finalize"
        return "rewrite"

    g = StateGraph(AgentState)
    g.add_node("route", route)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("finalize", finalize)
    g.set_entry_point("route")
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", decide, {"rewrite": "rewrite", "finalize": "finalize"})
    g.add_edge("rewrite", "route")
    g.add_edge("finalize", END)
    return g.compile()


class AgenticRetriever(BaseRetriever):
    """Wraps the LangGraph agent behind the common retriever interface."""
    name = "agentic"

    def __init__(self, llm, retrievers: dict[str, BaseRetriever], top_k: int = 5,
                 max_attempts: int = 3, rrf_k: int = 60, domain: str = "the provided documents"):
        self.top_k = top_k
        self.graph = build_agent(llm, retrievers, top_k=top_k, max_attempts=max_attempts,
                                 rrf_k=rrf_k, domain=domain)

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
