"""Hybrid retrieval: fuse BM25 and vector rankings with Reciprocal Rank Fusion."""
from __future__ import annotations

from dataclasses import replace

from src.retrievers.base import BaseRetriever, RetrievedChunk


def rrf_fuse(result_lists: list[list[RetrievedChunk]], k: int = 60,
             weights: list[float] | None = None) -> list[RetrievedChunk]:
    """score(d) = sum_i w_i / (k + rank_i(d)); rank starts at 1."""
    scores: dict[str, float] = {}
    first_seen: dict[str, RetrievedChunk] = {}
    for li, results in enumerate(result_lists):
        w = weights[li] if weights else 1.0
        for rank, r in enumerate(results, start=1):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + w / (k + rank)
            first_seen.setdefault(r.chunk_id, r)
    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [replace(first_seen[cid], score=scores[cid]) for cid in ordered]


class HybridRetriever(BaseRetriever):
    name = "hybrid"

    def __init__(self, bm25: BaseRetriever, vector: BaseRetriever, rrf_k: int = 60, candidates: int = 20):
        self.bm25, self.vector = bm25, vector
        self.rrf_k, self.candidates = rrf_k, candidates

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        n = max(k, self.candidates)
        fused = rrf_fuse([self.bm25.search(query, n), self.vector.search(query, n)], k=self.rrf_k)
        return fused[:k]
