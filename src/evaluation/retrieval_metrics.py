"""Ranking metrics against gold (relevant) ids. `retrieved` is an ordered list of ids."""
from __future__ import annotations

import math


def hit_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    return float(any(r in gold for r in retrieved[:k]))


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(retrieved[:k]) & gold) / len(gold)


def precision_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for r in retrieved[:k] if r in gold) / k


def mrr(retrieved: list[str], gold: set[str]) -> float:
    for rank, r in enumerate(retrieved, start=1):
        if r in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(retrieved[:k]) if r in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


def _dedupe(ids: list[str]) -> list[str]:
    return list(dict.fromkeys(ids))


def retrieval_scores(chunk_ids: list[str], doc_ids: list[str], gold_chunks: set[str],
                     gold_docs: set[str], k: int) -> dict[str, float]:
    """Chunk-level metrics are strict; doc-level metrics accept any chunk from the right page."""
    docs_ranked = _dedupe(doc_ids)
    return {
        f"hit@{k}": hit_at_k(chunk_ids, gold_chunks, k),
        f"recall@{k}": recall_at_k(chunk_ids, gold_chunks, k),
        f"precision@{k}": precision_at_k(chunk_ids, gold_chunks, k),
        "mrr": mrr(chunk_ids, gold_chunks),
        f"ndcg@{k}": ndcg_at_k(chunk_ids, gold_chunks, k),
        f"doc_hit@{k}": hit_at_k(docs_ranked, gold_docs, k),
        "doc_mrr": mrr(docs_ranked, gold_docs),
    }
