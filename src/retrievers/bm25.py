"""Keyword retrieval with BM25 (Okapi)."""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from src.ingest.store import ChunkStore
from src.retrievers.base import BaseRetriever, RetrievedChunk

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
STOPWORDS = set("""
a an the and or but if of to in on at by for with from as is are was were be been being
it its this that these those i you he she we they them my your our their what which who
whom how why when where do does did can could should would will shall may might must not
no so than too very just also into about over under then there here such only own same
""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; snake_case identifiers are kept whole AND split into parts."""
    tokens = []
    for tok in TOKEN_RE.findall(text.lower()):
        if tok in STOPWORDS:
            continue
        tokens.append(tok)
        if "_" in tok:
            tokens.extend(p for p in tok.split("_") if p and p not in STOPWORDS)
    return tokens


class BM25Retriever(BaseRetriever):
    name = "bm25"

    def __init__(self, store: ChunkStore, k1: float = 1.5, b: float = 0.75):
        self.store = store
        self.bm25 = BM25Okapi([tokenize(c.text) for c in store.chunks], k1=k1, b=b)

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top = np.argsort(scores)[::-1][:k]
        return [self.store.result(int(i), float(scores[i])) for i in top if scores[i] > 0]
