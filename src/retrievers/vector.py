"""Dense retrieval: sentence-transformers embeddings + FAISS inner-product index."""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.ingest.store import ChunkStore
from src.retrievers.base import BaseRetriever, RetrievedChunk


class VectorRetriever(BaseRetriever):
    name = "vector"

    def __init__(self, store: ChunkStore, model_name: str, index_dir: Path,
                 query_prefix: str = "", batch_size: int = 32):
        self.store = store
        self.query_prefix = query_prefix
        self.model = SentenceTransformer(model_name)
        embeddings = self._load_or_build(model_name, index_dir, batch_size)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine sim (vectors are normalized)
        self.index.add(embeddings)

    def _load_or_build(self, model_name: str, index_dir: Path, batch_size: int) -> np.ndarray:
        emb_path, meta_path = index_dir / "embeddings.npy", index_dir / "meta.json"
        meta = {"model": model_name, "fingerprint": self.store.fingerprint(), "n": len(self.store)}
        if emb_path.exists() and meta_path.exists():
            if json.loads(meta_path.read_text()) == meta:
                return np.load(emb_path)
        print(f"Embedding {len(self.store)} chunks with {model_name} ...")
        emb = self.model.encode([c.text for c in self.store.chunks], batch_size=batch_size,
                                normalize_embeddings=True, show_progress_bar=True)
        emb = np.asarray(emb, dtype="float32")
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, emb)
        meta_path.write_text(json.dumps(meta))
        return emb

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        q = self.model.encode([self.query_prefix + query], normalize_embeddings=True)
        scores, idxs = self.index.search(np.asarray(q, dtype="float32"), k)
        return [self.store.result(int(i), float(s)) for s, i in zip(scores[0], idxs[0]) if i >= 0]
