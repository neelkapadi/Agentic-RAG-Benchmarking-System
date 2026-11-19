"""In-memory chunk store backed by a JSONL file."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.ingest.chunker import Chunk
from src.retrievers.base import RetrievedChunk


class ChunkStore:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}

    @classmethod
    def load(cls, path: Path) -> "ChunkStore":
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: python -m scripts.build_index")
        with open(path, encoding="utf-8") as f:
            return cls([Chunk(**json.loads(line)) for line in f if line.strip()])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    def fingerprint(self) -> str:
        h = hashlib.sha1()
        for c in self.chunks:
            h.update(c.chunk_id.encode())
            h.update(c.text.encode())
        return h.hexdigest()

    def result(self, idx: int, score: float) -> RetrievedChunk:
        c = self.chunks[idx]
        return RetrievedChunk(chunk_id=c.chunk_id, doc_id=c.doc_id, text=c.text,
                              score=score, title=c.title, section=c.section)

    def __len__(self) -> int:
        return len(self.chunks)
