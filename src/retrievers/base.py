"""Common retriever interface so every pipeline is benchmarked the same way."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    title: str = ""
    section: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class BaseRetriever(ABC):
    name: str = "base"

    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        ...
