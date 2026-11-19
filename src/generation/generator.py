"""Shared answer generator: every pipeline uses the same prompt so only retrieval differs."""
from __future__ import annotations

from src.generation.llm import LLMResponse, OllamaClient
from src.generation.prompts import ANSWER_SYSTEM, ANSWER_USER, fill
from src.retrievers.base import RetrievedChunk


def format_context(chunks: list[RetrievedChunk], max_chars: int = 2000) -> str:
    return "\n\n".join(f"[{i}] {c.text[:max_chars]}" for i, c in enumerate(chunks, start=1))


class AnswerGenerator:
    def __init__(self, llm: OllamaClient, domain: str = "the provided documents"):
        self.llm = llm
        self.domain = domain

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> LLMResponse:
        messages = [
            {"role": "system", "content": fill(ANSWER_SYSTEM, domain=self.domain)},
            {"role": "user", "content": fill(ANSWER_USER, context=format_context(chunks), question=question)},
        ]
        return self.llm.chat(messages)
