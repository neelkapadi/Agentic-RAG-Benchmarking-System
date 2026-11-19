"""LLM-as-a-judge metrics: faithfulness, answer relevance and correctness."""
from __future__ import annotations

import re

from src.generation.generator import format_context
from src.generation.llm import OllamaClient, as_bool
from src.generation.prompts import (FAITHFULNESS_SYSTEM, FAITHFULNESS_USER, RELEVANCE_SYSTEM,
                                    RELEVANCE_USER, fill)
from src.retrievers.base import RetrievedChunk

ABSTAIN_RE = re.compile(r"i don'?t know|does not contain|doesn'?t contain|not (?:mentioned|provided)|no information", re.I)


def is_abstention(answer: str) -> bool:
    return bool(ABSTAIN_RE.search(answer)) and len(answer) < 300


def _score_1_5(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return (min(max(v, 1.0), 5.0) - 1.0) / 4.0  # normalise to 0..1


class Judge:
    def __init__(self, llm: OllamaClient, domain: str = "the provided documents"):
        self.llm = llm
        self.domain = domain

    def faithfulness(self, answer: str, contexts: list[RetrievedChunk]) -> dict:
        """Share of the answer's claims that are supported by the retrieved context."""
        if not answer.strip() or is_abstention(answer):
            return {"faithfulness": None, "claims": []}
        data, _ = self.llm.chat_json([
            {"role": "system", "content": FAITHFULNESS_SYSTEM},
            {"role": "user", "content": fill(FAITHFULNESS_USER, context=format_context(contexts), answer=answer)},
        ])
        claims = [c for c in data.get("claims", []) if isinstance(c, dict) and c.get("claim")]
        if not claims:
            return {"faithfulness": None, "claims": []}
        supported = sum(as_bool(c.get("supported")) for c in claims)
        return {"faithfulness": supported / len(claims), "claims": claims}

    def relevance(self, question: str, answer: str, reference: str) -> dict:
        data, _ = self.llm.chat_json([
            {"role": "system", "content": fill(RELEVANCE_SYSTEM, domain=self.domain)},
            {"role": "user", "content": fill(RELEVANCE_USER, question=question, reference=reference, answer=answer)},
        ])
        return {"answer_relevance": _score_1_5(data.get("relevance")),
                "correctness": _score_1_5(data.get("correctness")),
                "judge_reason": data.get("reason", "")}

    def evaluate(self, question: str, answer: str, reference: str, contexts: list[RetrievedChunk]) -> dict:
        out = {"abstained": is_abstention(answer)}
        out.update(self.faithfulness(answer, contexts))
        out.update(self.relevance(question, answer, reference))
        return out
