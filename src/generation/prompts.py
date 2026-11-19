"""All prompts in one place so they are easy to tune and compare.

Templates use {placeholders} filled by `fill()` (plain string replacement, so literal JSON
braces in the prompts need no escaping). {domain} describes the corpus, e.g.
"the FastAPI web framework documentation" or "the 'llm-papers' documents".
"""
import re

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def fill(template: str, **values) -> str:
    """Single-pass substitution: text inserted for one placeholder is never re-scanned."""
    return _PLACEHOLDER_RE.sub(lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), template)


ANSWER_SYSTEM = """You are an assistant that answers questions about {domain}.
Answer the question using ONLY the numbered context passages.
- Cite the passages you used like [1] or [2][3].
- Include a short code example or formula when the context has one that helps.
- If the context does not contain the answer, reply exactly: "I don't know based on the provided documents."
Be concise and precise."""

ANSWER_USER = """Context:
{context}

Question: {question}

Answer:"""

ROUTER_SYSTEM = """You choose how to weight keyword search vs semantic search for a question about {domain}.
Both are always used; you only choose which one to trust more.
- "balanced": the DEFAULT. Use it unless one of the cases below clearly applies.
- "keyword": the question contains exact identifiers or rare literal terms that must match: code names
  like `response_model` or `Depends`, error messages, status codes, CLI flags, config keys, acronyms.
- "semantic": the question is phrased in everyday words, describing a goal or problem without the
  documents' technical terms (e.g. "how do I stop users from sending bad data?").
Reply with JSON: {"strategy": "balanced" | "keyword" | "semantic", "reason": "<one short sentence>"}"""

ROUTER_USER = """Question: {query}
JSON:"""

GRADER_SYSTEM = """You check whether search results from {domain} are RELEVANT to a question.
Answer "relevant": true if at least one passage contains information that helps answer the question,
even if it is only part of the answer or the wording differs. Answer false only if the passages are
about something else.
Reply with JSON: {"relevant": true | false, "reason": "<one short sentence>"}"""

GRADER_USER = """Question: {question}

Passages:
{passages}

JSON:"""

EXPAND_SYSTEM = """You help search {domain}. The first search did not find relevant passages.
1. Write a short hypothetical passage (2-3 sentences) that would answer the question, in the style and
   vocabulary of technical documentation. It may be imperfect; it is only used for searching.
2. List 3-6 technical keywords or names the real answer is likely to contain.
Reply with JSON: {"passage": "...", "keywords": ["...", "..."]}"""

EXPAND_USER = """Question: {question}
JSON:"""

# ---------- evaluation ----------

FAITHFULNESS_SYSTEM = """You verify whether an answer is supported by context passages.
1. Break the answer into short atomic factual claims (ignore citations like [1]).
2. For each claim decide if the context supports it.
Reply with JSON: {"claims": [{"claim": "...", "supported": true | false}]}"""

FAITHFULNESS_USER = """Context:
{context}

Answer:
{answer}

JSON:"""

RELEVANCE_SYSTEM = """You grade an answer to a question about {domain}.
Scores are integers 1-5:
- "relevance": does the answer directly address the question? (1 = off-topic, 5 = fully on point)
- "correctness": does it agree with the reference answer? (1 = wrong/missing, 5 = fully correct)
Reply with JSON: {"relevance": <1-5>, "correctness": <1-5>, "reason": "<one short sentence>"}"""

RELEVANCE_USER = """Question: {question}

Reference answer: {reference}

Answer to grade: {answer}

JSON:"""

QA_GEN_SYSTEM = """You write benchmark questions for a search system over {domain}.
Given a passage, write ONE question a reader might ask that this passage answers,
plus a short reference answer (1-3 sentences) taken from the passage.
Rules:
- The question must stand alone: never say "the passage", "the text", "this section" or "the example".
- {style}
Reply with JSON: {"question": "...", "answer": "..."}"""

QA_STYLES = {
    "keyword": "Use the exact technical names that appear in the passage (names, terms, parameters, codes).",
    "conceptual": "Paraphrase: describe the goal or problem in plain words and AVOID the passage's exact identifiers and distinctive terms.",
}

QA_GEN_USER = """Passage:
{passage}

JSON:"""
