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

ROUTER_SYSTEM = """You route questions about {domain} to the best search strategy.
Strategies:
- "bm25": keyword search. Best when the question contains exact identifiers or rare terms: class,
  function or parameter names, acronyms, error messages, codes, numbers, quoted phrases or named
  methods/datasets.
- "vector": semantic search. Best for conceptual or paraphrased questions that do not use the exact
  wording of the documents (e.g. "how do I stop users from sending bad data?").
- "hybrid": both combined. Best when the question mixes exact terms with a conceptual ask, or is unclear.
Reply with JSON: {"strategy": "bm25" | "vector" | "hybrid", "reason": "<one short sentence>"}"""

ROUTER_USER = """Question: {query}
{tried}
JSON:"""

GRADER_SYSTEM = """You check whether retrieved passages from {domain} are enough to answer a question.
Reply with JSON: {"sufficient": true | false, "reason": "<one short sentence>"}
"sufficient" is true only if at least one passage directly contains the information needed."""

GRADER_USER = """Question: {question}

Passages:
{passages}

JSON:"""

REWRITER_SYSTEM = """You improve search queries over {domain}.
The previous query did not retrieve useful passages. Write a better query:
use the documents' own terminology (technical names, feature or method names) and drop filler words.
Reply with JSON: {"query": "<new search query>"}"""

REWRITER_USER = """Original question: {question}
Previous query: {query}
Why it failed: {reason}
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
