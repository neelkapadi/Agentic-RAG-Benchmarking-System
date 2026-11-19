"""Generate a gold Q&A set from a corpus's chunks with the local LLM.

Each question is written from ONE chunk, so that chunk is its gold retrieval target.
Half the questions are "keyword" (use exact names/terms -> favours BM25) and half are
"conceptual" (paraphrased -> favours vector search). This is what makes the
benchmark interesting for the agentic router.

Usage:
  python -m scripts.generate_qa --n 100                    # FastAPI corpus
  python -m scripts.generate_qa --corpus papers --n 30     # your own corpus
Then review the qa_gold.jsonl file by hand: fix bad questions or set "exclude": true.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict

from tqdm import tqdm

from src.config import load_config, resolve
from src.corpus import DEFAULT_CORPUS, apply_corpus
from src.generation.prompts import QA_GEN_SYSTEM, QA_GEN_USER, QA_STYLES, fill
from src.ingest.store import ChunkStore
from src.pipeline import domain_of, make_llm

BAD_PHRASES = re.compile(r"\b(the|this|that|given) (passage|text|section|example|context|document|paper)\b", re.I)
FASTAPI_PREFERRED = ("tutorial/", "advanced/", "how-to/", "deployment/")


def eligible(chunk) -> bool:
    body = chunk.text.split("\n\n", 1)[-1]
    code_chars = sum(len(m) for m in re.findall(r"```.*?```", body, re.S))
    return len(body) - code_chars >= 300 and chunk.doc_id != "index.md"


def pick_candidates(chunks: list, n: int, corpus: str, rng: random.Random) -> list:
    pool = [c for c in chunks if eligible(c)]
    if corpus == DEFAULT_CORPUS:
        # ~80% from tutorial/advanced/how-to/deployment, the rest from top-level pages.
        preferred = [c for c in pool if c.doc_id.startswith(FASTAPI_PREFERRED)]
        others = [c for c in pool if not c.doc_id.startswith(FASTAPI_PREFERRED)]
        rng.shuffle(preferred)
        rng.shuffle(others)
        out = preferred[: int(n * 0.8 * 1.5)] + others[: int(n * 0.2 * 1.5)]
        rng.shuffle(out)
        return out
    # Custom corpora: round-robin across documents so every file is covered.
    by_doc = defaultdict(list)
    for c in pool:
        by_doc[c.doc_id].append(c)
    for lst in by_doc.values():
        rng.shuffle(lst)
    out, lists = [], list(by_doc.values())
    while any(lists):
        for lst in lists:
            if lst:
                out.append(lst.pop())
    return out


def generate(cfg: dict, n: int, seed: int = 42, progress=None) -> int:
    store = ChunkStore.load(resolve(cfg["data"]["chunks_path"]))
    llm = make_llm(cfg, judge=True)
    llm.check()
    rng = random.Random(seed)
    candidates = pick_candidates(store.chunks, n, cfg.get("corpus", DEFAULT_CORPUS), rng)

    items = []
    with tqdm(total=n, desc="generating") as bar:
        for chunk in candidates:
            if len(items) >= n:
                break
            qtype = "keyword" if len(items) % 2 == 0 else "conceptual"
            data, _ = llm.chat_json([
                {"role": "system", "content": fill(QA_GEN_SYSTEM, domain=domain_of(cfg), style=QA_STYLES[qtype])},
                {"role": "user", "content": fill(QA_GEN_USER, passage=chunk.text[:2500])},
            ])
            q, a = str(data.get("question", "")).strip(), str(data.get("answer", "")).strip()
            if len(q) < 15 or len(a) < 10 or BAD_PHRASES.search(q):
                continue
            items.append({"id": f"q{len(items) + 1:03d}", "type": qtype, "question": q, "answer": a,
                          "gold_chunk_ids": [chunk.chunk_id], "gold_doc_ids": [chunk.doc_id],
                          "source_section": chunk.section, "reviewed": False})
            bar.update(1)
            if progress:
                progress(len(items), n)

    out_path = resolve(cfg["eval"]["qa_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"Wrote {len(items)} questions to {out_path}. Review them before benchmarking!")
    return len(items)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    generate(apply_corpus(load_config(args.config), args.corpus), args.n, args.seed)


if __name__ == "__main__":
    main()
