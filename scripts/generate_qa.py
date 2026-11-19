"""Generate a gold Q&A set from the chunked docs with the local LLM.

Each question is written from ONE chunk, so that chunk is its gold retrieval target.
Half the questions are "keyword" (use exact API names -> favours BM25) and half are
"conceptual" (paraphrased -> favours vector search). This is what makes the
benchmark interesting for the agentic router.

Usage:  python -m scripts.generate_qa --n 100
Then review data/eval/qa_gold.jsonl by hand: fix bad questions or set "exclude": true.
"""
from __future__ import annotations

import argparse
import json
import random
import re

from tqdm import tqdm

from src.config import load_config, resolve
from src.generation.prompts import QA_GEN_SYSTEM, QA_GEN_USER, QA_STYLES
from src.ingest.store import ChunkStore
from src.pipeline import make_llm

BAD_PHRASES = re.compile(r"\b(the|this|that|given) (passage|text|section|example|context|document)\b", re.I)
PREFERRED_FOLDERS = ("tutorial/", "advanced/", "how-to/", "deployment/")


def eligible(chunk) -> bool:
    body = chunk.text.split("\n\n", 1)[-1]
    code_chars = sum(len(m) for m in re.findall(r"```.*?```", body, re.S))
    prose_chars = len(body) - code_chars
    return prose_chars >= 300 and chunk.doc_id != "index.md"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = ChunkStore.load(resolve(cfg["data"]["chunks_path"]))
    llm = make_llm(cfg, judge=True)
    llm.check()

    rng = random.Random(args.seed)
    pool = [c for c in store.chunks if eligible(c)]
    preferred = [c for c in pool if c.doc_id.startswith(PREFERRED_FOLDERS)]
    others = [c for c in pool if not c.doc_id.startswith(PREFERRED_FOLDERS)]
    rng.shuffle(preferred)
    rng.shuffle(others)
    # ~80% from tutorial/advanced/how-to/deployment, the rest from top-level pages.
    candidates = preferred[: int(args.n * 0.8 * 1.5)] + others[: int(args.n * 0.2 * 1.5)]
    rng.shuffle(candidates)

    out_path = resolve(cfg["eval"]["qa_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    items = []
    with tqdm(total=args.n, desc="generating") as bar:
        for chunk in candidates:
            if len(items) >= args.n:
                break
            qtype = "keyword" if len(items) % 2 == 0 else "conceptual"
            data, _ = llm.chat_json([
                {"role": "system", "content": QA_GEN_SYSTEM.format(style=QA_STYLES[qtype])},
                {"role": "user", "content": QA_GEN_USER.format(passage=chunk.text[:2500])},
            ])
            q, a = str(data.get("question", "")).strip(), str(data.get("answer", "")).strip()
            if len(q) < 15 or len(a) < 10 or BAD_PHRASES.search(q):
                continue
            items.append({"id": f"q{len(items) + 1:03d}", "type": qtype, "question": q, "answer": a,
                          "gold_chunk_ids": [chunk.chunk_id], "gold_doc_ids": [chunk.doc_id],
                          "source_section": chunk.section, "reviewed": False})
            bar.update(1)

    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"Wrote {len(items)} questions to {out_path}. Review them before benchmarking!")


if __name__ == "__main__":
    main()
