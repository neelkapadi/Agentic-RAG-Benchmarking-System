"""Ask one question through one or more pipelines from the terminal.

Usage:
  python -m scripts.ask "How do I declare an optional query parameter?" --pipelines agentic
  python -m scripts.ask "What does response_model do?" --pipelines bm25 vector hybrid agentic
  python -m scripts.ask "..." --no-generate     # show retrieval only
  python -m scripts.ask "What is the main contribution?" --corpus papers
"""
from __future__ import annotations

import argparse
import json

from src.config import load_config
from src.corpus import DEFAULT_CORPUS, apply_corpus
from src.pipeline import build_pipelines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--pipelines", nargs="+", default=["agentic"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--no-generate", action="store_true")
    args = ap.parse_args()

    cfg = apply_corpus(load_config(args.config), args.corpus)
    pipes = build_pipelines(cfg, args.pipelines, with_generator=not args.no_generate)
    for name, pipe in pipes.items():
        res = pipe.run(args.question, generate=not args.no_generate)
        print(f"\n===== {name}  ({res.timings['total_s']:.2f}s) =====")
        for i, c in enumerate(res.contexts, 1):
            print(f"  [{i}] {c.chunk_id}  score={c.score:.3f}")
        if res.agent.get("trace"):
            print("  agent trace:")
            for step in res.agent["trace"]:
                print("   ", json.dumps(step, ensure_ascii=False))
        if res.answer:
            print("\n" + res.answer)


if __name__ == "__main__":
    main()
