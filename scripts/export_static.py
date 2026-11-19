"""Export published benchmark runs to a self-contained static website (free Hugging Face Static Space).

Usage:  python -m scripts.export_static [--corpus fastapi] [--out site]
Writes site/index.html (data embedded, no server needed) and site/README.md (Space config).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.config import ROOT, load_config
from src.evaluation.report import ORDER, findings
from src.evaluation.runner import summarize

REPO_URL = "https://github.com/neelkapadi/Agentic-RAG-Benchmarking-System"
CONTEXT_CHARS = 700


def _clean(v):
    if isinstance(v, float) and pd.isna(v):
        return None
    if hasattr(v, "item"):
        return v.item()
    return v


def export_run(run_dir: Path, k: int) -> dict:
    records = [json.loads(line) for line in open(run_dir / "records.jsonl", encoding="utf-8")]
    meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").exists() else {}
    summary, by_type = summarize(records, k)
    summary = summary.reindex([p for p in ORDER if p in summary.index])

    agent_recs = [r for r in records if r["pipeline"] == "agentic" and (r.get("agent") or {}).get("trace")]
    agent = {}
    if agent_recs:
        first = Counter((r["agent"].get("strategies") or ["?"])[0] for r in agent_recs)
        routing = Counter((r.get("type", "?"), (r["agent"].get("strategies") or ["?"])[0]) for r in agent_recs)
        agent = {
            "first_strategy": dict(first),
            "avg_attempts": sum(r["agent"].get("attempts") or 0 for r in agent_recs) / len(agent_recs),
            "accepted": sum(1 for r in agent_recs if r["agent"].get("sufficient")),
            "n": len(agent_recs),
            "routing": {t: {s: routing.get((t, s), 0) for s in ORDER[:3]} for t in sorted({t for t, _ in routing})},
        }

    questions: dict[str, dict] = {}
    for r in records:
        q = questions.setdefault(r["qid"], {"qid": r["qid"], "type": r.get("type"), "question": r["question"],
                                            "reference": r.get("reference", ""),
                                            "gold": r["gold_chunk_ids"], "results": {}})
        q["results"][r["pipeline"]] = {
            "answer": r.get("answer", ""),
            "hit": r.get(f"hit@{k}"), "mrr": r.get("mrr"),
            "faithfulness": r.get("faithfulness"), "correctness": r.get("correctness"),
            "relevance": r.get("answer_relevance"),
            "total_s": r["timings"]["total_s"], "llm_calls": r["tokens"]["llm_calls"],
            "contexts": [{"id": c["chunk_id"], "text": c["text"][:CONTEXT_CHARS],
                          "gold": c["chunk_id"] in r["gold_chunk_ids"]} for c in r["contexts"]],
            "trace": (r.get("agent") or {}).get("trace") or [],
        }

    hit = f"hit@{k}"
    return {
        "name": run_dir.name,
        "meta": meta,
        "summary": {p: {c: _clean(v) for c, v in row.items()} for p, row in summary.iterrows()},
        "by_type": ({t: {p: _clean(by_type.loc[(t, p), hit]) for p in ORDER if (t, p) in by_type.index}
                     for t in by_type.index.get_level_values(0).unique()} if not by_type.empty else {}),
        "findings": findings(summary, k),
        "agent": agent,
        "questions": sorted(questions.values(), key=lambda q: q["qid"]),
    }


SPACE_README = """---
title: Agentic RAG Benchmark Report
emoji: 📊
colorFrom: indigo
colorTo: green
sdk: static
app_file: index.html
pinned: false
short_description: BM25 vs vector vs hybrid vs agentic RAG on the FastAPI docs
---

Interactive benchmark report for the [Agentic RAG Benchmarking System]({repo}).
Generated with `python -m scripts.export_static`.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="fastapi")
    ap.add_argument("--out", default="site")
    args = ap.parse_args()
    k = load_config()["retrieval"]["top_k"]

    run_dirs = sorted((ROOT / "results" / args.corpus).glob("*/records.jsonl"))
    # Main run first, comparison runs after.
    runs = [export_run(p.parent, k) for p in sorted(run_dirs, key=lambda p: p.parent.name.startswith("compare"))]
    if not runs:
        raise SystemExit(f"No runs in results/{args.corpus}/")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    template = (ROOT / "scripts" / "static_template.html").read_text(encoding="utf-8")
    data = json.dumps({"k": k, "repo": REPO_URL, "runs": runs}, ensure_ascii=False)
    data = data.replace("</", "<\\/")  # safe inside <script>
    (out / "index.html").write_text(template.replace("/*__DATA__*/null", data), encoding="utf-8")
    (out / "README.md").write_text(SPACE_README.format(repo=REPO_URL), encoding="utf-8")
    size_kb = (out / "index.html").stat().st_size / 1024
    print(f"Wrote {out / 'index.html'} ({size_kb:.0f} KB, {len(runs)} runs) and {out / 'README.md'}")


if __name__ == "__main__":
    main()
