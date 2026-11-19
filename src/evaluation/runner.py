"""Benchmark runner: every pipeline x every gold question -> metrics -> summary.

Usage:
  python -m src.evaluation.runner                          # all pipelines, full QA set
  python -m src.evaluation.runner --limit 10 --no-judge    # quick smoke run
  python -m src.evaluation.runner --retrieval-only         # no LLM needed (except agentic)
  python -m src.evaluation.runner --run-name myrun --resume
Outputs go to results/<run-name>/ : records.jsonl, summary.csv, summary_by_type.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import load_config, resolve
from src.evaluation.llm_judge import Judge
from src.evaluation.retrieval_metrics import retrieval_scores
from src.pipeline import build_pipelines, make_llm


def load_qa(path: Path, limit: int | None = None) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m scripts.generate_qa")
    with open(path, encoding="utf-8") as f:
        qa = [json.loads(line) for line in f if line.strip()]
    qa = [q for q in qa if not q.get("exclude")]
    return qa[:limit] if limit else qa


def _done_keys(records_path: Path) -> set[tuple[str, str]]:
    if not records_path.exists():
        return set()
    with open(records_path, encoding="utf-8") as f:
        return {(r["qid"], r["pipeline"]) for r in map(json.loads, f)}


def summarize(records: list[dict], k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.json_normalize(records)
    metric_cols = [c for c in (f"hit@{k}", f"recall@{k}", f"precision@{k}", "mrr", f"ndcg@{k}",
                               f"doc_hit@{k}", "doc_mrr", "faithfulness", "answer_relevance",
                               "correctness", "abstained") if c in df.columns]
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def agg(g: pd.DataFrame) -> pd.Series:
        row = {c: g[c].mean() for c in metric_cols}
        for stage in ("retrieval_s", "generation_s", "total_s"):
            col = f"timings.{stage}"
            if col in g:
                row[f"{stage}_p50"] = g[col].quantile(0.5)
                row[f"{stage}_p95"] = g[col].quantile(0.95)
        for col in ("tokens.prompt", "tokens.completion", "tokens.llm_calls", "agent.attempts"):
            if col in g:
                row[col.split(".")[1] + "_mean"] = pd.to_numeric(g[col], errors="coerce").mean()
        row["n"] = len(g)
        return pd.Series(row)

    summary = df.groupby("pipeline").apply(agg, include_groups=False)
    by_type = df.groupby(["type", "pipeline"]).apply(agg, include_groups=False) if "type" in df else pd.DataFrame()
    return summary, by_type


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--pipelines", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-judge", action="store_true", help="skip LLM-judge metrics")
    ap.add_argument("--retrieval-only", action="store_true", help="skip answer generation and judging")
    args = ap.parse_args()

    cfg = load_config(args.config)
    k = cfg["retrieval"]["top_k"]
    names = args.pipelines or cfg["eval"]["pipelines"]
    generate = not args.retrieval_only
    use_judge = generate and cfg["eval"].get("judge", True) and not args.no_judge

    llm = make_llm(cfg)
    if generate or "agentic" in names:
        llm.check()
    judge = None
    if use_judge:
        judge_llm = make_llm(cfg, judge=True)
        judge_llm.check()
        judge = Judge(judge_llm)

    qa = load_qa(resolve(cfg["eval"]["qa_path"]), args.limit)
    pipelines = build_pipelines(cfg, names, llm=llm, with_generator=generate)

    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = resolve(cfg["eval"]["results_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    done = _done_keys(records_path) if args.resume else set()
    if not args.resume and records_path.exists():
        records_path.unlink()
    (out_dir / "config.json").write_text(json.dumps({"config": cfg, "args": vars(args)}, indent=2))

    with open(records_path, "a", encoding="utf-8") as f:
        for q in tqdm(qa, desc="questions"):
            for name, pipe in pipelines.items():
                if (q["id"], name) in done:
                    continue
                res = pipe.run(q["question"], generate=generate)
                rec = {"qid": q["id"], "type": q.get("type", "unknown"), "pipeline": name,
                       "question": q["question"], "reference": q.get("answer", ""),
                       "gold_chunk_ids": q["gold_chunk_ids"], **res.to_dict()}
                rec.update(retrieval_scores([c.chunk_id for c in res.contexts], [c.doc_id for c in res.contexts],
                                            set(q["gold_chunk_ids"]), set(q["gold_doc_ids"]), k))
                if judge:
                    rec.update(judge.evaluate(q["question"], res.answer, q.get("answer", ""), res.contexts))
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

    with open(records_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    summary, by_type = summarize(records, k)
    summary.to_csv(out_dir / "summary.csv")
    if not by_type.empty:
        by_type.to_csv(out_dir / "summary_by_type.csv")
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.precision", 3):
        print(summary)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
