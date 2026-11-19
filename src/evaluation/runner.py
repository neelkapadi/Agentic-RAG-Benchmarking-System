"""Benchmark runner: every pipeline x every gold question -> metrics -> summary.

Usage:
  python -m src.evaluation.runner                          # all pipelines, full QA set
  python -m src.evaluation.runner --limit 10 --no-judge    # quick smoke run
  python -m src.evaluation.runner --retrieval-only         # no LLM needed (except agentic)
  python -m src.evaluation.runner --run-name myrun --resume
  python -m src.evaluation.runner --corpus papers           # benchmark your own corpus
  python -m src.evaluation.runner --run-name myrun --judge-only   # re-score with the configured judge
Two passes: (1) generate all answers with the generator model, (2) judge them all with the
judge model, so a small GPU never has to swap models between every question.
Outputs go to results/<corpus>/<run-name>/ :
  records.jsonl, summary.csv, summary_by_type.csv, report.md, meta.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import load_config, resolve
from src.corpus import DEFAULT_CORPUS, apply_corpus
from src.evaluation.llm_judge import Judge
from src.evaluation.report import build_report
from src.evaluation.retrieval_metrics import retrieval_scores
from src.pipeline import build_pipelines, domain_of, make_llm


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


def _write_records(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def generation_pass(qa: list[dict], pipelines: dict, records_path: Path, done: set, generate: bool, k: int) -> list[str]:
    """Pass 1: retrieve + answer every (question, pipeline). Uses only the generator model."""
    failed: list[str] = []
    todo = [(q, n) for q in qa for n in pipelines if (q["id"], n) not in done]
    with open(records_path, "a", encoding="utf-8") as f:
        for q, name in tqdm(todo, desc="pass 1/2 generate"):
            try:
                res = pipelines[name].run(q["question"], generate=generate)
            except Exception as e:  # skip; a later --resume run retries it
                failed.append(f"{q['id']}/{name}: {e}")
                tqdm.write(f"  ! {q['id']}/{name} failed: {e}")
                continue
            rec = {"qid": q["id"], "type": q.get("type", "unknown"), "pipeline": name,
                   "question": q["question"], "reference": q.get("answer", ""),
                   "gold_chunk_ids": q["gold_chunk_ids"], **res.to_dict()}
            rec.update(retrieval_scores([c.chunk_id for c in res.contexts], [c.doc_id for c in res.contexts],
                                        set(q["gold_chunk_ids"]), set(q["gold_doc_ids"]), k))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    return failed


def judge_pass(records_path: Path, judge: Judge, judge_name: str) -> int:
    """Pass 2: score answers with the judge model. Records judged by a different model are
    re-judged, so one run never mixes judges. Only the judge model is loaded during this pass."""
    from src.retrievers.base import RetrievedChunk

    records = [json.loads(line) for line in open(records_path, encoding="utf-8")]
    todo = [i for i, r in enumerate(records) if r.get("answer") and r.get("judge_model") != judge_name]
    for n, i in enumerate(tqdm(todo, desc="pass 2/2 judge"), start=1):
        r = records[i]
        for key in ("faithfulness", "claims", "answer_relevance", "correctness", "judge_reason", "judge_error"):
            r.pop(key, None)
        contexts = [RetrievedChunk(**c) for c in r["contexts"]]
        try:
            r.update(judge.evaluate(r["question"], r["answer"], r.get("reference", ""), contexts))
            r["judge_model"] = judge_name
        except Exception as e:  # keep retrieval metrics; judge scores stay empty, retried on --resume
            r["judge_error"] = str(e)
            tqdm.write(f"  ! judge failed on {r['qid']}/{r['pipeline']}: {e}")
        if n % 10 == 0 or n == len(todo):
            _write_records(records_path, records)
    return len(todo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--pipelines", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run (same --run-name)")
    ap.add_argument("--no-judge", action="store_true", help="skip LLM-judge metrics")
    ap.add_argument("--judge-only", action="store_true",
                    help="only (re)judge an existing run with the configured judge model")
    ap.add_argument("--retrieval-only", action="store_true", help="skip answer generation and judging")
    args = ap.parse_args()

    cfg = apply_corpus(load_config(args.config), args.corpus)
    k = cfg["retrieval"]["top_k"]
    names = args.pipelines or cfg["eval"]["pipelines"]
    generate = not args.retrieval_only
    use_judge = generate and cfg["eval"].get("judge", True) and not args.no_judge
    judge_name = cfg["llm"]["judge_model"]

    if args.judge_only and not args.run_name:
        ap.error("--judge-only needs --run-name of an existing run")
    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = resolve(cfg["eval"]["results_dir"]) / run_name
    records_path = out_dir / "records.jsonl"

    if not args.judge_only:
        llm = make_llm(cfg)
        if generate or "agentic" in names:
            llm.check()
        qa = load_qa(resolve(cfg["eval"]["qa_path"]), args.limit)
        pipelines = build_pipelines(cfg, names, llm=llm, with_generator=generate)
        out_dir.mkdir(parents=True, exist_ok=True)
        done = _done_keys(records_path) if args.resume else set()
        if not args.resume and records_path.exists():
            records_path.unlink()
        (out_dir / "config.json").write_text(json.dumps({"config": cfg, "args": vars(args)}, indent=2))
        failed = generation_pass(qa, pipelines, records_path, done, generate, k)
        if failed:
            print(f"\n{len(failed)} runs failed; rerun with --resume --run-name {run_name} to retry them.")
    elif not records_path.exists():
        raise FileNotFoundError(records_path)

    if use_judge:
        judge_llm = make_llm(cfg, judge=True)
        judge_llm.check()
        n = judge_pass(records_path, Judge(judge_llm, domain_of(cfg)), judge_name)
        print(f"Judged {n} answers with {judge_name}.")

    records = [json.loads(line) for line in open(records_path, encoding="utf-8")]
    summary, by_type = summarize(records, k)
    summary.to_csv(out_dir / "summary.csv")
    if not by_type.empty:
        by_type.to_csv(out_dir / "summary_by_type.csv")
    judges = sorted({r["judge_model"] for r in records if r.get("judge_model")})
    meta = {"run": run_name, "corpus": cfg["corpus"], "domain": domain_of(cfg),
            "model": cfg["llm"]["model"] if generate or "agentic" in names else "–",
            "judge_model": ", ".join(judges) or "–",
            "embedding_model": cfg["embeddings"]["model"]}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "report.md").write_text(build_report(records, summary, by_type, k, meta), encoding="utf-8")
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.precision", 3):
        print(summary)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
