"""Turn benchmark records into a human-readable Markdown report (results/<corpus>/<run>/report.md)."""
from __future__ import annotations

import math
import time
from collections import Counter

import pandas as pd

ORDER = ["bm25", "vector", "hybrid", "agentic"]
BASELINES = ["bm25", "vector", "hybrid"]


def _fmt(v, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def md_table(df: pd.DataFrame, digits: int = 3, bold_max: list[str] | None = None,
             bold_min: list[str] | None = None) -> str:
    """Markdown table; optionally bolds the best value per column."""
    best = {}
    for col in bold_max or []:
        if col in df and df[col].notna().any():
            best[col] = df[col].max()
    for col in bold_min or []:
        if col in df and df[col].notna().any():
            best[col] = df[col].min()
    header = "| " + " | ".join([df.index.name or ""] + list(df.columns)) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    rows = []
    for idx, row in df.iterrows():
        cells = []
        for col in df.columns:
            text = _fmt(row[col], digits)
            if col in best and row[col] == best[col]:
                text = f"**{text}**"
            cells.append(text)
        rows.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join([header, sep] + rows)


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    return df.reindex([p for p in ORDER if p in df.index] + [p for p in df.index if p not in ORDER])


def findings(summary: pd.DataFrame, k: int) -> list[str]:
    out = []
    hit = f"hit@{k}"
    for metric, label in ((hit, f"Hit@{k}"), ("mrr", "MRR"), ("faithfulness", "faithfulness"),
                          ("correctness", "correctness")):
        if metric in summary and summary[metric].notna().any():
            out.append(f"Best **{label}**: `{summary[metric].idxmax()}` ({summary[metric].max():.3f})")
    if "agentic" in summary.index:
        base = [b for b in BASELINES if b in summary.index]
        for metric, label in ((hit, f"Hit@{k}"), ("correctness", "correctness")):
            if base and metric in summary and pd.notna(summary.loc["agentic", metric]):
                best_base = summary.loc[base, metric].idxmax()
                delta = summary.loc["agentic", metric] - summary.loc[best_base, metric]
                out.append(f"Agentic vs best baseline (`{best_base}`) on {label}: "
                           f"{'+' if delta >= 0 else ''}{delta * 100:.1f} points")
        if "total_s_p50" in summary and base:
            fastest = summary.loc[base, "total_s_p50"].idxmin()
            ratio = summary.loc["agentic", "total_s_p50"] / max(summary.loc[fastest, "total_s_p50"], 1e-9)
            out.append(f"Agentic median latency is {ratio:.1f}x `{fastest}` "
                       f"({summary.loc['agentic', 'total_s_p50']:.2f}s vs {summary.loc[fastest, 'total_s_p50']:.2f}s)")
    return out


def agent_section(records: list[dict], k: int) -> list[str]:
    recs = [r for r in records if r["pipeline"] == "agentic" and (r.get("agent") or {}).get("trace")]
    if not recs:
        return []
    lines = ["## Agent behaviour", ""]
    first = Counter((r["agent"].get("strategies") or ["?"])[0] for r in recs)
    attempts = [r["agent"].get("attempts") or 0 for r in recs]
    accepted = sum(1 for r in recs if r["agent"].get("sufficient"))
    lines.append(f"- First strategy chosen: " + ", ".join(f"`{s}` {n}" for s, n in first.most_common()))
    lines.append(f"- Average retrieve→grade attempts: {sum(attempts) / len(attempts):.2f}")
    lines.append(f"- Context accepted by grader: {accepted}/{len(recs)} "
                 f"({accepted / len(recs) * 100:.0f}%); the rest fell back to fusing all attempts")
    by_type = Counter((r.get("type", "?"), (r["agent"].get("strategies") or ["?"])[0]) for r in recs)
    types = sorted({t for t, _ in by_type})
    if types:
        lines += ["", "Routing by question type:", "",
                  "| type | " + " | ".join(ORDER[:3]) + " |", "|---|---|---|---|"]
        for t in types:
            lines.append(f"| {t} | " + " | ".join(str(by_type.get((t, s), 0)) for s in ORDER[:3]) + " |")
    misses = [r for r in recs if not r.get(f"hit@{k}")][:5]
    if misses:
        lines += ["", f"Questions where the agent missed the gold chunk (up to 5):", ""]
        for r in misses:
            lines.append(f"- **{r['qid']}** ({r.get('type')}): {r['question']} — strategies "
                         f"{' → '.join(r['agent'].get('strategies') or [])}")
    return lines + [""]


def build_report(records: list[dict], summary: pd.DataFrame, by_type: pd.DataFrame, k: int,
                 meta: dict) -> str:
    summary = _ordered(summary)
    hit = f"hit@{k}"
    quality_cols = [c for c in (hit, f"recall@{k}", "mrr", f"ndcg@{k}", f"doc_hit@{k}", "faithfulness",
                                "answer_relevance", "correctness", "abstained") if c in summary]
    cost_cols = [c for c in ("retrieval_s_p50", "retrieval_s_p95", "total_s_p50", "total_s_p95",
                             "llm_calls_mean", "prompt_mean", "completion_mean") if c in summary]
    quality = summary[quality_cols].copy()
    quality.index.name = "pipeline"
    cost = summary[cost_cols].copy()
    cost.index.name = "pipeline"

    lines = [
        f"# Benchmark report: {meta.get('run', '')}",
        "",
        f"- **Corpus:** `{meta.get('corpus')}` ({meta.get('domain', '')})",
        f"- **Questions:** {int(summary['n'].max()) if 'n' in summary else '?'} per pipeline · top-k = {k}",
        f"- **Generator / judge:** `{meta.get('model')}` / `{meta.get('judge_model')}` "
        f"· embeddings `{meta.get('embedding_model')}`",
        f"- **Generated:** {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Key findings",
        "",
        *[f"- {f}" for f in findings(summary, k)],
        "",
        "## Quality (higher is better, except abstained)",
        "",
        md_table(quality, bold_max=[c for c in quality_cols if c != "abstained"]),
        "",
        "## Latency and cost (lower is better)",
        "",
        md_table(cost, digits=2, bold_min=cost_cols),
        "",
    ]
    if not by_type.empty and hit in by_type:
        bt = by_type[hit].unstack("pipeline")
        bt = bt[[p for p in ORDER if p in bt.columns]]
        bt.index.name = "question type"
        lines += [f"## {hit} by question type", "", md_table(bt, bold_max=[]), ""]
    lines += agent_section(records, k)
    lines += ["## Metric definitions", "",
              f"- **Hit@{k} / Recall@{k} / MRR / nDCG@{k}**: was the gold chunk retrieved, and how high.",
              f"- **Doc-Hit@{k}**: looser; any chunk from the gold document counts.",
              "- **Faithfulness**: share of answer claims supported by the retrieved context (LLM judge).",
              "- **Answer relevance / correctness**: LLM judge 1–5 scores rescaled to 0–1.",
              "- **Abstained**: share of answers that said \"I don't know\".", ""]
    return "\n".join(lines)
