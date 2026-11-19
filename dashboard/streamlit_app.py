"""Benchmark dashboard.  Run:  streamlit run dashboard/streamlit_app.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve  # noqa: E402
from src.evaluation.runner import summarize  # noqa: E402

st.set_page_config(page_title="Agentic RAG Benchmark", layout="wide")
cfg = load_config()
K = cfg["retrieval"]["top_k"]
ORDER = ["bm25", "vector", "hybrid", "agentic"]

tab_bench, tab_explore, tab_live = st.tabs(["Benchmark", "Question explorer", "Live query"])

results_dir = resolve(cfg["eval"]["results_dir"])
runs = sorted([p for p in results_dir.glob("*/records.jsonl")], key=lambda p: p.stat().st_mtime, reverse=True)

if runs:
    run = st.sidebar.selectbox("Run", runs, format_func=lambda p: p.parent.name)
    with open(run, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    summary, by_type = summarize(records, K)
    summary = summary.reindex([p for p in ORDER if p in summary.index])
else:
    records, summary, by_type = [], pd.DataFrame(), pd.DataFrame()

with tab_bench:
    st.title("Agentic RAG Benchmark: FastAPI docs")
    if summary.empty:
        st.info("No results yet. Run `python -m src.evaluation.runner` first.")
    else:
        st.caption(f"Run **{run.parent.name}** · {int(summary['n'].max())} questions per pipeline · top-k = {K}")
        quality = [c for c in (f"hit@{K}", "mrr", f"ndcg@{K}", f"doc_hit@{K}", "faithfulness",
                               "answer_relevance", "correctness") if c in summary]
        st.subheader("Quality (higher is better)")
        st.dataframe(summary[quality].style.format("{:.3f}").highlight_max(axis=0, color="#2e7d3240"),
                     use_container_width=True)
        long = summary[quality].reset_index().melt(id_vars="pipeline", var_name="metric", value_name="score")
        st.plotly_chart(px.bar(long, x="metric", y="score", color="pipeline", barmode="group",
                               category_orders={"pipeline": ORDER}, range_y=[0, 1]), use_container_width=True)

        st.subheader("Latency & cost (lower is better)")
        cost = [c for c in summary.columns if c.endswith(("_p50", "_p95", "_mean"))]
        st.dataframe(summary[cost].style.format("{:.2f}"), use_container_width=True)
        df = pd.json_normalize(records)
        st.plotly_chart(px.box(df, x="pipeline", y="timings.total_s", points="outliers",
                               category_orders={"pipeline": ORDER}, labels={"timings.total_s": "total seconds"}),
                        use_container_width=True)

        if not by_type.empty:
            st.subheader(f"Hit@{K} by question type")
            bt = by_type[f"hit@{K}"].reset_index()
            st.plotly_chart(px.bar(bt, x="type", y=f"hit@{K}", color="pipeline", barmode="group",
                                   category_orders={"pipeline": ORDER}, range_y=[0, 1]), use_container_width=True)

        agent_recs = [r for r in records if r["pipeline"] == "agentic" and r.get("agent", {}).get("trace")]
        if agent_recs:
            st.subheader("Agent routing decisions")
            first = pd.Series([r["agent"]["strategies"][0] for r in agent_recs if r["agent"].get("strategies")])
            st.plotly_chart(px.pie(names=first.value_counts().index, values=first.value_counts().values,
                                   title="First strategy chosen"), use_container_width=True)

with tab_explore:
    if not records:
        st.info("No results yet.")
    else:
        qids = sorted({r["qid"] for r in records})
        qid = st.selectbox("Question", qids, format_func=lambda q: next(
            f"{q}: {r['question']}" for r in records if r["qid"] == q))
        rows = [r for r in records if r["qid"] == qid]
        st.markdown(f"**Reference answer:** {rows[0]['reference']}")
        st.markdown(f"**Gold chunk:** `{', '.join(rows[0]['gold_chunk_ids'])}`")
        for col, r in zip(st.columns(len(rows)), sorted(rows, key=lambda r: ORDER.index(r["pipeline"]))):
            with col:
                st.markdown(f"### {r['pipeline']}")
                st.metric(f"hit@{K}", f"{r.get(f'hit@{K}', 0):.0f}", help="1 if gold chunk was retrieved")
                st.caption(f"{r['timings']['total_s']:.2f}s · faithfulness={r.get('faithfulness')} "
                           f"· correctness={r.get('correctness')}")
                st.write(r.get("answer") or "_(retrieval only)_")
                with st.expander("Retrieved chunks"):
                    for c in r["contexts"]:
                        gold = "✅ " if c["chunk_id"] in r["gold_chunk_ids"] else ""
                        st.markdown(f"{gold}`{c['chunk_id']}`")
                if r.get("agent", {}).get("trace"):
                    with st.expander("Agent trace"):
                        st.json(r["agent"]["trace"])

with tab_live:
    st.subheader("Ask the FastAPI docs")

    @st.cache_resource(show_spinner="Loading pipelines (embedding model, indexes)...")
    def get_pipelines():
        from src.pipeline import build_pipelines
        return build_pipelines(cfg, ORDER)

    question = st.text_input("Question", "How do I make a query parameter optional?")
    chosen = st.multiselect("Pipelines", ORDER, default=["hybrid", "agentic"])
    if st.button("Ask", type="primary") and question and chosen:
        pipes = get_pipelines()
        for col, name in zip(st.columns(len(chosen)), chosen):
            with col, st.spinner(name):
                res = pipes[name].run(question)
                st.markdown(f"### {name}")
                st.caption(f"{res.timings['total_s']:.2f}s · {res.tokens['llm_calls']} LLM calls")
                st.write(res.answer)
                with st.expander("Sources"):
                    for i, c in enumerate(res.contexts, 1):
                        st.markdown(f"[{i}] `{c.chunk_id}`")
                if res.agent.get("trace"):
                    with st.expander("Agent trace"):
                        st.json(res.agent["trace"])
