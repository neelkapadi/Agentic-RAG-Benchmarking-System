"""Benchmark dashboard.  Run:  streamlit run dashboard/streamlit_app.py

Tabs
- Report:            metrics, charts, the generated Markdown report and downloads for a run
- Question explorer: compare the four pipelines' answers and retrieved chunks per question
- Ask:               live questions against the selected corpus
- Documents:         upload your own files -> new corpus -> generate test questions -> run benchmark
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import threading  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402

from src.config import demo_mode, load_config, resolve  # noqa: E402
from src.corpus import (DEFAULT_CORPUS, DEMO_LIMITS, TEMP_PREFIX, apply_corpus,  # noqa: E402
                        build_corpus, check_upload, cleanup_expired, corpus_dir, corpus_info,
                        count_temp_corpora, delete_corpus, display_name, list_corpora,
                        temp_corpus_name, validate_name)
from src.evaluation.report import build_report  # noqa: E402
from src.evaluation.runner import summarize  # noqa: E402
from src.ingest.loaders import SUPPORTED  # noqa: E402
from src.pipeline import ALL_PIPELINES, domain_of  # noqa: E402

st.set_page_config(page_title="Agentic RAG Benchmark", layout="wide")
BASE_CFG = load_config()
K = BASE_CFG["retrieval"]["top_k"]
ORDER = list(ALL_PIPELINES)
JOB_DIR = ROOT / "data" / "jobs"
DEMO = demo_mode()
DEMO_MAX_QUESTIONS = 10       # per browser session
DEMO_MAX_PIPELINES = 2        # per question
DEMO_MAX_CHARS = 300


# ------------------------------------------------------------------ shared state

@st.cache_resource
def llm_lock() -> threading.Lock:
    """One question at a time: the LLM (on a free CPU in demo mode) can't serve parallel users."""
    return threading.Lock()


def demo_guard(question: str, chosen: list[str]) -> str | None:
    if not DEMO:
        return None
    if len(question) > DEMO_MAX_CHARS:
        return f"Please keep questions under {DEMO_MAX_CHARS} characters in the public demo."
    if len(chosen) > DEMO_MAX_PIPELINES:
        return f"Pick at most {DEMO_MAX_PIPELINES} pipelines per question in the public demo."
    if st.session_state.get("asked", 0) >= DEMO_MAX_QUESTIONS:
        return f"You've reached the demo limit of {DEMO_MAX_QUESTIONS} questions. Run it locally for unlimited use."
    return None


@st.cache_resource
def job_registry() -> dict:
    """Background jobs (subprocess handles) survive Streamlit reruns."""
    return {}


@st.cache_resource(show_spinner="Loading pipelines (embedding model + indexes)...", max_entries=6)
def get_pipelines(corpus: str):
    from src.pipeline import build_pipelines
    return build_pipelines(apply_corpus(BASE_CFG, corpus), ORDER)


def start_job(key: str, args: list[str]) -> None:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    log_path = JOB_DIR / f"{key}.log"
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-u", *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8",
                                 "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"})
    job_registry()[key] = {"proc": proc, "log": log_path, "args": args}


def show_job(key: str) -> bool:
    """Render a job's status + log tail. Returns True while running."""
    job = job_registry().get(key)
    if not job:
        return False
    code = job["proc"].poll()
    text = job["log"].read_text(encoding="utf-8", errors="ignore") if job["log"].exists() else ""
    lines = [ln for ln in text.replace("\r", "\n").splitlines() if ln.strip()]
    if code is None:
        st.info("Running… press **Refresh** to update.")
    elif code == 0:
        st.success("Finished.")
    else:
        st.error(f"Failed (exit code {code}).")
    st.code("\n".join(lines[-15:]) or "(no output yet)", language="text")
    st.button("Refresh", key=f"refresh_{key}")
    return code is None


def load_records(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------------ sidebar

@st.cache_data(ttl=300, show_spinner=False)
def periodic_cleanup() -> list[str]:
    """Delete expired temporary (demo) corpora at most every 5 minutes."""
    return cleanup_expired()


SID = st.session_state.setdefault("sid", uuid.uuid4().hex)
if DEMO:
    periodic_cleanup()
corpora = list_corpora()
if DEMO:  # visitors only see the built-in corpus and their own temporary one
    corpora = [c for c in corpora if not c.startswith(TEMP_PREFIX) or c.startswith(f"{TEMP_PREFIX}{SID[:8]}-")]
if "pending_corpus" in st.session_state:  # set after building a corpus; applied before the widget exists
    st.session_state["corpus"] = st.session_state.pop("pending_corpus")
if "corpus" in st.session_state and st.session_state.corpus not in corpora:
    del st.session_state["corpus"]
corpus = st.sidebar.selectbox("Corpus", corpora, key="corpus", format_func=display_name)
cfg = apply_corpus(BASE_CFG, corpus)
info = corpus_info(corpus)
st.sidebar.caption(f"**Domain:** {domain_of(cfg)}")
if DEMO:
    st.sidebar.info("**Public demo.** Benchmark numbers come from a local GPU run; live answers here use a "
                    f"small model (`{cfg['llm']['model']}`) on a free CPU, so they are slower and weaker.")
if info:
    st.sidebar.caption(f"{len(info.get('documents', []))} documents · {info.get('n_chunks')} chunks · built {info.get('built_at')}")
    if info.get("expires_at"):
        mins = max(0, int((info["expires_at"] - time.time()) / 60))
        st.sidebar.caption(f"⏳ Temporary corpus: deleted in ~{mins} min.")

results_dir = resolve(cfg["eval"]["results_dir"])
runs = sorted(results_dir.glob("*/records.jsonl"), key=lambda p: (p.stat().st_mtime, p.parent.name), reverse=True)
run = st.sidebar.selectbox("Benchmark run", runs, format_func=lambda p: p.parent.name) if runs else None

records, summary, by_type = [], pd.DataFrame(), pd.DataFrame()
if run:
    records = load_records(run)
    summary, by_type = summarize(records, K)
    summary = summary.reindex([p for p in ORDER if p in summary.index])

tab_report, tab_explore, tab_ask, tab_docs = st.tabs(["📊 Report", "🔍 Question explorer", "💬 Ask", "📁 Documents"])

# ------------------------------------------------------------------ report

with tab_report:
    st.title(f"Benchmark report · {corpus}")
    if summary.empty:
        st.info("No benchmark runs for this corpus yet. Start one from the **Documents** tab "
                f"or run `python -m src.evaluation.runner --corpus {corpus}`.")
    else:
        meta_path = run.parent / "meta.json"
        meta = {"run": run.parent.name, "corpus": corpus, "domain": domain_of(cfg),
                "model": cfg["llm"]["model"], "judge_model": cfg["llm"]["judge_model"],
                "embedding_model": cfg["embeddings"]["model"]}
        if meta_path.exists():  # written when a run finishes
            meta.update(json.loads(meta_path.read_text()))
        else:
            st.caption("⏳ This run is still in progress or was interrupted; numbers are partial.")
        report_md = build_report(records, summary, by_type, K, meta)

        hit = f"hit@{K}"
        cols = st.columns(4)
        for col, (metric, label) in zip(cols, ((hit, f"Best Hit@{K}"), ("mrr", "Best MRR"),
                                               ("correctness", "Best correctness"),
                                               ("faithfulness", "Best faithfulness"))):
            if metric in summary and summary[metric].notna().any():
                col.metric(label, f"{summary[metric].max():.3f}", summary[metric].idxmax(), delta_color="off")

        quality = [c for c in (hit, "mrr", f"ndcg@{K}", f"doc_hit@{K}", "faithfulness",
                               "answer_relevance", "correctness") if c in summary]
        st.subheader("Quality (higher is better)")
        st.dataframe(summary[quality].style.format("{:.3f}", na_rep="–").highlight_max(axis=0, color="#2e7d3240"),
                     width="stretch")
        long = summary[quality].reset_index().melt(id_vars="pipeline", var_name="metric", value_name="score")
        st.plotly_chart(px.bar(long, x="metric", y="score", color="pipeline", barmode="group",
                               category_orders={"pipeline": ORDER}, range_y=[0, 1]), width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Latency")
            df = pd.json_normalize(records)
            st.plotly_chart(px.box(df, x="pipeline", y="timings.total_s", points="outliers",
                                   category_orders={"pipeline": ORDER},
                                   labels={"timings.total_s": "seconds per question"}), width="stretch")
        with c2:
            if not by_type.empty and hit in by_type:
                st.subheader(f"{hit} by question type")
                bt = by_type[hit].reset_index()
                st.plotly_chart(px.bar(bt, x="type", y=hit, color="pipeline", barmode="group",
                                       category_orders={"pipeline": ORDER}, range_y=[0, 1]),
                                width="stretch")

        st.subheader("Full report")
        with st.container(border=True):
            st.markdown(report_md)
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇ report.md", report_md, file_name=f"{corpus}_{run.parent.name}_report.md")
        d2.download_button("⬇ summary.csv", summary.to_csv(), file_name=f"{corpus}_{run.parent.name}_summary.csv")
        d3.download_button("⬇ records.jsonl", run.read_bytes(), file_name=f"{corpus}_{run.parent.name}_records.jsonl")

# ------------------------------------------------------------------ explorer

with tab_explore:
    if not records:
        st.info("No results yet.")
    else:
        qids = sorted({r["qid"] for r in records})
        questions = {r["qid"]: r["question"] for r in records}
        only_misses = st.checkbox("Only questions where some pipeline missed the gold chunk")
        if only_misses:
            qids = [q for q in qids if any(not r.get(f"hit@{K}") for r in records if r["qid"] == q)]
        if qids:
            qid = st.selectbox("Question", qids, format_func=lambda q: f"{q}: {questions[q]}")
            rows = sorted([r for r in records if r["qid"] == qid], key=lambda r: ORDER.index(r["pipeline"]))
            st.markdown(f"**Type:** {rows[0].get('type')} · **Reference answer:** {rows[0]['reference']}")
            st.markdown(f"**Gold chunk:** `{', '.join(rows[0]['gold_chunk_ids'])}`")
            for col, r in zip(st.columns(len(rows)), rows):
                with col:
                    st.markdown(f"### {r['pipeline']}")
                    st.markdown("✅ gold retrieved" if r.get(f"hit@{K}") else "❌ gold missed")
                    st.caption(f"{r['timings']['total_s']:.2f}s · faithfulness={r.get('faithfulness')} "
                               f"· correctness={r.get('correctness')}")
                    st.write(r.get("answer") or "_(retrieval only)_")
                    with st.expander("Retrieved chunks"):
                        for c in r["contexts"]:
                            mark = "✅ " if c["chunk_id"] in r["gold_chunk_ids"] else ""
                            st.markdown(f"{mark}`{c['chunk_id']}`")
                    if (r.get("agent") or {}).get("trace"):
                        with st.expander("Agent trace"):
                            st.json(r["agent"]["trace"])

# ------------------------------------------------------------------ ask

def ask(question: str, chosen: list[str], generate: bool = True) -> None:
    pipes = get_pipelines(corpus)
    for col, name in zip(st.columns(len(chosen)), chosen):
        with col, st.spinner(name):
            res = pipes[name].run(question, generate=generate)
            st.markdown(f"### {name}")
            st.caption(f"{res.timings['total_s']:.2f}s · {res.tokens['llm_calls']} LLM calls")
            if generate:
                st.write(res.answer)
            with st.expander("Sources", expanded=not generate):
                for i, c in enumerate(res.contexts, 1):
                    st.markdown(f"**[{i}]** `{c.chunk_id}` · score {c.score:.3f}")
                    st.caption(c.text[:400] + ("…" if len(c.text) > 400 else ""))
            if res.agent.get("trace"):
                with st.expander("Agent trace"):
                    st.json(res.agent["trace"])


with tab_ask:
    st.subheader(f"Ask {domain_of(cfg)}")
    question = st.text_input("Question", "How do I make a query parameter optional?" if corpus == DEFAULT_CORPUS else "")
    retrieval_only = st.toggle("Retrieval only: compare which passages BM25, vector and hybrid find (fast, no LLM)")
    options = [p for p in ORDER if p != "agentic"] if retrieval_only else ORDER
    chosen = st.multiselect("Pipelines", options, default=options if retrieval_only else ["hybrid", "agentic"],
                            key=f"pipes_{retrieval_only}")
    if DEMO and not retrieval_only:
        st.caption(f"Public demo on a free CPU with `{cfg['llm']['model']}`: expect ~1 min per pipeline (agentic ~2 min). "
                   f"{DEMO_MAX_QUESTIONS - st.session_state.get('asked', 0)} questions left in this session.")
    if st.button("Ask", type="primary") and question and chosen:
        if retrieval_only:
            if DEMO and len(question) > DEMO_MAX_CHARS:
                st.warning(f"Please keep questions under {DEMO_MAX_CHARS} characters in the public demo.")
            else:
                ask(question, chosen, generate=False)
        else:
            problem = demo_guard(question, chosen)
            lock = llm_lock()
            if problem:
                st.warning(problem)
            elif not lock.acquire(blocking=False):
                st.warning("Another request is being processed right now (one at a time). Please try again shortly.")
            else:
                try:
                    st.session_state["asked"] = st.session_state.get("asked", 0) + 1
                    ask(question, chosen)
                finally:
                    lock.release()

# ------------------------------------------------------------------ documents

with tab_docs:
    if DEMO:
        L = DEMO_LIMITS
        st.subheader("Try it on your own document")
        st.caption(f"Up to **{L.max_files} files**, **{L.max_file_mb:.0f} MB** each, **{L.max_pdf_pages} PDF pages** in "
                   f"total (PDF, DOCX, Markdown, TXT, HTML). Your corpus is visible only in this browser session and "
                   f"is deleted after **{L.ttl_minutes} minutes**. A new upload replaces your previous one. "
                   "Please don't upload confidential documents.")
        with st.form("demo_upload"):
            c1, c2 = st.columns(2)
            label = c1.text_input("Name", placeholder="e.g. my-paper")
            topic = c2.text_input("What is it about?", placeholder="e.g. a paper on retrieval-augmented generation")
            uploads = st.file_uploader("Files", type=[e.lstrip(".") for e in SUPPORTED], accept_multiple_files=True)
            go = st.form_submit_button("Build my corpus", type="primary")
        if go:
            name, lock = None, llm_lock()
            try:
                if count_temp_corpora() >= L.max_temp_corpora:
                    raise ValueError("The demo has too many active uploads right now. Please try again later.")
                if not lock.acquire(blocking=False):
                    raise ValueError("Another request is being processed right now (one at a time). Please try again shortly.")
                try:
                    with tempfile.TemporaryDirectory() as tmp, st.status("Building your corpus…", expanded=True) as status:
                        paths = []
                        for up in uploads or []:
                            p = Path(tmp) / Path(up.name).name
                            p.write_bytes(up.getbuffer())
                            paths.append(p)
                        check_upload(paths, L)
                        name = temp_corpus_name(SID, label or "docs")
                        for old in list_corpora():  # one corpus per visitor
                            if old.startswith(f"{TEMP_PREFIX}{SID[:8]}-") and old != name:
                                delete_corpus(old)
                        stats = build_corpus(BASE_CFG, name, paths, topic or f"the uploaded '{display_name(name).replace(' (temporary)', '')}' documents",
                                             log=status.write, replace=True,
                                             max_text_chars=L.max_text_chars,
                                             extra_meta={"expires_at": time.time() + L.ttl_minutes * 60})
                        for s in stats["skipped"]:
                            status.warning(f"Skipped {s}")
                        status.update(label=f"Ready: {stats['n_chunks']} chunks", state="complete")
                finally:
                    lock.release()
                get_pipelines.clear()
                st.session_state["pending_corpus"] = name
                st.rerun()
            except Exception as e:
                if name:
                    delete_corpus(name)  # never leave a half-built corpus behind
                st.error(str(e))
        st.caption("After building, your corpus is selected in the sidebar: use the **💬 Ask** tab. "
                   "Full benchmarks (question generation + LLM judge) need a GPU, so run them locally: "
                   "https://github.com/neelkapadi/Agentic-RAG-Benchmarking-System")
        st.stop()  # Documents is the last tab, so stopping here skips only the local-only controls
    st.subheader("1 · Upload documents")
    st.caption("PDF, DOCX, Markdown, TXT or HTML. Research papers work best as text PDFs (scanned PDFs have no text).")
    with st.form("upload"):
        c1, c2 = st.columns(2)
        new_name = c1.text_input("Corpus name", placeholder="e.g. llm-papers")
        new_domain = c2.text_input("What are these documents about?", placeholder="e.g. LLM research papers")
        uploads = st.file_uploader("Files", type=[e.lstrip(".") for e in SUPPORTED], accept_multiple_files=True)
        submitted = st.form_submit_button("Build corpus", type="primary")
    if submitted:
        try:
            name = validate_name(new_name)
            if name == DEFAULT_CORPUS:
                raise ValueError("'fastapi' is the built-in corpus; choose another name.")
            if not uploads:
                raise ValueError("Upload at least one file.")
            with tempfile.TemporaryDirectory() as tmp, st.status(f"Building corpus '{name}'…", expanded=True) as status:
                paths = []
                for up in uploads:
                    p = Path(tmp) / Path(up.name).name
                    p.write_bytes(up.getbuffer())
                    paths.append(p)
                stats = build_corpus(BASE_CFG, name, paths, new_domain or corpus_info(name).get("domain", ""),
                                     log=status.write)
                for s in stats["skipped"]:
                    status.warning(f"Skipped {s}")
                status.update(label=f"Corpus '{name}' ready: {stats['n_chunks']} chunks", state="complete")
            get_pipelines.clear()
            st.session_state["pending_corpus"] = name
            st.rerun()
        except Exception as e:
            st.error(str(e))

    if corpus != DEFAULT_CORPUS and info:
        st.markdown(f"**Corpus `{corpus}`** contains: " + ", ".join(f"`{d}`" for d in info.get("documents", [])))
        with st.expander("Preview chunks"):
            chunks = load_records(corpus_dir(corpus) / "chunks.jsonl")
            doc = st.selectbox("Document", info.get("documents", []))
            for ch in [c for c in chunks if c["doc_id"] == doc][:8]:
                st.markdown(f"`{ch['chunk_id']}` · {ch['section']}")
                st.caption(ch["text"][:500])

    st.divider()
    st.subheader(f"2 · Generate test questions for `{corpus}`")
    qa_path = resolve(cfg["eval"]["qa_path"])
    if qa_path.exists():
        st.caption(f"{sum(1 for _ in open(qa_path, encoding='utf-8'))} questions in `{qa_path.relative_to(ROOT)}`. "
                   "Review them by hand before trusting the numbers.")
    n_q = st.number_input("Number of questions", 4, 300, 20 if corpus != DEFAULT_CORPUS else 100, step=2)
    qa_key = f"qa_{corpus}"
    if st.button("Generate questions", disabled=show_job(qa_key) if qa_key in job_registry() else False):
        start_job(qa_key, ["-m", "scripts.generate_qa", "--corpus", corpus, "--n", str(n_q)])
        st.rerun()

    st.divider()
    st.subheader(f"3 · Run the benchmark on `{corpus}`")
    c1, c2, c3 = st.columns(3)
    pipes_sel = c1.multiselect("Pipelines", ORDER, default=ORDER, key="bench_pipes")
    limit = c2.number_input("Max questions (0 = all)", 0, 1000, 0)
    use_judge = c3.checkbox("LLM judge (slower)", value=True)
    bench_key = f"bench_{corpus}"
    running = show_job(bench_key) if bench_key in job_registry() else False
    if st.button("Run benchmark", type="primary", disabled=running or not qa_path.exists() or not pipes_sel):
        args = ["-m", "src.evaluation.runner", "--corpus", corpus, "--pipelines", *pipes_sel]
        if limit:
            args += ["--limit", str(limit)]
        if not use_judge:
            args.append("--no-judge")
        start_job(bench_key, args)
        st.rerun()
    if not qa_path.exists():
        st.caption("Generate test questions first.")
