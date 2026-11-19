"""Tests for multi-format loaders, corpora, prompt filling and the report (no LLM needed)."""
from pathlib import Path

import pandas as pd
import pytest

from src.config import load_config
from src.corpus import DEFAULT_CORPUS, apply_corpus, validate_name
from src.evaluation.report import build_report
from src.evaluation.runner import summarize
from src.generation.prompts import fill
from src.ingest.chunker import chunk_document
from src.ingest.loaders import html_to_markdown, load_file

BODY = ("Retrieval augmented generation combines a retriever with a generator so that answers are "
        "grounded in documents. ") * 6


def make_pdf(path: Path) -> None:
    import pymupdf
    pdf = pymupdf.open()
    for i in range(3):
        page = pdf.new_page()
        page.insert_text((72, 40), "Journal of Testing 2026", fontsize=8)  # running header
        y = 90
        if i == 0:
            page.insert_text((72, y), "Agentic Retrieval Study", fontsize=20)
            y += 40
        heading = ["1 Introduction", "2 Method", "References"][i]
        page.insert_text((72, y), heading, fontsize=13, fontname="hebo")
        tb = pymupdf.Rect(72, y + 15, 520, y + 300)
        text = BODY if i < 2 else "[1] Someone et al. A cited paper. 2024."
        page.insert_textbox(tb, text, fontsize=10)
    pdf.save(path)


def test_pdf_headings_headers_and_references(tmp_path):
    p = tmp_path / "paper.pdf"
    make_pdf(p)
    doc = load_file(p)
    assert doc.title == "Agentic Retrieval Study"
    assert "## 1 Introduction" in doc.text and "## 2 Method" in doc.text
    assert "Journal of Testing" not in doc.text          # running header removed
    assert "cited paper" not in doc.text                 # bibliography dropped
    sections = {c.section for c in chunk_document(doc, 800, 100, 50)}
    assert any("2 Method" in s for s in sections)


def test_docx_heading_styles(tmp_path):
    import docx
    d = docx.Document()
    d.add_heading("Design Doc", level=0)
    d.add_heading("Goals", level=1)
    d.add_paragraph("We want fast retrieval. " * 5)
    p = tmp_path / "design.docx"
    d.save(p)
    doc = load_file(p)
    assert doc.title == "Design Doc" and "## Goals" in doc.text


def test_html_to_markdown():
    md = html_to_markdown("<html><script>x=1</script><h1>Guide</h1><p>Hello <b>world</b></p>"
                          "<h2>Install</h2><pre>pip install x</pre><ul><li>a</li><li>b</li></ul></html>")
    assert md.startswith("# Guide") and "## Install" in md and "```\npip install x\n```" in md
    assert "x=1" not in md and "* a" in md


def test_txt_and_md(tmp_path):
    (tmp_path / "notes.txt").write_text("Plain notes. " * 10)
    (tmp_path / "readme.md").write_text("# Readme { #top }\n\nSome text here. " * 3)
    assert load_file(tmp_path / "notes.txt").title == "notes"
    md = load_file(tmp_path / "readme.md")
    assert md.title == "Readme" and "{ #top }" not in md.text
    (tmp_path / "empty.txt").write_text("hi")
    assert load_file(tmp_path / "empty.txt") is None


def test_apply_corpus_paths():
    cfg = load_config()
    fa = apply_corpus(cfg, DEFAULT_CORPUS)
    assert fa["data"]["chunks_path"] == cfg["data"]["chunks_path"]
    assert fa["eval"]["results_dir"].replace("\\", "/").endswith("results/fastapi")
    assert "FastAPI" in fa["domain"]
    with pytest.raises(FileNotFoundError):
        apply_corpus(cfg, "does-not-exist")
    with pytest.raises(ValueError):
        validate_name("Bad Name!")


def test_fill_is_single_pass_and_keeps_json_braces():
    out = fill('Q: {question}\nC: {context}\nJSON: {"a": 1}', question="{context}", context="ctx")
    assert out == 'Q: {context}\nC: ctx\nJSON: {"a": 1}'


def test_report_builds():
    recs = []
    for pipe, hit in (("bm25", 0.0), ("vector", 1.0), ("hybrid", 1.0), ("agentic", 1.0)):
        recs.append({"qid": "q1", "type": "keyword", "pipeline": pipe, "question": "q", "reference": "r",
                     "gold_chunk_ids": ["a"], "answer": "x", "contexts": [], "hit@5": hit, "mrr": hit,
                     "timings": {"retrieval_s": 0.1, "generation_s": 1.0, "total_s": 1.1 + (pipe == "agentic")},
                     "tokens": {"prompt": 10, "completion": 5, "llm_calls": 1},
                     "agent": {"trace": [{"step": "route"}], "strategies": ["vector"], "attempts": 1,
                               "sufficient": True} if pipe == "agentic" else {}})
    summary, by_type = summarize(recs, 5)
    md = build_report(recs, summary, by_type, 5, {"run": "t", "corpus": "fastapi"})
    assert "## Key findings" in md and "Agentic vs best baseline" in md and "## Agent behaviour" in md
    assert isinstance(summary, pd.DataFrame)


def test_judge_pass_rejudges_other_models_only(tmp_path):
    import json
    from src.evaluation.runner import judge_pass

    class FakeJudge:
        calls = 0

        def evaluate(self, question, answer, reference, contexts):
            FakeJudge.calls += 1
            return {"faithfulness": 0.5, "answer_relevance": 1.0, "correctness": 0.75}

    ctx = [{"chunk_id": "a::000", "doc_id": "a", "text": "t", "score": 1.0, "title": "", "section": ""}]
    base = {"qid": "q1", "question": "q", "reference": "r", "answer": "ans", "contexts": ctx}
    recs = [{**base, "pipeline": "bm25", "judge_model": "old-judge", "faithfulness": 1.0},
            {**base, "pipeline": "vector", "judge_model": "new-judge", "faithfulness": 0.9},
            {**base, "pipeline": "hybrid"},
            {**base, "pipeline": "agentic", "answer": ""}]          # nothing to judge
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    assert judge_pass(path, FakeJudge(), "new-judge") == 2
    out = {r["pipeline"]: r for r in map(json.loads, path.read_text().splitlines())}
    assert out["bm25"]["judge_model"] == "new-judge" and out["bm25"]["faithfulness"] == 0.5
    assert out["vector"]["faithfulness"] == 0.9                     # untouched
    assert out["hybrid"]["correctness"] == 0.75
    assert "judge_model" not in out["agentic"]


def test_upload_limits_and_temp_corpora(tmp_path, monkeypatch):
    import json
    import time
    import pymupdf
    from src import corpus as corpus_mod
    from src.corpus import UploadLimits, check_upload, cleanup_expired, display_name, temp_corpus_name

    limits = UploadLimits(max_files=2, max_file_mb=1, max_pdf_pages=3)
    pdf = pymupdf.open()
    for _ in range(4):
        pdf.new_page().insert_text((72, 72), "page text " * 20)
    big_pdf = tmp_path / "four_pages.pdf"
    pdf.save(big_pdf)
    small = tmp_path / "a.txt"
    small.write_text("hello " * 50)

    with pytest.raises(ValueError, match="4 pages"):
        check_upload([big_pdf], limits)
    with pytest.raises(ValueError, match="At most 2 files"):
        check_upload([small, small, small], limits)
    (tmp_path / "huge.txt").write_bytes(b"x" * (2 * 1_048_576))
    with pytest.raises(ValueError, match="MB"):
        check_upload([tmp_path / "huge.txt"], limits)
    check_upload([small], limits)  # ok

    name = temp_corpus_name("abcdef1234567890", "My Paper!!")
    assert name == "tmp-abcdef12-my-paper" and display_name(name) == "my-paper (temporary)"

    monkeypatch.setattr(corpus_mod, "CORPORA_DIR", tmp_path / "corpora")
    for n, expires in (("tmp-a-old", time.time() - 10), ("tmp-b-new", time.time() + 600)):
        d = tmp_path / "corpora" / n
        d.mkdir(parents=True)
        (d / "corpus.json").write_text(json.dumps({"expires_at": expires}))
    assert cleanup_expired() == ["tmp-a-old"]
    assert (tmp_path / "corpora" / "tmp-b-new").exists()


def test_build_corpus_text_limit(tmp_path, monkeypatch):
    from src import corpus as corpus_mod
    from src.corpus import build_corpus

    monkeypatch.setattr(corpus_mod, "CORPORA_DIR", tmp_path / "corpora")
    f = tmp_path / "long.txt"
    f.write_text("word " * 5000)
    with pytest.raises(ValueError, match="limit is 1,000"):
        build_corpus(load_config(), "tmp-x-long", [f], build_vectors=False, max_text_chars=1000)
    assert not (tmp_path / "corpora" / "tmp-x-long" / "corpus.json").exists()  # never listed
    stats = build_corpus(load_config(), "tmp-x-ok", [f], build_vectors=False, extra_meta={"expires_at": 1})
    assert stats["expires_at"] == 1 and (tmp_path / "corpora" / "tmp-x-ok" / "corpus.json").exists()
