"""Corpora: each document collection gets its own chunks, vector index, Q&A set and results.

    fastapi (built-in)      paths from configs/default.yaml
    <name>  (your docs)     data/corpora/<name>/{files/, chunks.jsonl, vector_index/, qa_gold.jsonl, corpus.json}
                            results/<name>/<run>/
"""
from __future__ import annotations

import copy
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from src.config import ROOT, resolve

DEFAULT_CORPUS = "fastapi"
CORPORA_DIR = ROOT / "data" / "corpora"
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,48}$")


def corpus_dir(name: str) -> Path:
    return CORPORA_DIR / name


def validate_name(name: str) -> str:
    name = name.strip().lower()
    if not NAME_RE.match(name):
        raise ValueError("Corpus name must be 1-49 chars: lowercase letters, digits, '-' or '_'.")
    return name


def list_corpora() -> list[str]:
    custom = sorted(p.parent.name for p in CORPORA_DIR.glob("*/corpus.json")) if CORPORA_DIR.exists() else []
    return [DEFAULT_CORPUS] + [c for c in custom if c != DEFAULT_CORPUS]


def corpus_info(name: str) -> dict:
    meta = corpus_dir(name) / "corpus.json"
    return json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}


def apply_corpus(cfg: dict, name: str | None) -> dict:
    """Return a copy of cfg whose data/index/QA/results paths point at the given corpus."""
    name = name or DEFAULT_CORPUS
    out = copy.deepcopy(cfg)
    out["corpus"] = name
    results_root = resolve(cfg["eval"]["results_dir"])
    if name == DEFAULT_CORPUS:
        out["domain"] = cfg["data"].get("domain", "the FastAPI documentation")
        out["eval"]["results_dir"] = str(results_root / name)
        return out
    d = corpus_dir(name)
    if not (d / "corpus.json").exists():
        raise FileNotFoundError(f"Corpus '{name}' not found. Build it with: "
                                f"python -m scripts.build_index --corpus {name} --input <folder>")
    out["domain"] = corpus_info(name).get("domain") or f"the '{name}' documents"
    out["data"]["chunks_path"] = str(d / "chunks.jsonl")
    out["embeddings"]["index_dir"] = str(d / "vector_index")
    out["eval"]["qa_path"] = str(d / "qa_gold.jsonl")
    out["eval"]["results_dir"] = str(results_root / name)
    return out


# ---------------------------------------------------------------- public-demo uploads

TEMP_PREFIX = "tmp-"


@dataclass(frozen=True)
class UploadLimits:
    max_files: int = 3
    max_file_mb: float = 10.0
    max_pdf_pages: int = 100        # total across all PDFs in the corpus
    max_text_chars: int = 350_000   # total extracted text (~100 pages); covers DOCX/MD/TXT/HTML
    ttl_minutes: int = 60
    max_temp_corpora: int = 20      # across all visitors, protects the Space's disk


DEMO_LIMITS = UploadLimits()


def check_upload(files: list[Path], limits: UploadLimits) -> None:
    """Raise ValueError with a user-friendly message if the upload breaks a limit."""
    from src.ingest.loaders import SUPPORTED

    if not files:
        raise ValueError("Upload at least one file.")
    if len(files) > limits.max_files:
        raise ValueError(f"At most {limits.max_files} files per corpus in the public demo.")
    pages = 0
    for f in files:
        if f.suffix.lower() not in SUPPORTED:
            raise ValueError(f"{f.name}: unsupported file type.")
        size_mb = f.stat().st_size / 1_048_576
        if size_mb > limits.max_file_mb:
            raise ValueError(f"{f.name} is {size_mb:.1f} MB; the limit is {limits.max_file_mb:.0f} MB per file.")
        if f.suffix.lower() == ".pdf":
            import pymupdf
            try:
                with pymupdf.open(f) as pdf:
                    pages += pdf.page_count
            except Exception as e:
                raise ValueError(f"{f.name}: could not open PDF ({e}).") from e
    if pages > limits.max_pdf_pages:
        raise ValueError(f"The PDFs have {pages} pages in total; the limit is {limits.max_pdf_pages}.")


def temp_corpus_name(session_id: str, name: str) -> str:
    """Per-visitor corpus name, e.g. tmp-1a2b3c4d-mypaper."""
    short = re.sub(r"[^a-z0-9_-]", "-", name.strip().lower())[:30].strip("-") or "docs"
    return validate_name(f"{TEMP_PREFIX}{session_id[:8]}-{short}")


def display_name(corpus: str) -> str:
    if corpus.startswith(TEMP_PREFIX):
        return corpus.split("-", 2)[-1] + " (temporary)"
    return corpus


def delete_corpus(name: str) -> None:
    if name != DEFAULT_CORPUS:
        shutil.rmtree(corpus_dir(name), ignore_errors=True)


def cleanup_expired(now: float | None = None) -> list[str]:
    """Delete temporary corpora past their expiry time, plus half-built ones older than an hour."""
    now = now or time.time()
    removed = []
    if not CORPORA_DIR.exists():
        return removed
    for d in CORPORA_DIR.glob(f"{TEMP_PREFIX}*"):
        meta = d / "corpus.json"
        expires = json.loads(meta.read_text(encoding="utf-8")).get("expires_at", 0) if meta.exists() \
            else d.stat().st_mtime + 3600
        if expires < now:
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d.name)
    return removed


def count_temp_corpora() -> int:
    return len(list(CORPORA_DIR.glob(f"{TEMP_PREFIX}*"))) if CORPORA_DIR.exists() else 0


# ---------------------------------------------------------------- build

def build_corpus(cfg: dict, name: str, files: list[Path], domain: str = "",
                 build_vectors: bool = True, log=print, max_text_chars: int | None = None,
                 extra_meta: dict | None = None, replace: bool = False) -> dict:
    """Copy files into the corpus folder, parse, chunk, and index them. Returns stats.

    replace=True starts from an empty corpus (used for the demo's one-corpus-per-visitor rule).
    corpus.json is written last, so a corpus only appears in listings once fully built.
    """
    from src.ingest.chunker import chunk_document
    from src.ingest.loaders import SUPPORTED, load_file
    from src.ingest.store import ChunkStore

    name = validate_name(name)
    if name == DEFAULT_CORPUS:
        raise ValueError(f"'{DEFAULT_CORPUS}' is the built-in corpus; pick another name.")
    d = corpus_dir(name)
    if replace:
        shutil.rmtree(d, ignore_errors=True)
    (d / "corpus.json").unlink(missing_ok=True)
    files_dir = d / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        if f.suffix.lower() in SUPPORTED and f.resolve().parent != files_dir.resolve():
            shutil.copy2(f, files_dir / f.name)

    c = cfg["chunking"]
    docs, chunks, skipped = [], [], []
    for path in sorted(files_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED:
            continue
        try:
            doc = load_file(path, doc_id=path.name)
        except Exception as e:  # one bad file should not kill the corpus
            skipped.append(f"{path.name}: {e}")
            continue
        if doc is None:
            skipped.append(f"{path.name}: no extractable text (scanned PDF?)")
            continue
        docs.append(doc)
        chunks.extend(chunk_document(doc, c["max_chars"], c["overlap_chars"], c["min_chars"]))
        log(f"  {path.name}: {len(doc.text):,} chars")
    if not chunks:
        raise ValueError("No text could be extracted from the uploaded files. " + "; ".join(skipped))
    total_chars = sum(len(doc.text) for doc in docs)
    if max_text_chars and total_chars > max_text_chars:
        raise ValueError(f"The documents contain {total_chars:,} characters of text; "
                         f"the limit is {max_text_chars:,} (about 100 pages).")

    store = ChunkStore(chunks)
    store.save(d / "chunks.jsonl")
    log(f"{len(docs)} documents -> {len(chunks)} chunks")

    if build_vectors:
        from src.retrievers.vector import VectorRetriever
        e = cfg["embeddings"]
        VectorRetriever(store, e["model"], d / "vector_index", e.get("query_prefix", ""), e.get("batch_size", 32))
        log("Vector index ready.")

    stats = {"name": name, "domain": domain.strip() or f"the '{name}' documents",
             "documents": [doc.doc_id for doc in docs], "n_chunks": len(chunks), "n_chars": total_chars,
             "skipped": skipped, "built_at": time.strftime("%Y-%m-%d %H:%M:%S"), **(extra_meta or {})}
    (d / "corpus.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
