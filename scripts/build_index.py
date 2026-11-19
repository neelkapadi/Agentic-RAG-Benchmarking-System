"""Build a corpus: load docs -> chunk -> save chunks.jsonl -> build the vector index.

Usage:
  python -m scripts.build_index                                   # built-in FastAPI corpus
  python -m scripts.build_index --corpus papers --input C:\\papers --domain "LLM research papers"
  python -m scripts.build_index --corpus papers                   # rebuild from data/corpora/papers/files
Supported files: .pdf .docx .md .txt .html
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.config import load_config, resolve
from src.corpus import DEFAULT_CORPUS, build_corpus, corpus_info
from src.ingest.chunker import chunk_document
from src.ingest.loader import load_documents
from src.ingest.loaders import SUPPORTED
from src.ingest.store import ChunkStore


def build_fastapi(cfg: dict, skip_vector: bool) -> None:
    d, c = cfg["data"], cfg["chunking"]
    docs = load_documents(resolve(d["docs_dir"]), resolve(d["src_root"]), d.get("exclude"))
    chunks = [ch for doc in docs for ch in chunk_document(doc, c["max_chars"], c["overlap_chars"], c["min_chars"])]
    store = ChunkStore(chunks)
    store.save(resolve(d["chunks_path"]))

    lengths = sorted(len(ch.text) for ch in chunks)
    sections = Counter(ch.doc_id.split("/")[0] if "/" in ch.doc_id else "(root)" for ch in chunks)
    print(f"Documents: {len(docs)}  Chunks: {len(chunks)}")
    print(f"Chunk chars: min={lengths[0]} median={lengths[len(lengths) // 2]} max={lengths[-1]}")
    print("Chunks by folder:", dict(sections.most_common()))

    if not skip_vector:
        from src.retrievers.vector import VectorRetriever
        e = cfg["embeddings"]
        VectorRetriever(store, e["model"], resolve(e["index_dir"]), e.get("query_prefix", ""), e.get("batch_size", 32))
        print("Vector index ready.")


def collect_files(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files += [f for f in sorted(p.rglob("*")) if f.suffix.lower() in SUPPORTED]
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(item)
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS, help="corpus name (default: fastapi)")
    ap.add_argument("--input", nargs="+", default=[], help="files or folders to add to the corpus")
    ap.add_argument("--domain", default="", help='what the documents are about, e.g. "LLM research papers"')
    ap.add_argument("--skip-vector", action="store_true", help="only write chunks.jsonl")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.corpus == DEFAULT_CORPUS:
        if args.input:
            ap.error("--input needs a custom --corpus name (the 'fastapi' corpus is built-in).")
        build_fastapi(cfg, args.skip_vector)
        return

    files = collect_files(args.input)
    domain = args.domain or corpus_info(args.corpus).get("domain", "")
    stats = build_corpus(cfg, args.corpus, files, domain, build_vectors=not args.skip_vector)
    for s in stats["skipped"]:
        print("  skipped:", s)
    print(f"Corpus '{stats['name']}' ready. Ask:  python -m scripts.ask \"...\" --corpus {stats['name']}")


if __name__ == "__main__":
    main()
