"""Load docs -> chunk -> save chunks.jsonl -> build the vector index.

Usage:  python -m scripts.build_index [--config configs/default.yaml] [--skip-vector]
"""
from __future__ import annotations

import argparse
from collections import Counter

from src.config import load_config, resolve
from src.ingest.chunker import chunk_document
from src.ingest.loader import load_documents
from src.ingest.store import ChunkStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-vector", action="store_true", help="only write chunks.jsonl")
    args = ap.parse_args()
    cfg = load_config(args.config)
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

    if not args.skip_vector:
        from src.retrievers.vector import VectorRetriever
        e = cfg["embeddings"]
        VectorRetriever(store, e["model"], resolve(e["index_dir"]), e.get("query_prefix", ""), e.get("batch_size", 32))
        print("Vector index ready.")


if __name__ == "__main__":
    main()
