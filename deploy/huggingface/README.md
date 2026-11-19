---
title: Agentic RAG Benchmark
emoji: 🔎
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: BM25 vs vector vs hybrid vs agentic RAG on the FastAPI docs
---

# Agentic RAG Benchmarking System: live demo

Compare **BM25, vector, hybrid and agentic (LangGraph)** retrieval pipelines on the FastAPI
documentation, using only open-source models served by Ollama.

- **📊 Report:** benchmark results from a local GPU run (faithfulness, relevance, correctness,
  Hit@5, MRR, latency).
- **🔍 Question explorer:** compare the four pipelines' answers and retrieved chunks per question.
- **💬 Ask:** ask your own questions. This uses `qwen2.5:1.5b` on a free CPU, so expect about 1 min per pipeline (about 2 min for agentic).
  Turn on *retrieval only* for an instant BM25 vs vector vs hybrid comparison.
- **📁 Documents:** upload up to 3 files (≤ 10 MB each, ≤ 100 PDF pages in total) and ask questions about
  them. Uploads are private to your session and deleted after 60 minutes.

Source code, setup and full results: https://github.com/neelkapadi/Agentic-RAG-Benchmarking-System
