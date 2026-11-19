"""REST API.  Run:  uvicorn src.api.app:app --reload
Pipelines are built once per corpus on first use and cached.
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import load_config
from src.corpus import DEFAULT_CORPUS, apply_corpus, list_corpora
from src.pipeline import ALL_PIPELINES, RAGPipeline, build_pipelines

app = FastAPI(title="Agentic RAG Benchmarking System", version="0.2.0")


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    pipeline: str = "agentic"
    corpus: str = DEFAULT_CORPUS
    generate: bool = True


@lru_cache(maxsize=4)
def _pipelines(corpus: str) -> dict[str, RAGPipeline]:
    return build_pipelines(apply_corpus(load_config(), corpus), list(ALL_PIPELINES))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/pipelines")
def pipelines() -> list[str]:
    return list(ALL_PIPELINES)


@app.get("/corpora")
def corpora() -> list[str]:
    return list_corpora()


@app.post("/query")
def query(req: QueryRequest) -> dict:
    if req.pipeline not in ALL_PIPELINES:
        raise HTTPException(400, f"pipeline must be one of {ALL_PIPELINES}")
    if req.corpus not in list_corpora():
        raise HTTPException(404, f"unknown corpus '{req.corpus}'")
    return _pipelines(req.corpus)[req.pipeline].run(req.question, generate=req.generate).to_dict()
