"""REST API.  Run:  uvicorn src.api.app:app --reload
Pipelines are built once on first use and cached.
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import load_config
from src.pipeline import ALL_PIPELINES, RAGPipeline, build_pipelines

app = FastAPI(title="Agentic RAG Benchmarking System", version="0.1.0")


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    pipeline: str = "agentic"
    generate: bool = True


@lru_cache(maxsize=1)
def _pipelines() -> dict[str, RAGPipeline]:
    return build_pipelines(load_config(), list(ALL_PIPELINES))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/pipelines")
def pipelines() -> list[str]:
    return list(ALL_PIPELINES)


@app.post("/query")
def query(req: QueryRequest) -> dict:
    if req.pipeline not in ALL_PIPELINES:
        raise HTTPException(400, f"pipeline must be one of {ALL_PIPELINES}")
    return _pipelines()[req.pipeline].run(req.question, generate=req.generate).to_dict()
