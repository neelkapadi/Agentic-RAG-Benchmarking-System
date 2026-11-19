"""Config loading and path helpers.

Environment variables override the YAML (used by the Docker / Hugging Face deployment):
  RAG_LLM_MODEL, RAG_JUDGE_MODEL, RAG_OLLAMA_URL, RAG_AGENT_MAX_ATTEMPTS, DEMO_MODE=1
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"

ENV_OVERRIDES: dict[str, tuple[tuple[str, str], type]] = {
    "RAG_LLM_MODEL": (("llm", "model"), str),
    "RAG_JUDGE_MODEL": (("llm", "judge_model"), str),
    "RAG_OLLAMA_URL": (("llm", "base_url"), str),
    "RAG_AGENT_MAX_ATTEMPTS": (("agent", "max_attempts"), int),
}


def demo_mode() -> bool:
    """Public read-mostly mode: no uploads, no background jobs, rate-limited questions."""
    return os.environ.get("DEMO_MODE", "").strip().lower() in {"1", "true", "yes"}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for var, ((section, key), cast) in ENV_OVERRIDES.items():
        if os.environ.get(var):
            cfg.setdefault(section, {})[key] = cast(os.environ[var])
    return cfg


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
