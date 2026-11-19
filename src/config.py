"""Config loading and path helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
