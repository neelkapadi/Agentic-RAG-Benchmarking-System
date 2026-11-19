"""Download the FastAPI English docs + code examples (MIT licensed) into data/raw/.

Usage:  python -m scripts.download_docs
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from src.config import ROOT

REPO = "https://github.com/fastapi/fastapi.git"


def main() -> None:
    raw = ROOT / "data" / "raw"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "fastapi"
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", REPO, str(src)], check=True)
        subprocess.run(["git", "sparse-checkout", "set", "docs/en/docs", "docs_src"], cwd=src, check=True)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=src, capture_output=True, text=True).stdout.strip()
        for name, rel in (("fastapi_docs", "docs/en/docs"), ("docs_src", "docs_src")):
            shutil.rmtree(raw / name, ignore_errors=True)
            shutil.copytree(src / rel, raw / name, ignore=shutil.ignore_patterns("img", "css", "js"))
    (raw / "SOURCE.txt").write_text(f"{REPO}\ncommit {commit}\nLicense: MIT\n")
    print(f"Docs downloaded to {raw} (commit {commit[:10]})")


if __name__ == "__main__":
    main()
