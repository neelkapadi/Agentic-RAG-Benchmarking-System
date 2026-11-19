"""Load FastAPI markdown docs and turn mkdocs-specific syntax into plain markdown.

FastAPI docs embed code with include directives such as
``{* ../../docs_src/query_params/tutorial001.py hl[9] *}`` or ``{!../../docs_src/x.html!}``.
We inline the referenced files so code examples become part of the retrievable text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

INCLUDE_STAR_RE = re.compile(r"\{\*\s*(?P<path>\S+)[^*]*\*\}")
INCLUDE_BANG_RE = re.compile(r"\{!>?\s*(?P<path>[^!]+?)\s*!\}")
ANCHOR_RE = re.compile(r"[ \t]*\{\s*#[\w\-]+\s*\}")
ATTR_RE = re.compile(r"\{\s*[.:][^}\n]*\}")
ADMONITION_RE = re.compile(r"^///[ \t]*(\w+)?.*$", re.M)
HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>\n]*>")
CODE_FENCE_SPLIT_RE = re.compile(r"(```.*?```)", re.S)
LANGS = {"py": "python", "js": "javascript", "ts": "typescript", "sh": "bash"}


@dataclass
class Document:
    doc_id: str  # path relative to docs_dir, e.g. "tutorial/query-params.md"
    title: str
    text: str


def _resolve_include(path_str: str, src_root: Path) -> Path | None:
    idx = path_str.find("docs_src/")
    if idx == -1:
        return None
    p = src_root / path_str[idx:]
    if p.exists():
        return p
    # Some pages reference a base file that only exists in versioned variants.
    variants = sorted(p.parent.glob(f"{p.stem}_*{p.suffix}"))
    for pref in ("py310", "py39"):
        for v in variants:
            if pref in v.stem:
                return v
    return variants[0] if variants else None


def _inline_includes(text: str, src_root: Path) -> str:
    def repl(m: re.Match) -> str:
        p = _resolve_include(m.group("path"), src_root)
        if p is None:
            return ""
        code = p.read_text(encoding="utf-8", errors="ignore").strip()
        ext = p.suffix.lstrip(".")
        return f"```{LANGS.get(ext, ext)}\n{code}\n```"

    text = INCLUDE_STAR_RE.sub(repl, text)
    return INCLUDE_BANG_RE.sub(repl, text)


def _clean_prose(text: str) -> str:
    text = ANCHOR_RE.sub("", text)
    text = ATTR_RE.sub("", text)
    text = ADMONITION_RE.sub(lambda m: f"**{m.group(1).capitalize()}:**" if m.group(1) else "", text)
    return HTML_TAG_RE.sub("", text)


def clean_markdown(text: str, src_root: Path) -> str:
    text = _inline_includes(text, src_root)
    # Only clean prose; leave code blocks untouched (they may contain HTML or braces).
    parts = CODE_FENCE_SPLIT_RE.split(text)
    parts = [p if i % 2 else _clean_prose(p) for i, p in enumerate(parts)]
    text = "".join(parts)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _is_excluded(rel: str, exclude: list[str]) -> bool:
    return any(rel == e or (e.endswith("/") and rel.startswith(e)) for e in exclude)


def load_documents(docs_dir: Path, src_root: Path, exclude: list[str] | None = None) -> list[Document]:
    exclude = exclude or []
    docs = []
    for path in sorted(docs_dir.rglob("*.md")):
        rel = path.relative_to(docs_dir).as_posix()
        if _is_excluded(rel, exclude):
            continue
        text = clean_markdown(path.read_text(encoding="utf-8"), src_root)
        if len(text) < 200:
            continue
        title_match = re.search(r"^#\s+(.+)$", text, re.M)
        title = title_match.group(1).strip() if title_match else path.stem
        docs.append(Document(doc_id=rel, title=title, text=text))
    return docs
