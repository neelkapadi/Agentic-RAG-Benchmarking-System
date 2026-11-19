"""Heading-aware markdown chunker that never splits inside a code block."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from src.ingest.loader import Document

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str  # includes a "Title > Section" breadcrumb line for retrieval context

    def to_dict(self) -> dict:
        return asdict(self)


def _is_fence(line: str) -> bool:
    return line.strip().startswith("```")


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading_path, body) pairs, ignoring '#' lines inside code."""
    sections: list[tuple[str, str]] = []
    path: list[tuple[int, str]] = []
    buf: list[str] = []
    in_code = False

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(h for _, h in path), body))

    for line in text.splitlines():
        if _is_fence(line):
            in_code = not in_code
        m = None if in_code else HEADING_RE.match(line)
        if m:
            flush()
            buf = []
            level = len(m.group(1))
            path = [(lvl, h) for lvl, h in path if lvl < level] + [(level, m.group(2).strip())]
        else:
            buf.append(line)
    flush()
    return sections


def split_blocks(body: str) -> list[str]:
    """Split on blank lines, keeping fenced code blocks whole."""
    blocks, cur, in_code = [], [], False
    for line in body.splitlines():
        if _is_fence(line):
            in_code = not in_code
        if not in_code and not line.strip():
            if cur:
                blocks.append("\n".join(cur))
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return [b for b in blocks if b.strip()]


def _hard_split(block: str, max_chars: int) -> list[str]:
    """Split an oversized block by lines (and overlong lines by characters).
    If the block is a fenced code block, every piece is re-wrapped in its own fence."""
    lines = block.splitlines()
    fence = lines[0].strip() if lines and _is_fence(lines[0]) else None
    if fence:
        lines = lines[1:-1] if len(lines) > 1 and _is_fence(lines[-1]) else lines[1:]
    budget = max_chars - (len(fence) + 8 if fence else 0)
    parts: list[str] = []
    for line in lines:
        parts.extend([line[i:i + budget] for i in range(0, len(line), budget)] or [""])
    pieces, cur = [], ""
    for part in parts:
        if cur and len(cur) + len(part) + 1 > budget:
            pieces.append(cur.rstrip("\n"))
            cur = ""
        cur += part + "\n"
    if cur.strip():
        pieces.append(cur.rstrip("\n"))
    return [f"{fence}\n{p}\n```" for p in pieces] if fence else pieces


def pack_blocks(blocks: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    for block in blocks:
        for piece in _hard_split(block, max_chars) if len(block) > max_chars else [block]:
            if cur and size + len(piece) + 2 > max_chars:
                chunks.append("\n\n".join(cur))
                tail = cur[-1] if len(cur[-1]) <= overlap_chars else None
                cur, size = ([tail], len(tail) + 2) if tail else ([], 0)
            cur.append(piece)
            size += len(piece) + 2
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def chunk_document(doc: Document, max_chars: int = 1800, overlap_chars: int = 300,
                   min_chars: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading_path, body in split_sections(doc.text):
        section = heading_path or doc.title
        breadcrumb = section if section.startswith(doc.title) else f"{doc.title} > {section}"
        for text in pack_blocks(split_blocks(body), max_chars, overlap_chars):
            if len(text) < min_chars:
                continue
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}::{len(chunks):03d}",
                doc_id=doc.doc_id,
                title=doc.title,
                section=section,
                text=f"{breadcrumb}\n\n{text}",
            ))
    return chunks
