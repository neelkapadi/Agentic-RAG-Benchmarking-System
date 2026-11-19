"""Load any supported file into a Document whose text is light markdown (# headings + paragraphs).

Supported: .md .markdown .txt .pdf .docx .html .htm
The chunker splits on markdown headings, so each loader tries hard to recover real section
headings (PDF: font size / bold, DOCX: heading styles, HTML: <h1>-<h4>).
"""
from __future__ import annotations

import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

from src.ingest.loader import Document, clean_markdown

SUPPORTED = {".md", ".markdown", ".txt", ".pdf", ".docx", ".html", ".htm"}
NUMBERED_HEADING_RE = re.compile(r"^(\d+(\.\d+)*\.?|[IVX]+\.|[A-Z]\.)\s+[A-Z]")
REFERENCES_RE = re.compile(r"^(\d+(\.\d+)*\.?\s+)?(references|bibliography|works cited)\s*$", re.I)


def _title_from(text: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, re.M)
    return m.group(1).strip() if m else fallback


# ---------------------------------------------------------------- PDF

def _join_lines(lines: list[str]) -> str:
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if out.endswith("-") and line[:1].islower():
            out = out[:-1] + line  # re-join hyphenated words
        else:
            out = f"{out} {line}" if out else line
    return out


def _norm(text: str) -> str:
    return re.sub(r"\d+", "#", text.strip().lower())


def pdf_to_markdown(path: Path) -> str:
    import pymupdf

    pages: list[list[list[tuple[str, float, bool]]]] = []  # page -> block -> line(text, size, bold)
    size_chars: Counter = Counter()
    with pymupdf.open(path) as pdf:
        for page in pdf:
            blocks = []
            for b in page.get_text("dict")["blocks"]:
                if b.get("type") != 0:
                    continue
                lines = []
                for line in b["lines"]:
                    spans = [s for s in line["spans"] if s["text"].strip()]
                    if not spans:
                        continue
                    text = "".join(s["text"] for s in line["spans"]).strip()
                    size = max(s["size"] for s in spans)
                    bold = all((s["flags"] & 16) or "bold" in s["font"].lower() for s in spans)
                    for s in spans:
                        size_chars[round(s["size"], 1)] += len(s["text"].strip())
                    lines.append((text, size, bold))
                if lines:
                    blocks.append(lines)
            pages.append(blocks)

    if sum(size_chars.values()) < 50:
        return ""  # scanned / image-only PDF
    body = size_chars.most_common(1)[0][0]

    # Running headers/footers: text at the top/bottom of most pages.
    edge: Counter = Counter()
    for blocks in pages:
        edge.update({_norm(_join_lines([t for t, _, _ in b])) for b in blocks[:2] + blocks[-2:]})
    repeated = {k for k, v in edge.items() if len(pages) >= 3 and v >= max(3, 0.5 * len(pages))}

    def is_heading(text: str, size: float, bold: bool) -> bool:
        if len(text) > 100 or len(text) < 3 or text.endswith((".", ",", ";", ":")) and not NUMBERED_HEADING_RE.match(text):
            return False
        if size >= body * 1.15:
            return True
        return bold and size >= body * 0.95 and (bool(NUMBERED_HEADING_RE.match(text)) or len(text.split()) <= 8)

    out: list[str] = []
    title_done = False
    for blocks in pages:
        for block in blocks:
            if _norm(_join_lines([t for t, _, _ in block])) in repeated:
                continue
            i = 0
            while i < len(block) and is_heading(*block[i]):  # leading heading lines of the block
                i += 1
            if i:
                heading = _join_lines([t for t, _, _ in block[:i]])
                if REFERENCES_RE.match(heading):
                    return "\n\n".join(out)  # drop the bibliography and everything after it
                level = 1 if not title_done and max(s for _, s, _ in block[:i]) >= body * 1.4 else 2
                title_done = title_done or level == 1
                out.append(f"{'#' * level} {heading}")
            rest = _join_lines([t for t, _, _ in block[i:]])
            if rest and not re.fullmatch(r"\d{1,4}", rest):  # skip bare page numbers
                out.append(rest)
    return "\n\n".join(out)


# ---------------------------------------------------------------- DOCX

def docx_to_markdown(path: Path) -> str:
    import docx

    d = docx.Document(str(path))
    parts = []
    for p in d.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        style = (p.style.name if p.style is not None else "").lower()
        m = re.match(r"heading\s*(\d)", style)
        if style == "title":
            parts.append(f"# {text}")
        elif m:
            parts.append(f"{'#' * min(int(m.group(1)) + 1, 4)} {text}")
        elif "list" in style:
            parts.append(f"* {text}")
        else:
            parts.append(text)
    for table in d.tables:
        rows = [" | ".join(c.text.strip() for c in row.cells) for row in table.rows]
        parts.append("\n".join(rows))
    return "\n\n".join(parts)


# ---------------------------------------------------------------- HTML

class _HTMLToMarkdown(HTMLParser):
    SKIP = {"script", "style", "nav", "footer", "header", "noscript", "svg"}
    BLOCK = {"p", "div", "li", "tr", "br", "section", "article", "table", "ul", "ol", "blockquote"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip_depth += 1
        elif m := re.fullmatch(r"h([1-6])", tag):
            self.parts.append("\n\n" + "#" * min(int(m.group(1)), 4) + " ")
        elif tag == "pre":
            self.parts.append("\n\n```\n")
        elif tag == "li":
            self.parts.append("\n* ")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip_depth = max(0, self.skip_depth - 1)
        elif tag == "pre":
            self.parts.append("\n```\n\n")
        elif re.fullmatch(r"h[1-6]", tag) or tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)


def html_to_markdown(html: str) -> str:
    parser = _HTMLToMarkdown()
    parser.feed(html)
    text = "".join(parser.parts)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------------------------------------------------------- entry point

def load_file(path: Path, doc_id: str | None = None, src_root: Path | None = None) -> Document | None:
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {ext}")
    if ext == ".pdf":
        text = pdf_to_markdown(path)
    elif ext == ".docx":
        text = docx_to_markdown(path)
    elif ext in {".html", ".htm"}:
        text = html_to_markdown(path.read_text(encoding="utf-8", errors="ignore"))
    elif ext in {".md", ".markdown"}:
        text = clean_markdown(path.read_text(encoding="utf-8", errors="ignore"), src_root)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < 50:
        return None
    return Document(doc_id=doc_id or path.name, title=_title_from(text, path.stem), text=text)
