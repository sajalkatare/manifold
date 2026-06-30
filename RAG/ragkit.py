from __future__ import annotations
import os
import re
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional


# ==========================================================================
# Data model
# ==========================================================================
@dataclass
class Page:
    page: int
    text: str
    tables: list = field(default_factory=list)   # list[list[list[str]]]
    source: str = ""


# ==========================================================================
# LAYER 1 — EXTRACTION
# ==========================================================================
def extract_naive(path: str) -> list[Page]:
    """The way every tutorial does it: pypdf, raw text stream, no layout awareness.
    Fast and simple — and it silently merges columns and flattens tables."""
    from pypdf import PdfReader
    reader = PdfReader(path)
    src = os.path.basename(path)
    pages = []
    for i, pg in enumerate(reader.pages):
        pages.append(Page(page=i + 1, text=pg.extract_text() or "", source=src))
    return pages


def _detect_two_columns(words, page_width):
    """Return a mid-x split point if the page is clearly two-column, else None.
    Robust heuristic: a real two-column page has a clean gutter — almost no word
    crosses the vertical centerline. A single-column paragraph has many words
    spanning the centerline. So we count centerline-crossing words."""
    if not words:
        return None
    split = page_width / 2
    crossing = [w for w in words if w["x0"] < split < w["x1"]]
    left = [w for w in words if (w["x0"] + w["x1"]) / 2 < split]
    right = [w for w in words if (w["x0"] + w["x1"]) / 2 > split]
    # Two columns only if both sides are well populated AND very few words cross
    # the centerline. A couple of full-width lines (a running header, a section
    # title) are tolerated via the proportional threshold; a single-column
    # paragraph crosses the centerline on nearly every line and is rejected.
    tol = max(3, int(0.04 * len(words)))
    if len(left) > 8 and len(right) > 8 and len(crossing) <= tol:
        return split
    return None


def _reflow(words) -> str:
    """Group words into visual lines (by 'top') and join left-to-right."""
    lines: dict[int, list] = {}
    for w in words:
        lines.setdefault(round(w["top"]), []).append(w)
    out = []
    for top in sorted(lines):
        row = sorted(lines[top], key=lambda x: x["x0"])
        out.append(" ".join(w["text"] for w in row))
    return "\n".join(out)


def _table_to_text(table) -> str:
    """Render an extracted table as clean, answerable lines:
        header1: value1 | header2: value2 ...
    so a chunk built from it keeps each value next to its label."""
    if not table or len(table) < 2:
        return ""
    header = [h or "" for h in table[0]]
    lines = ["[TABLE]"]
    for row in table[1:]:
        cells = []
        for h, v in zip(header, row):
            v = (v or "").strip()
            if v:
                cells.append(f"{h.strip()}: {v}" if h.strip() else v)
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def extract_structured(path: str) -> list[Page]:
    """The production-aware way: pdfplumber.
    - Tables are pulled out as real rows and rendered label:value (not number soup).
    - Two-column pages are detected and read column-by-column (not across).
    The point is not 'pdfplumber good' — it's that extraction is a DECISION
    that preserves the structure your documents carry."""
    import pdfplumber
    src = os.path.basename(path)
    pages: list[Page] = []
    with pdfplumber.open(path) as pdf:
        for i, pg in enumerate(pdf.pages):
            raw_tables = pg.extract_tables() or []
            # remove table regions from the word flow so we don't duplicate them
            table_bboxes = [tuple(t.bbox) for t in (pg.find_tables() or [])]

            def _outside_tables(w):
                cx, cy = (w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2
                for (x0, top, x1, bottom) in table_bboxes:
                    if x0 <= cx <= x1 and top <= cy <= bottom:
                        return False
                return True

            words = [w for w in pg.extract_words() if _outside_tables(w)]
            split = _detect_two_columns(words, pg.width)
            if split:
                left = [w for w in words if (w["x0"] + w["x1"]) / 2 < split]
                right = [w for w in words if (w["x0"] + w["x1"]) / 2 >= split]
                body = _reflow(left) + "\n" + _reflow(right)
            else:
                body = _reflow(words)

            body = _strip_furniture(body).strip()
            # NOTE: tables are kept OUT of page.text (in page.tables) so they can
            # become atomic chunks later — a table must never be split mid-row.
            pages.append(Page(page=i + 1, text=body, tables=raw_tables, source=src))
    return pages


_FURNITURE = re.compile(
    r"(ACME CORP|CONFIDENTIAL|Policy Manual|Employee Handbook 2024|^Manual \d{4})", re.I)


def _strip_furniture(text: str) -> str:
    """Drop running headers/footers (page furniture) so they don't pollute chunks.
    Handles both full header lines and the column-split fragments they leave behind
    (e.g. 'Manual 2024 Page 2')."""
    keep = []
    for line in text.splitlines():
        s = line.strip()
        if _FURNITURE.search(s):
            continue
        if re.fullmatch(r"(Page\s+\d+\s*)+", s):
            continue
        # column-split header tail like 'Manual 2024 Page 2'
        if re.search(r"\bManual\b.*Page\s+\d+", s):
            continue
        keep.append(line)
    return "\n".join(keep)


def extract_txt(path: str) -> list[Page]:
    """Plain-text fallback so learners without PDF tooling can still take part."""
    with open(path, encoding="utf-8") as f:
        return [Page(page=1, text=f.read(), source=os.path.basename(path))]


def extract(path: str, structured: bool = True) -> list[Page]:
    """One entry point: pick extractor by file type and mode."""
    if path.lower().endswith(".txt"):
        return extract_txt(path)
    return extract_structured(path) if structured else extract_naive(path)


# ==========================================================================
# EXTRACTION QUALITY CHECKS
# ==========================================================================
def quality_report(text: str) -> dict:
    """Cheap, eyes-on checks for the tells of broken extraction.
    Returns a dict of flags + a human-readable summary. Not a grade — a smoke test."""
    flags = {}

    # 1. words split across line breaks: 'effecti\nve', 'appear i\nn'
    split_words = re.findall(r"[A-Za-z]{1,3}\n[a-z]{1,3}\b", text)
    flags["split_words"] = len(split_words)

    # 2. very short "lines" that look like a flattened table column
    lines = [l for l in text.splitlines() if l.strip()]
    short_lines = [l for l in lines if 0 < len(l.strip()) <= 18]
    flags["suspicious_short_lines"] = len(short_lines)

    # 3. interleaving tell: a sick-leave line immediately followed by a
    #    parental-leave line (or vice versa) — meaning got scrambled
    interleave = 0
    for a, b in zip(lines, lines[1:]):
        al, bl = a.lower(), b.lower()
        if ("sick leave" in al and "parental leave" in bl) or \
           ("parental leave" in al and "sick leave" in bl):
            interleave += 1
    flags["column_interleave_hits"] = interleave

    # 4. empty / near-empty extraction
    flags["char_count"] = len(text)
    flags["looks_empty"] = len(text.strip()) < 50

    problems = []
    if flags["split_words"]:
        problems.append(f"{flags['split_words']} word(s) split across line breaks")
    if flags["column_interleave_hits"]:
        problems.append(f"{flags['column_interleave_hits']} column-interleave hit(s) "
                        f"(sick/parental leave scrambled together)")
    if flags["suspicious_short_lines"] > 6:
        problems.append(f"{flags['suspicious_short_lines']} very short lines "
                        f"(possible flattened table)")
    if flags["looks_empty"]:
        problems.append("extraction looks empty")

    flags["summary"] = problems or ["no obvious extraction problems detected"]
    return flags


# ==========================================================================
# LAYER 2 — CHUNKING
# ==========================================================================
def chunk_fixed(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """Naive fixed-size: fast and brutal. Cuts sentences, tables, clauses in half."""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chunk_recursive(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """Boundary-respecting: split on the biggest natural boundary that fits
    (paragraph -> line -> sentence -> word), keeping units of meaning whole.
    A small, dependency-free stand-in for LangChain's RecursiveCharacterTextSplitter."""
    def _split(t: str, seps: list[str]) -> list[str]:
        if len(t) <= size:
            return [t]
        sep = next((s for s in seps if s and s in t), "")
        parts = list(t) if sep == "" else t.split(sep)
        out, buf = [], ""
        for p in parts:
            piece = p if sep == "" else (p + sep)
            if len(buf) + len(piece) <= size:
                buf += piece
            else:
                if buf:
                    out.append(buf)
                if len(piece) > size:
                    out.extend(_split(piece, seps[seps.index(sep) + 1:] if sep else [""]))
                    buf = ""
                else:
                    buf = piece
        if buf:
            out.append(buf)
        return out

    raw = [c.strip() for c in _split(text, _SEPARATORS) if c.strip()]
    # add a little overlap so an answer near a boundary isn't lost
    if overlap and len(raw) > 1:
        merged = [raw[0]]
        for prev, cur in zip(raw, raw[1:]):
            tail = prev[-overlap:]
            merged.append((tail + " " + cur).strip())
        return merged
    return raw


def is_heading(line: str) -> bool:
    """Heading detector tuned for real policy/handbook docs.
    A heading is one of:
      - starts with 'Section N' or a numeric prefix like '2.1'
      - short Title-Case / ALL-CAPS line (<= 6 words, no trailing period)
    A normal sentence fragment ('A medical certificate is required') is NOT a heading."""
    s = line.strip()
    if not s or len(s) > 60:
        return False
    if re.match(r"^(Section\s+\d+|\d+(\.\d+)*)([\s\.\—:\-]|$)", s):
        return True
    words = s.split()
    if len(words) > 6 or s.endswith("."):
        return False
    # Title Case (most words capitalised) or ALL CAPS
    caps = sum(1 for w in words if w[:1].isupper())
    if s.isupper() and len(s) >= 4:
        return True
    return len(words) >= 2 and caps >= max(2, len(words) - 1)


def chunk_by_section(pages: list[Page]) -> list[tuple[str, int, str]]:
    """Structure-aware: keep a heading with the text under it.
    Returns (chunk_text, page_number, section_title) tuples so we never split a
    heading from its first paragraph."""
    out = []
    for pg in pages:
        section = "Introduction"
        buf = []
        for line in pg.text.splitlines():
            if is_heading(line):
                if buf:
                    out.append(("\n".join(buf).strip(), pg.page, section))
                    buf = []
                section = line.strip()
            elif line.strip():
                buf.append(line)
        if buf:
            out.append(("\n".join(buf).strip(), pg.page, section))
    return [(t, p, s) for (t, p, s) in out if t]


# ==========================================================================
# METADATA — attach at chunk birth
# ==========================================================================
def make_chunk_id(source: str, page: int, section: str, index: int) -> str:
    base = f"{source}|p{page}|{section}|{index}"
    short = hashlib.md5(base.encode()).hexdigest()[:6]
    sec = re.sub(r"[^a-z0-9]+", "-", section.lower()).strip("-")[:20] or "body"
    return f"{os.path.splitext(source)[0]}_p{page}_{sec}_{index:03d}_{short}"


def to_records(chunks, source: str, page: int, section: str = "body",
               document_type: str = "policy") -> list[dict]:
    """Turn raw chunk strings into retrieval-ready records.
    Every chunk leaves here with its address: source, page, section, chunk_id."""
    records = []
    for i, content in enumerate(chunks):
        records.append({
            "chunk_id": make_chunk_id(source, page, section, i),
            "content": content,
            "source": source,
            "page": page,
            "section": section,
            "document_type": document_type,   # used in Session 2/3
            "chunk_index": i,
            "char_len": len(content),
        })
    return records


def records_from_pages(pages: list[Page], document_type: str = "policy",
                       size: int = 500, overlap: int = 50) -> list[dict]:
    """Convenience: structured pages -> metadata-rich chunk records.
    This is the heart of the deliverable. Two kinds of chunks:
      - PROSE   : section-aware + recursively sized (never splits a heading off)
      - TABLE   : one atomic chunk per table, rendered label:value (never split mid-row)
    """
    src = pages[0].source
    out = []

    # 1) prose chunks from page.text
    for (sec_text, page, section) in chunk_by_section(pages):
        pieces = chunk_recursive(sec_text, size=size, overlap=overlap)
        out.extend(to_records(pieces, source=src, page=page, section=section,
                              document_type=document_type))

    # 2) atomic table chunks from page.tables
    for pg in pages:
        for ti, table in enumerate(pg.tables or []):
            text = _table_to_text(table)
            if not text:
                continue
            section = f"Table {ti + 1}"
            out.append(to_records([text], source=src, page=pg.page,
                                  section=section, document_type=document_type)[0])

    # re-number globally so the index is unique across the whole document
    for i, r in enumerate(out):
        r["global_index"] = i
    return out


if __name__ == "__main__":
    print("ragkit loaded OK. Import it from the demo scripts, e.g.:")
    print("  from ragkit import extract_naive, extract_structured, quality_report")
