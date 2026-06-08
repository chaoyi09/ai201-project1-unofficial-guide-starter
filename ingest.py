"""Ingest .txt documents from documents/, clean noise, chunk, and write chunks.json.

Spec (planning.md):
  - chunk_size = 300 chars, overlap = 50 chars
  - strip Reddit markdown (>, *bold*, ---, **bold**) and RMP boilerplate
    (tag lines, "For Credit: Yes" metadata lines)
  - metadata per chunk: source filename, professor (if detectable),
    course (if detectable), chunk_index
"""

from __future__ import annotations

import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

DOCS_DIR = Path(__file__).parent / "documents"
OUTPUT_PATH = Path(__file__).parent / "chunks.json"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# --- regexes for cleaning -----------------------------------------------------

# Reddit blockquote markers at the start of a line: "> quoted text"
RE_REDDIT_BLOCKQUOTE = re.compile(r"^\s*>+\s?", re.MULTILINE)
# Horizontal rules / separators: "---", "***", "===" on their own line
RE_HORIZONTAL_RULE = re.compile(r"^\s*([-*=]\s*){3,}\s*$", re.MULTILINE)
# Bold / italic markdown: **text** or *text* or __text__
RE_MD_BOLD = re.compile(r"(\*\*|__)(.+?)\1")
RE_MD_ITALIC = re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)")
# Inline links: [text](url) → keep text
RE_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
# Bare URLs
RE_URL = re.compile(r"https?://\S+")
# RMP boilerplate lines we want to drop entirely
RE_RMP_BOILERPLATE = re.compile(
    r"^\s*("
    r"For Credit:.*|"
    r"Attendance:.*|"
    r"Would Take Again:.*|"
    r"Grade(?: Received)?:.*|"
    r"Textbook:.*|"
    r"Online Class:.*|"
    r"Tags:.*|"
    r"Difficulty:\s*\d.*|"
    r"Quality:\s*\d.*|"
    r"\d+\s+helpful\b.*|"
    r"\d+\s+thumbs?\s+(up|down).*|"
    r"Report\s*$|"
    r"Helpful\s*$"
    r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# Collapse runs of whitespace / blank lines
RE_MULTI_SPACE = re.compile(r"[ \t]+")
RE_MULTI_NEWLINE = re.compile(r"\n{3,}")

# --- metadata detection -------------------------------------------------------

# Known courses from planning.md domain
COURSE_RE = re.compile(r"\bCS\s?\d{4}\b", re.IGNORECASE)

# Known professors from planning.md (extend as new docs are added)
KNOWN_PROFESSORS = [
    "Iraklis Tsekourakis",
    "Iraklis",
    "Mahsa Derakhshan",
    "Mahsa",
    "Aloupis",
    "Ravi",
    "Shesh",
    "Yu",
    "Tahmasebi",
]


@dataclass
class Chunk:
    text: str
    source: str
    professor: str | None
    course: str | None
    chunk_index: int


# --- cleaning -----------------------------------------------------------------

def clean_text(raw: str) -> str:
    # Strip UTF-8 BOM if present (common when files are saved from Windows editors)
    text = raw.lstrip("﻿")

    # Drop RMP boilerplate lines first (before stripping markdown that may match)
    text = RE_RMP_BOILERPLATE.sub("", text)

    # Reddit-style noise
    text = RE_REDDIT_BLOCKQUOTE.sub("", text)
    text = RE_HORIZONTAL_RULE.sub("", text)

    # Markdown formatting → keep the visible text
    text = RE_MD_LINK.sub(r"\1", text)
    text = RE_MD_BOLD.sub(r"\2", text)
    text = RE_MD_ITALIC.sub(r"\1", text)
    text = RE_URL.sub("", text)

    # Whitespace normalization
    text = RE_MULTI_SPACE.sub(" ", text)
    text = RE_MULTI_NEWLINE.sub("\n\n", text)

    # Trim each line
    text = "\n".join(line.strip() for line in text.splitlines())
    # Drop empty trailing/leading lines
    return text.strip()


# --- chunking -----------------------------------------------------------------

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Fixed-size character chunking with overlap.

    Designed for the short, opinion-dense corpus described in planning.md
    (RMP reviews + Reddit comments). We do not try to split on sentence
    boundaries because the 50-char overlap is meant to bridge them.
    """
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("invalid chunk size/overlap")

    if not text:
        return []

    step = size - overlap
    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        piece = text[i : i + size].strip()
        if piece:
            chunks.append(piece)
        if i + size >= n:
            break
        i += step
    return chunks


# --- metadata -----------------------------------------------------------------

def detect_professor(chunk: str, filename: str) -> str | None:
    haystack = f"{filename} {chunk}"
    for name in KNOWN_PROFESSORS:
        if re.search(rf"\b{re.escape(name)}\b", haystack, re.IGNORECASE):
            return name
    return None


def detect_course(chunk: str, filename: str) -> str | None:
    haystack = f"{filename} {chunk}"
    m = COURSE_RE.search(haystack)
    if not m:
        return None
    return m.group(0).upper().replace(" ", "")


# --- pipeline -----------------------------------------------------------------

def ingest_file(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    cleaned = clean_text(raw)
    pieces = chunk_text(cleaned)
    chunks: list[Chunk] = []
    for idx, piece in enumerate(pieces):
        chunks.append(
            Chunk(
                text=piece,
                source=path.name,
                professor=detect_professor(piece, path.name),
                course=detect_course(piece, path.name),
                chunk_index=idx,
            )
        )
    return chunks


def ingest_directory(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    if not docs_dir.exists():
        raise FileNotFoundError(f"documents directory not found: {docs_dir}")
    txt_files = sorted(docs_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"no .txt files found in {docs_dir} — add your source files first"
        )
    all_chunks: list[Chunk] = []
    for path in txt_files:
        all_chunks.extend(ingest_file(path))
    return all_chunks


def write_chunks(chunks: list[Chunk], output_path: Path = OUTPUT_PATH) -> None:
    payload = [asdict(c) for c in chunks]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_sample(chunks: list[Chunk], n: int = 5) -> None:
    sample = random.sample(chunks, k=min(n, len(chunks)))
    print(f"\n--- {len(sample)} random sample chunks ---")
    for c in sample:
        print(f"\n[source={c.source}  professor={c.professor}  course={c.course}  idx={c.chunk_index}]")
        print(c.text)
    print("\n--- end sample ---")


def main() -> int:
    try:
        chunks = ingest_directory()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    write_chunks(chunks)
    total = len(chunks)
    print(f"wrote {total} chunks → {OUTPUT_PATH.relative_to(Path.cwd()) if OUTPUT_PATH.is_relative_to(Path.cwd()) else OUTPUT_PATH}")
    if not (50 <= total <= 2000):
        print(
            f"WARNING: chunk count {total} is outside the expected 50–2000 range "
            "(spec sanity check — adjust source docs or chunk size if needed)",
            file=sys.stderr,
        )
    print_sample(chunks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
