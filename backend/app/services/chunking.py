"""Split a document into retrieval units that keep their page number.

Chunks are the unit of embedding, semantic search and RAG citation, so each one
carries the page it came from and — for Malayalam sources — both the English
text that gets indexed and the original passage that gets shown as evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import settings

HEADING_RE = re.compile(r"^(#{1,3}\s+.+|[A-Z][A-Z0-9 ,./&()-]{6,80})$")
PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+")


@dataclass
class Chunk:
    index: int
    text: str
    original_text: str = ""
    page_number: int | None = None
    section: str = ""
    language: str = "English"

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class PageInput:
    """One page's text as the chunker sees it."""

    page_number: int
    text: str
    original_text: str = ""
    language: str = "English"
    section: str = ""

    @property
    def indexable(self) -> str:
        return self.text or self.original_text

    @property
    def evidence(self) -> str:
        return self.original_text or self.text


def _detect_section(block: str, current: str) -> str:
    first = block.strip().splitlines()[0].strip() if block.strip() else ""
    if HEADING_RE.match(first) and len(first) <= 90:
        return first.lstrip("# ").strip()
    return current


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    """Break an over-long block on sentence boundaries with a little overlap."""
    if len(text) <= size:
        return [text]
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        return [text[start : start + size] for start in range(0, len(text), size - overlap)]

    parts: list[str] = []
    buffer = ""
    for sentence in sentences:
        if buffer and len(buffer) + len(sentence) + 1 > size:
            parts.append(buffer.strip())
            # Carry the tail forward so a fact split across the boundary is
            # still retrievable from at least one chunk in full.
            buffer = buffer[-overlap:] if overlap else ""
            buffer = f"{buffer} {sentence}".strip()
        else:
            buffer = f"{buffer} {sentence}".strip()
        while len(buffer) > size * 1.6:
            parts.append(buffer[:size].strip())
            buffer = buffer[size - overlap :]
    if buffer.strip():
        parts.append(buffer.strip())
    return [part for part in parts if part]


def _proportional_slice(source: str, ratio_start: float, ratio_end: float) -> str:
    """Approximate the passage of ``source`` matching a slice of its translation.

    Malayalam and its English rendering differ in length, so a chunk's original
    evidence is located by position rather than by an exact alignment. It is an
    approximation, and the viewer always offers the full original page too.
    """
    if not source:
        return ""
    length = len(source)
    start = max(0, min(length, int(length * ratio_start)))
    end = max(start, min(length, int(length * ratio_end)))
    return source[start:end].strip()


def chunk_pages(
    pages: list[PageInput],
    size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Chunk page-by-page so every chunk keeps an exact page citation."""
    size = size or settings.kmrl_chunk_chars
    overlap = overlap or settings.kmrl_chunk_overlap

    chunks: list[Chunk] = []
    section = ""
    index = 0

    for page in pages:
        indexable = (page.indexable or "").strip()
        if not indexable:
            continue
        evidence = (page.evidence or "").strip()
        translated = bool(evidence) and evidence != indexable

        blocks = [b.strip() for b in PARAGRAPH_SPLIT.split(indexable) if b.strip()]
        if not blocks:
            blocks = [indexable]

        buffer = ""
        consumed = 0  # characters of ``indexable`` already emitted

        def flush(buffer_text: str, start_offset: int) -> None:
            nonlocal index
            if not buffer_text.strip():
                return
            end_offset = start_offset + len(buffer_text)
            original = (
                _proportional_slice(
                    evidence,
                    start_offset / max(1, len(indexable)),
                    end_offset / max(1, len(indexable)),
                )
                if translated
                else ""
            )
            chunks.append(
                Chunk(
                    index=index,
                    text=buffer_text.strip(),
                    original_text=original,
                    page_number=page.page_number,
                    section=page.section or section,
                    language=page.language,
                )
            )
            index += 1

        for block in blocks:
            section = _detect_section(block, section)
            for part in _split_long(block, size, overlap):
                if buffer and len(buffer) + len(part) + 2 > size:
                    flush(buffer, consumed)
                    consumed += len(buffer)
                    buffer = part
                else:
                    buffer = f"{buffer}\n\n{part}".strip() if buffer else part
        flush(buffer, consumed)

    return chunks


def chunk_text(text: str, page_number: int | None = None) -> list[Chunk]:
    """Convenience wrapper for a single blob of text (used by tests)."""
    return chunk_pages([PageInput(page_number=page_number or 1, text=text)])


def build_page_inputs(document_pages, fallback_text: str = "") -> list[PageInput]:
    """Turn ORM ``DocumentPage`` rows into chunker inputs.

    The English (translated) text is what gets indexed; the original page text
    is kept as the evidence a citation displays.
    """
    inputs: list[PageInput] = []
    for page in document_pages:
        indexed = (page.translated_text or "").strip() or (page.text or "").strip()
        inputs.append(
            PageInput(
                page_number=page.page_number,
                text=indexed,
                original_text=(page.text or "").strip(),
                language=page.language or "English",
            )
        )
    if not inputs and fallback_text.strip():
        inputs.append(PageInput(page_number=1, text=fallback_text.strip()))
    return inputs
