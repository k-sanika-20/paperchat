"""Page records -> retrieval chunks.

Two things matter for citation quality:
  1. A chunk must never span two pages, or the page number in the citation
     becomes a lie.
  2. Each chunk carries a short provenance header (source / page / section)
     that is prepended for BM25 and embedding, so a query like
     "what does the methodology section say" has lexical purchase.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .ingest import PageRecord

_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    page: int
    text: str
    section: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        base = f"{self.source}, p.{self.page}"
        return f"{base} - {self.section}" if self.section else base

    @property
    def indexable_text(self) -> str:
        """Text actually sent to the embedder / BM25 (includes provenance)."""
        head = f"[{self.source} | page {self.page}"
        head += f" | {self.section}]" if self.section else "]"
        return f"{head}\n{self.text}"

    def to_metadata(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source": self.source,
            "page": self.page,
            "section": self.section,
            **self.metadata,
        }


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Recursive character splitting.

    Reimplemented in ~30 lines instead of importing LangChain's splitter so
    the ingestion path has no framework dependency and stays unit-testable.
    Behaviour matches RecursiveCharacterTextSplitter closely enough for RAG.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    def _recurse(segment: str, seps: Sequence[str]) -> list[str]:
        if len(segment) <= chunk_size:
            return [segment]
        if not seps:
            return [segment[i:i + chunk_size] for i in range(0, len(segment), chunk_size)]
        sep, rest = seps[0], seps[1:]
        parts = segment.split(sep) if sep else list(segment)
        out, buf = [], ""
        for part in parts:
            candidate = f"{buf}{sep}{part}" if buf else part
            if len(candidate) <= chunk_size:
                buf = candidate
            else:
                if buf:
                    out.append(buf)
                buf = part if len(part) <= chunk_size else ""
                if len(part) > chunk_size:
                    out.extend(_recurse(part, rest))
        if buf:
            out.append(buf)
        return out

    pieces = [p for p in _recurse(text, _SEPARATORS) if p.strip()]

    # Re-apply overlap by prefixing each chunk with the tail of the previous.
    if overlap <= 0 or len(pieces) < 2:
        return pieces
    overlapped = [pieces[0]]
    for prev, cur in zip(pieces, pieces[1:]):
        tail = prev[-overlap:]
        # Start the overlap at a word boundary so we don't cut mid-token.
        space = tail.find(" ")
        if space != -1:
            tail = tail[space + 1:]
        overlapped.append(f"{tail} {cur}".strip())
    return overlapped


def chunk_pages(
    pages: Sequence[PageRecord],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    min_chars: int = 80,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for record in pages:
        for i, piece in enumerate(_split_text(record.text, chunk_size, chunk_overlap)):
            if len(piece.strip()) < min_chars:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{record.doc_id}:p{record.page}:c{i}",
                    doc_id=record.doc_id,
                    source=record.source,
                    page=record.page,
                    text=piece.strip(),
                    section=record.section,
                )
            )
    return chunks
