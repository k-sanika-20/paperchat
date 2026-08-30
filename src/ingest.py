"""PDF -> clean, page-tagged text records.

Uses pypdf directly (rather than a LangChain loader) so we control page
metadata, repeated header/footer stripping and section detection - all of
which the citation quality depends on.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

# Matches "3.1 Methodology", "IV. RESULTS", "Abstract", "References" etc.
_SECTION_RE = re.compile(
    r"^\s*(?:"
    r"(?P<num>\d+(?:\.\d+)*)\.?\s+(?P<t1>[A-Z][^\n]{2,70})"          # 3.1 Method
    r"|(?P<rom>[IVXLC]+)\.\s+(?P<t2>[A-Z][A-Za-z ]{2,70})"           # IV. Results
    r"|(?P<t3>ABSTRACT|Abstract|INTRODUCTION|Introduction|RELATED WORK|"
    r"Related Work|METHOD(?:S|OLOGY)?|Method(?:s|ology)?|EXPERIMENTS?|"
    r"Experiments?|RESULTS?|Results?|DISCUSSION|Discussion|CONCLUSIONS?|"
    r"Conclusions?|REFERENCES|References|APPENDIX|Appendix)"
    r")\s*$"
)

_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
    "ﬄ": "ffl", "’": "'", "‘": "'", "“": '"',
    "”": '"', "–": "-", "—": "-", " ": " ",
}


@dataclass
class PageRecord:
    """One page of one document, cleaned and tagged."""
    doc_id: str
    source: str          # display filename
    page: int            # 1-indexed, matches what the PDF reader shows
    text: str
    section: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def file_hash(path: str | Path) -> str:
    """Stable content hash - lets us skip re-embedding unchanged PDFs."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _normalise(text: str) -> str:
    for bad, good in _LIGATURES.items():
        text = text.replace(bad, good)
    # Join words broken across a line by hyphenation: "struc-\ntural" -> "structural"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse single newlines inside a paragraph, keep blank-line breaks.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_repeated_lines(pages: list[str], threshold: float = 0.6) -> list[str]:
    """Remove running headers/footers.

    A short line that appears on more than `threshold` of pages is almost
    certainly a header, footer or journal watermark - not content. Leaving
    them in pollutes both BM25 and the embeddings.
    """
    if len(pages) < 4:
        return pages
    counts: Counter[str] = Counter()
    for page in pages:
        for line in {ln.strip() for ln in page.splitlines() if ln.strip()}:
            if len(line) <= 90:
                counts[line] += 1
    cutoff = max(2, int(threshold * len(pages)))
    noise = {line for line, n in counts.items() if n >= cutoff}
    if not noise:
        return pages
    cleaned = []
    for page in pages:
        kept = [ln for ln in page.splitlines() if ln.strip() not in noise]
        cleaned.append("\n".join(kept))
    return cleaned


def _detect_sections(raw_pages: list[str]) -> list[str | None]:
    """Carry the most recent section heading forward across pages."""
    sections: list[str | None] = []
    current: str | None = None
    for page in raw_pages:
        for line in page.splitlines():
            m = _SECTION_RE.match(line.strip())
            if m:
                title = m.group("t1") or m.group("t2") or m.group("t3")
                num = m.group("num") or m.group("rom")
                if title:
                    current = f"{num} {title}".strip() if num else title.strip()
        sections.append(current)
    return sections


def load_pdf(path: str | Path, doc_id: str | None = None) -> list[PageRecord]:
    """Read one PDF into page records. Empty pages are dropped."""
    path = Path(path)
    reader = PdfReader(str(path))
    raw_pages = []
    for page in reader.pages:
        try:
            raw_pages.append(page.extract_text() or "")
        except Exception:
            raw_pages.append("")

    if not any(p.strip() for p in raw_pages):
        raise ValueError(
            f"No text extracted from {path.name}. It is probably a scanned PDF - "
            "run it through OCR (e.g. ocrmypdf) first."
        )

    sections = _detect_sections(raw_pages)
    cleaned = _strip_repeated_lines(raw_pages)

    doc_id = doc_id or file_hash(path)
    records: list[PageRecord] = []
    for idx, (raw, section) in enumerate(zip(cleaned, sections), start=1):
        text = _normalise(raw)
        if len(text) < 40:          # cover pages, blank pages, figure-only pages
            continue
        records.append(
            PageRecord(
                doc_id=doc_id,
                source=path.name,
                page=idx,
                text=text,
                section=section,
            )
        )
    return records


def load_many(paths: Iterable[str | Path]) -> list[PageRecord]:
    records: list[PageRecord] = []
    for p in paths:
        records.extend(load_pdf(p))
    return records
