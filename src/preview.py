"""Render a cited PDF page as an image, with the quoted passage highlighted.

A page number is a claim the reader has to take on trust. Rendering the
actual page beside the answer turns it into something checkable at a
glance, which is the whole argument this project makes about citations.

PyMuPDF is optional. Without it every function here returns None and the
UI falls back to showing the passage text, exactly as before.
"""
from __future__ import annotations

import re
from pathlib import Path

from .config import settings

_CACHE: dict[tuple, bytes] = {}
_CACHE_LIMIT = 40


def _fitz():
    try:
        import fitz  # PyMuPDF
        return fitz
    except ImportError:
        return None


def available() -> bool:
    """True when page rendering is possible in this environment."""
    return _fitz() is not None


def sources_dir(index_dir: str | Path) -> Path:
    return Path(index_dir) / "sources"


def source_path(index_dir: str | Path, doc_id: str) -> Path:
    return sources_dir(index_dir) / f"{doc_id}.pdf"


def _spans(text: str, n: int = 4, words: int = 6) -> list[str]:
    """Pick a few distinctive word-runs to search for on the page.

    Stored chunk text has been normalised - hyphenation rejoined,
    ligatures replaced, newlines collapsed - so it will not match the raw
    PDF byte for byte. Short runs of ordinary words match far more often
    than one long span, and one hit is enough to place a highlight.
    """
    tokens = [t for t in re.split(r"\s+", text.strip()) if t]
    if not tokens:
        return []
    if len(tokens) <= words:
        return [" ".join(tokens)]

    # Sample across the passage so a highlight still lands when the chunk
    # straddles a column break or a figure.
    starts = [round(i * (len(tokens) - words) / max(1, n - 1)) for i in range(n)]
    seen, out = set(), []
    for s in starts:
        span = " ".join(tokens[s:s + words])
        if span not in seen:
            seen.add(span)
            out.append(span)
    return out


def render_page(
    pdf_path: str | Path,
    page: int,
    highlight: str | None = None,
    dpi: int | None = None,
    max_width: int = 1100,
) -> bytes | None:
    """Return a PNG of `page` (1-indexed), or None if rendering is unavailable.

    `highlight` is passage text to mark on the page; failure to locate it
    is not an error, the page still renders.
    """
    fitz = _fitz()
    if fitz is None:
        return None

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return None

    dpi = dpi or settings.preview_dpi
    key = (str(pdf_path), page, (highlight or "")[:120], dpi)
    if key in _CACHE:
        return _CACHE[key]

    try:
        with fitz.open(str(pdf_path)) as doc:
            if not (1 <= page <= doc.page_count):
                return None
            pg = doc[page - 1]

            if highlight:
                hits = []
                for span in _spans(highlight):
                    try:
                        hits = pg.search_for(span)
                    except Exception:
                        hits = []
                    if hits:
                        break
                for rect in hits[:12]:
                    try:
                        annot = pg.add_highlight_annot(rect)
                        # Muted amber rather than highlighter yellow, so the
                        # page still reads as a document.
                        annot.set_colors(stroke=(0.98, 0.86, 0.55))
                        annot.update(opacity=0.45)
                    except Exception:
                        pass

            pix = pg.get_pixmap(dpi=dpi)
            if pix.width > max_width:  # keep payloads small for the browser
                pix = pg.get_pixmap(dpi=int(dpi * max_width / pix.width))
            png = pix.tobytes("png")
    except Exception:
        return None

    if len(_CACHE) >= _CACHE_LIMIT:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = png
    return png


def page_count(pdf_path: str | Path) -> int | None:
    fitz = _fitz()
    if fitz is None or not Path(pdf_path).exists():
        return None
    try:
        with fitz.open(str(pdf_path)) as doc:
            return doc.page_count
    except Exception:
        return None
