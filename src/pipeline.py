"""End-to-end orchestration: PDFs in, queryable index out.

Incremental by design. The manifest records a content hash per document, so
re-running over a folder where one paper changed re-embeds only that paper.
On the Gemini free tier that difference is minutes, not seconds.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Sequence

from .chunking import Chunk, chunk_pages
from .config import settings
from .embeddings import EmbeddingClient
from .ingest import file_hash, load_pdf
from .preview import source_path, sources_dir
from .retrieve import HybridRetriever
from .store import VectorStore, index_path

Progress = Callable[[str, float], None]


def _noop(_msg: str, _frac: float) -> None:
    pass


def build_index(
    pdf_paths: Sequence[str | Path],
    index_name: str = "default",
    storage_dir: str | Path | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    rebuild: bool = False,
    progress: Progress | None = None,
) -> dict:
    """Build (or incrementally extend) an index. Returns build stats."""
    progress = progress or _noop
    storage_dir = Path(storage_dir or settings.storage_dir)
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    target = index_path(index_name, storage_dir)

    embedder = EmbeddingClient()

    existing_store: VectorStore | None = None
    known: dict[str, dict] = {}
    if target.exists() and not rebuild:
        try:
            existing_store, manifest = VectorStore.load(target)
            if manifest.get("embed_model") == embedder.model:
                known = manifest.get("documents", {})
            else:
                progress("Embedding model changed - rebuilding from scratch.", 0.0)
                existing_store = None
        except Exception:
            existing_store = None

    # -- 1. decide what actually needs doing --------------------------
    documents = dict(known)
    todo: list[tuple[Path, str]] = []
    skipped: list[str] = []

    for path in pdf_paths:
        path = Path(path)
        digest = file_hash(path)
        if digest in documents and existing_store is not None:
            skipped.append(path.name)
        else:
            todo.append((path, digest))

    if not todo and existing_store is not None:
        progress("Index already up to date.", 1.0)
        return {
            "index": index_name,
            "path": str(target),
            "n_chunks": len(existing_store),
            "n_documents": len(documents),
            "new_chunks": 0,
            "skipped": skipped,
            "seconds": 0.0,
        }

    # -- 2. one document at a time, checkpointing after each ----------
    #
    # Embedding a large corpus on a rate-limited free tier takes minutes,
    # and a single 429 used to throw away every vector bought so far.
    # Saving per document means an interrupted build resumes where it
    # stopped: re-running skips whatever already landed.
    store = existing_store
    started = time.time()
    total_new = 0
    warnings: list[str] = []

    def manifest_now() -> dict:
        return {
            "index_name": index_name,
            "embed_model": embedder.model,
            "chat_model": settings.chat_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "documents": documents,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    n_docs = len(todo)
    for i, (path, digest) in enumerate(todo):
        lo = 0.02 + 0.95 * i / n_docs          # this document's slice of the bar
        hi = 0.02 + 0.95 * (i + 1) / n_docs

        progress(f"Reading {path.name}", lo)
        pages = load_pdf(path, doc_id=digest)
        chunks = chunk_pages(pages, chunk_size, chunk_overlap)

        if not chunks:
            warnings.append(f"{path.name}: no usable text, skipped")
            progress(f"{path.name}: no usable text, skipped", hi)
            continue

        if len(chunks) > 300:
            warnings.append(
                f"{path.name} produced {len(chunks)} chunks ({len(pages)} pages). "
                f"A document this large will dominate retrieval and takes "
                f"~{len(chunks) // max(1, settings.embed_rpm) + 1} min to embed."
            )

        eta = len(chunks) / max(1, settings.embed_rpm)
        progress(
            f"{path.name}: {len(pages)} pages, {len(chunks)} chunks "
            f"(~{eta:.1f} min at {settings.embed_rpm}/min)", lo
        )

        def on_batch(done: int, total: int, _lo=lo, _hi=hi, _name=path.name) -> None:
            progress(f"{_name}: embedded {done}/{total}",
                     _lo + (_hi - _lo) * done / max(1, total))

        vectors = embedder.embed_documents(
            [c.indexable_text for c in chunks], progress=on_batch
        )

        if store is None:
            store = VectorStore(dim=vectors.shape[1])
        store.add(vectors, chunks)
        total_new += len(chunks)

        documents[digest] = {
            "source": path.name,
            "pages": len(pages),
            "chunks": len(chunks),
        }

        # Keep the PDF beside the index so cited pages can be rendered
        # later. Uploads live in a temp dir that is gone by query time.
        if settings.store_source_pdfs:
            try:
                dest = source_path(target, digest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    dest.write_bytes(path.read_bytes())
                documents[digest]["has_source"] = True
            except OSError:
                documents[digest]["has_source"] = False

        store.save(target, manifest_now())     # checkpoint
        progress(f"{path.name}: saved ({len(store)} chunks total)", hi)

    if store is None or not len(store):
        raise ValueError(
            "No text chunks produced. These are probably scanned PDFs - "
            "run them through OCR first (ocrmypdf in.pdf out.pdf)."
        )

    elapsed = time.time() - started
    progress("Done.", 1.0)

    return {
        "index": index_name,
        "path": str(target),
        "n_chunks": len(store),
        "n_documents": len(documents),
        "new_chunks": total_new,
        "skipped": skipped,
        "warnings": warnings,
        "backend": store.backend,
        "seconds": round(elapsed, 1),
        "rate_limited_seconds": round(getattr(embedder, "waited", 0.0), 1),
    }


def load_retriever(
    index_name: str = "default",
    storage_dir: str | Path | None = None,
    **kwargs,
) -> tuple[HybridRetriever, dict]:
    storage_dir = Path(storage_dir or settings.storage_dir)
    store, manifest = VectorStore.load(index_path(index_name, storage_dir))
    embedder = EmbeddingClient(model=manifest.get("embed_model"))
    return HybridRetriever(store, embedder, **kwargs), manifest


def load_chain(index_name: str = "default", storage_dir: str | Path | None = None, **kwargs):
    from .chain import RAGChain

    retriever, manifest = load_retriever(index_name, storage_dir, **kwargs)
    return RAGChain(retriever), manifest
