"""FAISS index + chunk metadata, persisted together.

Layout on disk (one directory per index):
    storage/<name>/index.faiss     the vector index
    storage/<name>/chunks.jsonl    one chunk per line, row i == vector i
    storage/<name>/manifest.json   models, params, per-document content hashes

The manifest is what makes re-indexing cheap: if a PDF's content hash is
already in the manifest and the embedding model hasn't changed, we skip it.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from .chunking import Chunk


def _import_faiss():
    try:
        import faiss  # type: ignore
        return faiss
    except ImportError:
        return None


class VectorStore:
    """Cosine-similarity store over L2-normalised vectors.

    Uses faiss.IndexFlatIP when faiss is available. Falls back to a numpy
    matmul otherwise - exact same results, just slower above ~50k chunks -
    so the project still runs if a `pip install faiss-cpu` fails.
    """

    def __init__(self, dim: int, chunks: list[Chunk] | None = None) -> None:
        self.dim = dim
        self.chunks: list[Chunk] = chunks or []
        self._faiss = _import_faiss()
        self._index = self._faiss.IndexFlatIP(dim) if self._faiss else None
        self._matrix: np.ndarray | None = None

    # -- construction -------------------------------------------------
    @property
    def backend(self) -> str:
        return "faiss.IndexFlatIP" if self._index is not None else "numpy (faiss not installed)"

    def add(self, vectors: np.ndarray, chunks: Sequence[Chunk]) -> None:
        if len(vectors) != len(chunks):
            raise ValueError(f"{len(vectors)} vectors vs {len(chunks)} chunks")
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if self._index is not None:
            self._index.add(vectors)
        else:
            self._matrix = (
                vectors if self._matrix is None
                else np.vstack([self._matrix, vectors])
            )
        self.chunks.extend(chunks)

    def __len__(self) -> int:
        return len(self.chunks)

    # -- search -------------------------------------------------------
    def search(self, query_vector: np.ndarray, k: int = 20) -> list[tuple[int, float]]:
        if not self.chunks:
            return []
        k = min(k, len(self.chunks))
        q = np.ascontiguousarray(query_vector.reshape(1, -1), dtype="float32")
        if self._index is not None:
            scores, ids = self._index.search(q, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
        sims = (self._matrix @ q[0])
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]

    # -- persistence --------------------------------------------------
    def save(self, directory: str | Path, manifest: dict) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if self._index is not None:
            self._faiss.write_index(self._index, str(directory / "index.faiss"))
        else:
            np.save(directory / "index.npy", self._matrix)

        with open(directory / "chunks.jsonl", "w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

        manifest = {**manifest, "dim": self.dim, "n_chunks": len(self.chunks)}
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> tuple["VectorStore", dict]:
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

        chunks: list[Chunk] = []
        with open(directory / "chunks.jsonl", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    chunks.append(Chunk(**json.loads(line)))

        store = cls(dim=manifest["dim"], chunks=chunks)
        faiss_path = directory / "index.faiss"
        npy_path = directory / "index.npy"

        if faiss_path.exists() and store._faiss is not None:
            store._index = store._faiss.read_index(str(faiss_path))
        elif npy_path.exists():
            store._index = None
            store._matrix = np.load(npy_path)
        elif faiss_path.exists() and store._faiss is None:
            raise RuntimeError(
                "This index was built with faiss but faiss is not installed here. "
                "Run: pip install faiss-cpu"
            )
        else:
            raise FileNotFoundError(f"No index file found in {directory}")
        return store, manifest


def index_path(name: str, storage_dir: str | Path) -> Path:
    return Path(storage_dir) / name


def list_indexes(storage_dir: str | Path) -> list[str]:
    root = Path(storage_dir)
    if not root.exists():
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and (p / "manifest.json").exists()
    )
