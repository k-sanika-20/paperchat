"""Hybrid retrieval: dense (FAISS) + lexical (BM25), fused with RRF.

Why hybrid. Dense embeddings are good at paraphrase ("how did they validate
the model?" -> a passage titled "Experimental verification") but bad at rare
literal tokens - dataset names, symbols, acronyms, citation keys. In a paper
corpus those rare tokens are exactly what people search for. BM25 nails them.

Why RRF rather than a weighted sum of scores. Cosine similarity and BM25
scores live on different, corpus-dependent scales, so blending them needs
per-query normalisation that is fragile. Reciprocal Rank Fusion ignores the
scores and uses only rank position:

    RRF(d) = sum over retrievers r of  w_r / (k + rank_r(d))

with k=60 (Cormack et al., 2009). It is scale-free and hard to break.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .bm25 import BM25
from .chunking import Chunk
from .config import settings
from .embeddings import EmbeddingClient
from .store import VectorStore


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    # Set by the re-ranker when it runs; None means it did not.
    rerank_score: float | None = None

    @property
    def matched_by(self) -> str:
        if self.dense_rank is not None and self.lexical_rank is not None:
            return "both"
        if self.dense_rank is not None:
            return "semantic"
        return "keyword"


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingClient,
        use_hybrid: bool | None = None,
        dense_weight: float | None = None,
        rrf_k: int | None = None,
        candidate_k: int | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.use_hybrid = settings.use_hybrid if use_hybrid is None else use_hybrid
        self.dense_weight = (
            settings.dense_weight if dense_weight is None else dense_weight
        )
        self.rrf_k = rrf_k or settings.rrf_k
        self.candidate_k = candidate_k or settings.candidate_k
        # BM25 is built over the same provenance-prefixed text the embedder saw.
        self._bm25 = (
            BM25([c.indexable_text for c in store.chunks])
            if self.use_hybrid and store.chunks
            else None
        )

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        sources: list[str] | None = None,
        mode: Literal["hybrid", "semantic", "keyword"] | None = None,
    ) -> list[RetrievedChunk]:
        k = k or settings.top_k
        if not self.store.chunks:
            return []

        mode = mode or ("hybrid" if self.use_hybrid and self._bm25 else "semantic")
        # Over-fetch before filtering so a source filter can't starve the result.
        fetch = max(self.candidate_k, k * 4)

        dense: list[tuple[int, float]] = []
        lexical: list[tuple[int, float]] = []

        if mode in ("hybrid", "semantic"):
            qv = self.embedder.embed_query(query)
            dense = self.store.search(qv, k=fetch)
        if mode in ("hybrid", "keyword") and self._bm25 is not None:
            lexical = self._bm25.search(query, k=fetch)

        if mode == "semantic":
            fused = {i: (s, r, None) for r, (i, s) in enumerate(dense)}
        elif mode == "keyword":
            fused = {i: (s, None, r) for r, (i, s) in enumerate(lexical)}
        else:
            fused = self._rrf(dense, lexical)

        results = [
            RetrievedChunk(
                chunk=self.store.chunks[idx],
                score=score,
                dense_rank=d_rank,
                lexical_rank=l_rank,
            )
            for idx, (score, d_rank, l_rank) in fused.items()
        ]
        if sources:
            wanted = set(sources)
            results = [r for r in results if r.chunk.source in wanted]

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def _rrf(
        self,
        dense: list[tuple[int, float]],
        lexical: list[tuple[int, float]],
    ) -> dict[int, tuple[float, int | None, int | None]]:
        w_dense = self.dense_weight
        w_lex = 1.0 - self.dense_weight
        scores: dict[int, float] = {}
        d_ranks: dict[int, int] = {}
        l_ranks: dict[int, int] = {}

        for rank, (idx, _) in enumerate(dense):
            scores[idx] = scores.get(idx, 0.0) + w_dense / (self.rrf_k + rank + 1)
            d_ranks[idx] = rank
        for rank, (idx, _) in enumerate(lexical):
            scores[idx] = scores.get(idx, 0.0) + w_lex / (self.rrf_k + rank + 1)
            l_ranks[idx] = rank

        return {
            idx: (score, d_ranks.get(idx), l_ranks.get(idx))
            for idx, score in scores.items()
        }
