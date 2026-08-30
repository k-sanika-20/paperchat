"""Embedding client: batching, retry, and normalisation.

The Gemini free tier rate-limits aggressively, and a 200-page paper set is
thousands of chunks. Naive per-chunk calls get you a 429 within seconds, so
everything here is batched with exponential backoff.
"""
from __future__ import annotations

import re
import time
from collections import deque
from typing import Callable, Sequence

import numpy as np

from .config import settings


class RateLimiter:
    """Sliding-window limiter, requests-per-minute.

    The Gemini free tier meters embeddings PER TEXT, not per call, so a
    batch of 32 spends 32 of the minute's allowance. Batching keeps HTTP
    overhead down but buys nothing against quota - only waiting does.
    """

    def __init__(self, per_minute: int) -> None:
        self.per_minute = max(1, per_minute)
        self._times: deque[float] = deque()

    def acquire(self, n: int = 1, notify: Callable[[float], None] | None = None) -> None:
        while True:
            now = time.time()
            while self._times and now - self._times[0] >= 60.0:
                self._times.popleft()
            if len(self._times) + n <= self.per_minute:
                self._times.extend([now] * n)
                return
            # Wait until the oldest request falls out of the window.
            wait = 60.0 - (now - self._times[0]) + 0.25
            if notify:
                notify(wait)
            time.sleep(max(0.1, wait))


def _retry_after(exc: Exception) -> float | None:
    """Pull the server's own suggested delay out of a 429, if it gave one."""
    text = str(exc)
    for pattern in (r"retry in ([\d.]+)\s*s", r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'",
                    r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"'):
        m = re.search(pattern, text, re.I)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


class EmbeddingClient:
    def __init__(
        self,
        model: str | None = None,
        batch_size: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        from .compat import GoogleGenerativeAIEmbeddings  # imported lazily

        settings.validate()
        self.model = model or settings.embed_model
        self.batch_size = batch_size or settings.embed_batch_size
        self.max_retries = max_retries or settings.embed_max_retries
        self._backend = GoogleGenerativeAIEmbeddings(
            model=self.model,
            google_api_key=settings.google_api_key,
        )
        self._dim: int | None = None
        self._limiter = RateLimiter(settings.embed_rpm)
        self.waited = 0.0

    # -- internals ----------------------------------------------------
    def _with_retry(self, fn: Callable[[], list], label: str):
        delay = 2.0
        last: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - provider raises many types
                last = exc
                message = str(exc).lower()
                transient = any(
                    s in message
                    for s in ("429", "rate", "quota", "resource_exhausted", "timeout",
                              "503", "500", "unavailable", "deadline", "internal")
                )
                if not transient or attempt == self.max_retries:
                    raise
                # Prefer the server's own retry hint over our guess; a
                # per-minute quota needs a full window, not 2 seconds.
                hint = _retry_after(exc)
                pause = max(hint + 1.0 if hint else 0.0, delay)
                if "quota" in message or "resource_exhausted" in message:
                    pause = max(pause, 20.0)
                self.waited += pause
                time.sleep(pause)
                delay = min(delay * 2, 90.0)
        raise RuntimeError(f"{label} failed after {self.max_retries} attempts") from last

    # -- public -------------------------------------------------------
    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_query("dimension probe"))
        return self._dim

    def embed_documents(
        self,
        texts: Sequence[str],
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Embed a corpus. Returns an L2-normalised (n, d) float32 array."""
        vectors: list[list[float]] = []
        total = len(texts)
        for start in range(0, total, self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            # Spend quota deliberately rather than discovering it is gone.
            self._limiter.acquire(
                len(batch),
                notify=lambda w: progress and progress(start, total),
            )
            got = self._with_retry(
                lambda b=batch: self._backend.embed_documents(b),
                f"embed_documents[{start}:{start + len(batch)}]",
            )
            vectors.extend(got)
            if progress:
                progress(min(start + len(batch), total), total)
        return normalize(np.asarray(vectors, dtype="float32"))

    def embed_query(self, text: str) -> np.ndarray:
        vec = self._with_retry(
            lambda: self._backend.embed_query(text), "embed_query"
        )
        return normalize(np.asarray([vec], dtype="float32"))[0]


def normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows so an inner-product index gives cosine similarity."""
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype("float32")
