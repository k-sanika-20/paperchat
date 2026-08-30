"""Okapi BM25, implemented directly.

Written out rather than pulled from a library because it is 60 lines, it
removes a dependency, and knowing exactly how the lexical half of the
hybrid scores documents makes the retrieval behaviour debuggable.

Scoring (Robertson & Sparck Jones):
    score(q, d) = sum_t IDF(t) * (f(t,d) * (k1+1)) /
                              (f(t,d) + k1 * (1 - b + b * |d|/avgdl))
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")

# Deliberately small: technical papers make heavy use of words a big stoplist
# would eat ("no", "all", "between"), so we only drop the truly empty ones.
_STOPWORDS = frozenset(
    "a an the and or of to in for on at by with is are was were be been being "
    "this that these those it its as from we our you your they their he she".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class BM25:
    def __init__(
        self,
        corpus: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.docs: list[list[str]] = [tokenize(d) for d in corpus]
        self.doc_len = [len(d) for d in self.docs]
        self.n_docs = len(self.docs)
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0

        self.tf: list[Counter[str]] = [Counter(d) for d in self.docs]
        df: Counter[str] = Counter()
        for tokens in self.docs:
            df.update(set(tokens))
        # Lucene-style IDF: always positive, so common terms cannot subtract score.
        self.idf = {
            term: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))
            for term, n in df.items()
        }
        # term -> doc ids, so we only score documents that can possibly match.
        self.postings: dict[str, list[int]] = {}
        for i, tokens in enumerate(self.docs):
            for term in set(tokens):
                self.postings.setdefault(term, []).append(i)

    def search(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        """Return [(doc_index, score)] sorted best-first."""
        terms = tokenize(query)
        if not terms or not self.n_docs:
            return []
        scores: dict[int, float] = {}
        for term in terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id in self.postings.get(term, ()):
                freq = self.tf[doc_id][term]
                denom = freq + self.k1 * (
                    1 - self.b + self.b * self.doc_len[doc_id] / (self.avgdl or 1)
                )
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * freq * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:k]
