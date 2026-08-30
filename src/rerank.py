"""LLM re-ranking: a precision pass over the recall pass.

Hybrid retrieval optimises for RECALL - cast wide, make sure the right
passage is somewhere in the top 20. It is not good at ordering those 20,
because BM25 and cosine both score surface similarity, not whether a
passage actually answers the question. A passage that merely shares
vocabulary with the query ranks alongside one that contains the answer.

So: retrieve ~20 with the cheap retrievers, then have the model read all
of them against the question and score each 0-3. Keep the best k. This is
the same recall-then-precision shape as a cross-encoder re-ranker, using
the model already in the stack instead of shipping a second one - which
matters when the deployment target has under a gigabyte of RAM.

Cost is one extra LLM call per question. Everything is truncated hard so
that call stays small.
"""
from __future__ import annotations

import json
import re
from typing import Sequence

from .config import settings
from .retrieve import RetrievedChunk

RERANK_SYSTEM = """You score how well each numbered passage answers a specific question. You are not answering the question - only judging evidence.

Scale:
  3 - contains the answer, or the specific fact/number/definition asked for
  2 - directly about the question's subject and materially useful, but partial
  1 - same general topic, does not address what was asked
  0 - unrelated, or only shares vocabulary by coincidence

Judge only what the passage says. Do not reward a passage for being well written, for being long, or for repeating the question's words. A passage that uses the query's terminology but answers a different question is a 1, not a 2.

Return ONLY a JSON array, one object per passage, no prose and no code fence:
[{"id": 1, "score": 3}, {"id": 2, "score": 0}]

Every passage id you were given must appear exactly once."""

RERANK_USER = """Question: {question}

Passages:

{passages}

Score every passage. JSON array only."""


def _fmt(hits: Sequence[RetrievedChunk], max_chars: int) -> str:
    out = []
    for i, hit in enumerate(hits, start=1):
        body = hit.chunk.text
        if len(body) > max_chars:
            body = body[:max_chars].rsplit(" ", 1)[0] + " ..."
        out.append(f"[{i}] ({hit.chunk.citation})\n{body}")
    return "\n\n".join(out)


def parse_scores(raw: str, n: int) -> dict[int, float]:
    """Pull {id: score} out of the model's reply.

    Deliberately forgiving: models wrap JSON in code fences, add a
    sentence of preamble, or emit trailing commas. A re-ranker that
    throws on cosmetic noise is worse than no re-ranker, because the
    fallback (fusion order) is already decent.
    """
    if not raw:
        return {}

    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.M).strip()

    payload = None
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        candidate = text[start:end + 1]
        candidate = re.sub(r",\s*([\]}])", r"\1", candidate)  # trailing commas
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            payload = None

    scores: dict[int, float] = {}
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("id"))
                val = float(item.get("score"))
            except (TypeError, ValueError):
                continue
            if 1 <= idx <= n:
                scores[idx] = max(0.0, min(3.0, val))

    if not scores:
        # Last resort: scrape "id: score" style pairs out of prose.
        for a, b in re.findall(r'"?id"?\s*[:=]\s*(\d+)[^0-9-]{0,20}?"?score"?\s*[:=]\s*(-?\d+(?:\.\d+)?)', text, re.I):
            idx, val = int(a), float(b)
            if 1 <= idx <= n:
                scores[idx] = max(0.0, min(3.0, val))

    return scores


class Reranker:
    def __init__(self, llm, max_chars: int | None = None, min_score: float | None = None):
        self.llm = llm
        self.max_chars = max_chars or settings.rerank_max_chars
        self.min_score = settings.rerank_min_score if min_score is None else min_score
        self.last_scores: dict[int, float] = {}
        self.last_status: str = "not run"

    def rerank(self, question: str, hits: Sequence[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        """Return the best k hits by model-judged relevance.

        Falls back to the incoming fusion order on any failure - a bad
        re-rank should never be worse than no re-rank.
        """
        hits = list(hits)
        self.last_scores, self.last_status = {}, "not run"
        if len(hits) <= 1:
            self.last_status = "skipped (nothing to reorder)"
            return hits[:k]

        try:
            reply = self.llm.invoke([
                ("system", RERANK_SYSTEM),
                ("human", RERANK_USER.format(
                    question=question, passages=_fmt(hits, self.max_chars))),
            ])
            content = getattr(reply, "content", reply)
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "") if isinstance(p, dict) else str(p) for p in content
                )
            scores = parse_scores(str(content), len(hits))
        except Exception as exc:  # noqa: BLE001
            self.last_status = f"failed, kept fusion order ({type(exc).__name__})"
            return hits[:k]

        if not scores:
            self.last_status = "unparseable reply, kept fusion order"
            return hits[:k]

        self.last_scores = scores
        coverage = len(scores) / len(hits)

        scored = []
        for i, hit in enumerate(hits, start=1):
            # Unscored passages inherit a neutral 1.0 rather than 0, so a
            # partial reply cannot silently delete good candidates.
            s = scores.get(i, 1.0)
            hit.rerank_score = s
            if s >= self.min_score:
                scored.append((s, i, hit))

        if not scored:
            self.last_status = "everything scored below threshold, kept fusion order"
            return hits[:k]

        # Stable: fusion rank breaks ties, so equal scores keep retrieval order.
        scored.sort(key=lambda t: (-t[0], t[1]))
        kept = [h for _, _, h in scored[:k]]
        dropped = len(hits) - len(scored)
        self.last_status = (
            f"scored {len(scores)}/{len(hits)} passages"
            + (f", dropped {dropped} below {self.min_score}" if dropped else "")
            + ("" if coverage > 0.8 else " (partial reply)")
        )
        return kept
