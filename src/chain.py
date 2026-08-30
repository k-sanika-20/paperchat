"""Answer generation with validated citations."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .config import settings
from .prompts import ANSWER_SYSTEM, ANSWER_USER, CONDENSE_SYSTEM, CONDENSE_USER
from .rerank import Reranker
from .retrieve import HybridRetriever, RetrievedChunk

_CITE_RE = re.compile(r"\[(\d{1,2})\]")
_REFUSAL = "I couldn't find this in the uploaded documents."


@dataclass
class Source:
    marker: int
    source: str
    page: int
    section: str | None
    text: str
    matched_by: str
    score: float
    doc_id: str = ""
    rerank_score: float | None = None


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Source] = field(default_factory=list)
    all_retrieved: list[RetrievedChunk] = field(default_factory=list)
    is_refusal: bool = False
    invalid_markers: list[int] = field(default_factory=list)
    rerank_status: str | None = None
    search_query: str | None = None

    @property
    def cited_markers(self) -> list[int]:
        return [s.marker for s in self.sources]


def format_context(hits: Sequence[RetrievedChunk], max_chars: int = 1800) -> str:
    """Render retrieved chunks as numbered, attributed passages."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        body = hit.chunk.text
        if len(body) > max_chars:
            body = body[:max_chars].rsplit(" ", 1)[0] + " ..."
        blocks.append(f"[{i}] {hit.chunk.citation}\n{body}")
    return "\n\n".join(blocks)


class RAGChain:
    def __init__(self, retriever: HybridRetriever, model: str | None = None) -> None:
        from .compat import ChatGoogleGenerativeAI  # lazy: keeps tests offline

        settings.validate()
        self.retriever = retriever
        self.model_name = model or settings.chat_model
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=settings.temperature,
            google_api_key=settings.google_api_key,
        )
        self.reranker = Reranker(self.llm)
        self.use_rerank = settings.use_rerank

    # -- helpers ------------------------------------------------------
    def _complete(self, system: str, user: str) -> str:
        response = self.llm.invoke(
            [("system", system), ("human", user)]
        )
        content = getattr(response, "content", response)
        if isinstance(content, list):  # some versions return content parts
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content).strip()

    def condense(self, question: str, history: Iterable[tuple[str, str]]) -> str:
        """Turn a follow-up into a standalone query. No-op on the first turn."""
        history = list(history)
        if not history:
            return question
        rendered = "\n".join(
            f"{role.capitalize()}: {msg}" for role, msg in history[-6:]
        )
        try:
            rewritten = self._complete(
                CONDENSE_SYSTEM,
                CONDENSE_USER.format(history=rendered, question=question),
            )
            return rewritten or question
        except Exception:
            return question  # never let query rewriting break the answer path

    # -- retrieval ----------------------------------------------------
    def _gather(
        self,
        question: str,
        k: int | None,
        sources: list[str] | None,
        mode: str | None,
        history,
        rerank: bool | None,
    ):
        """Condense -> retrieve wide -> re-rank down. Shared by both paths."""
        k = k or settings.top_k
        rerank = self.use_rerank if rerank is None else rerank
        search_query = self.condense(question, history or [])

        # Over-fetch when re-ranking, because the point is to give the
        # precision pass something to choose from.
        fetch = max(settings.rerank_candidates, k) if rerank else k
        hits = self.retriever.retrieve(
            search_query, k=fetch, sources=sources, mode=mode  # type: ignore[arg-type]
        )

        status = None
        if rerank and len(hits) > 1:
            hits = self.reranker.rerank(search_query, hits, k)
            status = self.reranker.last_status
        else:
            hits = hits[:k]
            if rerank:
                status = "skipped (nothing to reorder)"

        return hits, search_query, status

    # -- main ---------------------------------------------------------
    def answer(
        self,
        question: str,
        k: int | None = None,
        sources: list[str] | None = None,
        mode: str | None = None,
        history: Iterable[tuple[str, str]] | None = None,
        rerank: bool | None = None,
    ) -> Answer:
        hits, search_query, status = self._gather(
            question, k, sources, mode, history, rerank)

        if not hits:
            return Answer(
                question=question,
                text=_REFUSAL + " (Nothing was retrieved - is an index loaded?)",
                is_refusal=True,
                search_query=search_query,
            )

        text = self._complete(
            ANSWER_SYSTEM,
            ANSWER_USER.format(context=format_context(hits), question=question),
        )
        answer = self._attach_sources(question, text, hits)
        answer.rerank_status = status
        answer.search_query = search_query
        return answer

    def stream(
        self,
        question: str,
        k: int | None = None,
        sources: list[str] | None = None,
        mode: str | None = None,
        history: Iterable[tuple[str, str]] | None = None,
        rerank: bool | None = None,
    ) -> "StreamingAnswer":
        """Retrieve now, generate lazily. See StreamingAnswer."""
        hits, search_query, status = self._gather(
            question, k, sources, mode, history, rerank)
        return StreamingAnswer(self, question, hits, search_query, status)

    # -- citation validation -----------------------------------------
    @staticmethod
    def _attach_sources(
        question: str, text: str, hits: Sequence[RetrievedChunk]
    ) -> Answer:
        """Map [n] markers back to chunks, and drop markers the model invented.

        This is the bit that makes the citations trustworthy: a marker the
        model made up is stripped from the text rather than shown to the user.
        """
        used, invalid = [], []
        for raw in _CITE_RE.findall(text):
            n = int(raw)
            if 1 <= n <= len(hits):
                if n not in used:
                    used.append(n)
            elif n not in invalid:
                invalid.append(n)

        for n in invalid:
            text = text.replace(f"[{n}]", "")
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        text = re.sub(r"[ \t]{2,}", " ", text).strip()

        is_refusal = _REFUSAL.lower() in text.lower()

        sources = [
            Source(
                marker=n,
                source=hits[n - 1].chunk.source,
                page=hits[n - 1].chunk.page,
                section=hits[n - 1].chunk.section,
                text=hits[n - 1].chunk.text,
                matched_by=hits[n - 1].matched_by,
                score=hits[n - 1].score,
                doc_id=hits[n - 1].chunk.doc_id,
                rerank_score=hits[n - 1].rerank_score,
            )
            for n in sorted(used)
        ]
        return Answer(
            question=question,
            text=text,
            sources=sources,
            all_retrieved=list(hits),
            is_refusal=is_refusal,
            invalid_markers=invalid,
        )


class StreamingAnswer:
    """Two-phase answer: stream the text, then validate the citations.

    Citation validation can only run on the finished text - you cannot
    know a marker is invalid until you have it. So the UI streams raw
    deltas for responsiveness, then swaps in the validated text once the
    stream closes. `finalize()` is idempotent.
    """

    def __init__(self, chain: "RAGChain", question: str,
                 hits: Sequence[RetrievedChunk],
                 search_query: str | None, rerank_status: str | None):
        self.chain = chain
        self.question = question
        self.hits = list(hits)
        self.search_query = search_query
        self.rerank_status = rerank_status
        self.raw = ""
        self._answer: Answer | None = None
        self._error: Exception | None = None

    def tokens(self):
        """Yield text deltas. Safe to pass straight to st.write_stream."""
        if not self.hits:
            self.raw = _REFUSAL + " (Nothing was retrieved - is an index loaded?)"
            yield self.raw
            return

        user = ANSWER_USER.format(
            context=format_context(self.hits), question=self.question)
        messages = [("system", ANSWER_SYSTEM), ("human", user)]

        try:
            for chunk in self.chain.llm.stream(messages):
                piece = getattr(chunk, "content", chunk)
                if isinstance(piece, list):
                    piece = "".join(
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in piece
                    )
                piece = str(piece)
                if piece:
                    self.raw += piece
                    yield piece
        except Exception as exc:  # noqa: BLE001
            # Fall back to a blocking call rather than losing the answer:
            # streaming is a nicety, the answer is the product.
            self._error = exc
            try:
                self.raw = self.chain._complete(ANSWER_SYSTEM, user)
                yield self.raw
            except Exception as exc2:  # noqa: BLE001
                self._error = exc2
                raise

    def finalize(self) -> Answer:
        if self._answer is None:
            self._answer = self.chain._attach_sources(
                self.question, self.raw, self.hits)
            self._answer.rerank_status = self.rerank_status
            self._answer.search_query = self.search_query
        return self._answer
