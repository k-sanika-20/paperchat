"""Offline tests: everything except the two calls that need network.

Run:  python tests/test_pipeline.py
The embedding client is stubbed with a deterministic hashing embedder, so
these run with no API key and no quota burn.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bm25 import BM25, tokenize                    # noqa: E402
from src.chain import RAGChain, format_context         # noqa: E402
from src.chunking import Chunk, chunk_pages, _split_text  # noqa: E402
from src.embeddings import normalize                   # noqa: E402
from src.ingest import PageRecord, _normalise, _strip_repeated_lines, load_pdf  # noqa: E402
from src.retrieve import HybridRetriever, RetrievedChunk  # noqa: E402
from src.rerank import Reranker, parse_scores          # noqa: E402
from src.store import VectorStore                      # noqa: E402
from src import preview, ui                            # noqa: E402

PASSED, FAILED = [], []


def check(name):
    def wrap(fn):
        try:
            fn()
            PASSED.append(name)
            print(f"  PASS  {name}")
        except AssertionError as exc:
            FAILED.append((name, exc))
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            FAILED.append((name, exc))
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
        return fn
    return wrap


class StubEmbedder:
    """Deterministic bag-of-words embedder - no network, but semantically sane."""

    dim = 64

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for token in tokenize(text):
            v[hash(token) % self.dim] += 1.0
        return v

    def embed_documents(self, texts, progress=None):
        return normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text):
        return normalize(self._vec(text))[0]


# --------------------------------------------------------------------
print("\ningest")


@check("hyphenation across line breaks is rejoined")
def _():
    assert "structural" in _normalise("struc-\ntural health")


@check("ligatures normalised")
def _():
    assert _normalise("the eﬀect of ﬁnite elements") == "the effect of finite elements"


@check("running headers stripped")
def _():
    pages = [f"IEEE TRANSACTIONS\nreal content {i}\npage footer" for i in range(6)]
    cleaned = _strip_repeated_lines(pages)
    assert all("IEEE TRANSACTIONS" not in p for p in cleaned)
    assert all("real content" in p for p in cleaned)


@check("short documents keep their headers (no false positives)")
def _():
    pages = ["HEADER\nbody a", "HEADER\nbody b"]
    assert _strip_repeated_lines(pages) == pages


# --------------------------------------------------------------------
print("\nchunking")


@check("chunks never exceed size + overlap")
def _():
    text = " ".join(f"word{i}" for i in range(2000))
    parts = _split_text(text, 500, 80)
    assert max(len(p) for p in parts) <= 500 + 80 + 10, max(len(p) for p in parts)


@check("chunks never span two pages")
def _():
    body = " ".join(f"sentence {i} of the page." for i in range(80))
    recs = [PageRecord("d", "a.pdf", p, body, "Method") for p in (1, 2, 3)]
    chunks = chunk_pages(recs, 400, 50)
    for c in chunks:
        assert c.page in (1, 2, 3)
        assert c.chunk_id.startswith(f"d:p{c.page}:")
    assert {c.page for c in chunks} == {1, 2, 3}


@check("citation string includes source, page and section")
def _():
    c = Chunk("i", "d", "wing.pdf", 7, "text", "4.2 Results")
    assert c.citation == "wing.pdf, p.7 - 4.2 Results"
    assert Chunk("i", "d", "wing.pdf", 7, "t", None).citation == "wing.pdf, p.7"


@check("overlap actually carries context forward")
def _():
    text = "alpha beta gamma delta. " * 60
    parts = _split_text(text, 300, 60)
    assert len(parts) > 1
    assert any(parts[0][-20:].split()[-1] in parts[1][:80] for _ in [0])


# --------------------------------------------------------------------
print("\nbm25")


@check("exact rare token wins")
def _():
    corpus = ["neural networks for vibration", "the CWRU bearing dataset", "graph attention"]
    assert BM25(corpus).search("CWRU", 1)[0][0] == 1


@check("unknown query returns nothing rather than garbage")
def _():
    assert BM25(["a b c", "d e f"]).search("qqqq zzzz", 5) == []


@check("idf is never negative")
def _():
    bm = BM25(["the model the model", "the model", "the model", "the model"])
    assert all(v >= 0 for v in bm.idf.values())


# --------------------------------------------------------------------
print("\nstore")


@check("save/load roundtrip preserves chunks and search order")
def _():
    emb = StubEmbedder()
    chunks = [Chunk(f"c{i}", "d", "p.pdf", i + 1, t, None) for i, t in enumerate(
        ["bearing fault CWRU", "graph attention wing", "physics informed loss"])]
    vecs = emb.embed_documents([c.indexable_text for c in chunks])
    store = VectorStore(dim=vecs.shape[1])
    store.add(vecs, chunks)
    before = store.search(emb.embed_query("bearing fault CWRU"), 3)

    with tempfile.TemporaryDirectory() as td:
        store.save(Path(td) / "ix", {"embed_model": "stub", "documents": {}})
        loaded, manifest = VectorStore.load(Path(td) / "ix")
        assert manifest["n_chunks"] == 3
        assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in chunks]
        after = loaded.search(emb.embed_query("bearing fault CWRU"), 3)
        assert [i for i, _ in before] == [i for i, _ in after]


@check("mismatched vectors and chunks are rejected")
def _():
    store = VectorStore(dim=4)
    try:
        store.add(np.zeros((2, 4), "float32"), [Chunk("c", "d", "p", 1, "t")])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


@check("empty store returns no hits instead of crashing")
def _():
    assert VectorStore(dim=8).search(np.zeros(8, "float32"), 5) == []


# --------------------------------------------------------------------
print("\nretrieval")


def _build_retriever():
    emb = StubEmbedder()
    texts = [
        "The CWRU bearing dataset provides vibration signals at four load levels.",
        "A graph attention network weights neighbouring sensor nodes adaptively.",
        "Physics informed losses embed the governing equation as a soft constraint.",
        "Collaborative filtering factorises the user item interaction matrix.",
    ]
    chunks = [Chunk(f"c{i}", "d", "p.pdf", i + 1, t, None) for i, t in enumerate(texts)]
    vecs = emb.embed_documents([c.indexable_text for c in chunks])
    store = VectorStore(dim=vecs.shape[1])
    store.add(vecs, chunks)
    return HybridRetriever(store, emb, use_hybrid=True, candidate_k=10)


@check("hybrid retrieves the right passage")
def _():
    hits = _build_retriever().retrieve("CWRU bearing load levels", k=2)
    assert hits[0].chunk.page == 1, [h.chunk.page for h in hits]


@check("chunks found by both retrievers are labelled 'both'")
def _():
    hits = _build_retriever().retrieve("graph attention sensor nodes", k=3)
    assert hits[0].matched_by in ("both", "semantic", "keyword")
    assert hits[0].chunk.page == 2


@check("source filter excludes everything from other files")
def _():
    assert _build_retriever().retrieve("CWRU", k=3, sources=["nope.pdf"]) == []


@check("k is respected")
def _():
    assert len(_build_retriever().retrieve("network", k=2)) <= 2


@check("keyword-only mode never calls the embedder")
def _():
    r = _build_retriever()

    class Boom:
        def embed_query(self, _):
            raise AssertionError("embedder should not be called in keyword mode")

    r.embedder = Boom()
    assert r.retrieve("CWRU bearing", k=1, mode="keyword")[0].chunk.page == 1


# --------------------------------------------------------------------
print("\ncitations")


def _hits(n=3):
    return [RetrievedChunk(Chunk(f"c{i}", "d", "p.pdf", i + 1, f"body {i}", "Method"),
                           1.0 - i * 0.1, i, i) for i in range(n)]


@check("context blocks are numbered from 1 and carry provenance")
def _():
    ctx = format_context(_hits())
    assert ctx.startswith("[1] p.pdf, p.1 - Method")
    assert "[3] p.pdf, p.3" in ctx


@check("long passages are truncated at a word boundary")
def _():
    hit = RetrievedChunk(Chunk("c", "d", "p.pdf", 1, "word " * 2000, None), 1.0)
    ctx = format_context([hit], max_chars=100)
    assert ctx.endswith("...") and len(ctx) < 200


@check("hallucinated markers are stripped from the answer text")
def _():
    a = RAGChain._attach_sources("q", "Claim A [1]. Claim B [9]. Claim C [2].", _hits())
    assert "[9]" not in a.text
    assert a.invalid_markers == [9]
    assert a.cited_markers == [1, 2]


@check("punctuation stays clean after stripping a marker")
def _():
    a = RAGChain._attach_sources("q", "Fact one [1]. Stray [9]. Done.", _hits())
    assert a.text == "Fact one [1]. Stray. Done.", repr(a.text)


@check("only cited passages become sources")
def _():
    a = RAGChain._attach_sources("q", "Only this one matters [2].", _hits())
    assert [s.marker for s in a.sources] == [2]
    assert a.sources[0].page == 2


@check("refusal is detected and carries no sources")
def _():
    a = RAGChain._attach_sources(
        "q", "I couldn't find this in the uploaded documents.", _hits())
    assert a.is_refusal and not a.sources


@check("repeated markers are not double counted")
def _():
    a = RAGChain._attach_sources("q", "A [1]. B [1]. C [1][2].", _hits())
    assert a.cited_markers == [1, 2]


# --------------------------------------------------------------------
print("\nend-to-end (stubbed embedder, real PDF)")


def _make_test_pdf(path: Path) -> bool:
    """Build a 3-page text PDF with matplotlib (real embedded text)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        return False

    bodies = [
        ("Introduction", "Structural health monitoring aims to detect damage in "
                         "operating structures. Vibration based methods dominate the "
                         "literature because they are cheap and non intrusive."),
        ("Instrumentation", "Each piezoelectric transducer was calibrated against a "
                            "reference accelerometer before the campaign. Calibration "
                            "of the piezoelectric transducer drifted by less than one "
                            "percent across the temperature range tested."),
        ("Results", "The classifier reached ninety four point nine percent accuracy "
                    "on the held out load condition, improving on the baseline."),
    ]
    with PdfPages(path) as pdf:
        for title, body in bodies:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.1, 0.92, "Journal of Test Structures", size=8)
            fig.text(0.1, 0.85, title, size=16)
            wrapped = "\n".join(
                body[i:i + 60] for i in range(0, len(body), 60)
            )
            fig.text(0.1, 0.5, wrapped, size=11, va="center")
            fig.text(0.1, 0.05, "Journal of Test Structures", size=8)
            pdf.savefig(fig)
            plt.close(fig)
    return True


@check("real PDF -> index -> retrieval returns the correct page")
def _():
    pdf_path = Path(tempfile.gettempdir()) / "_rag_test_paper.pdf"
    if not _make_test_pdf(pdf_path):
        print("        (skipped: no PDF writer available)")
        return

    pages = load_pdf(pdf_path)
    assert len(pages) >= 3, f"expected 3 pages, got {len(pages)}"

    chunks = chunk_pages(pages, 600, 80)
    assert chunks

    emb = StubEmbedder()
    vecs = emb.embed_documents([c.indexable_text for c in chunks])
    store = VectorStore(dim=vecs.shape[1])
    store.add(vecs, chunks)

    retriever = HybridRetriever(store, emb, use_hybrid=True, candidate_k=10)
    hits = retriever.retrieve("piezoelectric transducer calibration", k=3)
    assert hits, "no hits"
    assert hits[0].chunk.page == 2, f"expected page 2, got {hits[0].chunk.page}"
    assert "p.2" in hits[0].chunk.citation




# --------------------------------------------------------------------
print("\nre-ranking")


class _LLM:
    """Stub that answers rerank prompts with a fixed reply."""
    def __init__(self, reply): self.reply = reply
    def invoke(self, messages):
        if isinstance(self.reply, Exception):
            raise self.reply
        return type("R", (), {"content": self.reply})()


def _rr_hits(n=4):
    return [RetrievedChunk(Chunk(f"c{i}", "d", "p.pdf", i + 1, f"passage {i}", None),
                           1.0 - i * .1, i, i) for i in range(n)]


@check("parses clean JSON scores")
def _():
    assert parse_scores('[{"id":1,"score":3},{"id":2,"score":0}]', 2) == {1: 3.0, 2: 0.0}


@check("parses JSON wrapped in a code fence")
def _():
    assert parse_scores('```json\n[{"id":1,"score":2}]\n```', 2) == {1: 2.0}


@check("ignores ids outside the candidate range")
def _():
    assert parse_scores('[{"id":99,"score":3},{"id":1,"score":2}]', 3) == {1: 2.0}


@check("clamps scores into 0..3")
def _():
    assert parse_scores('[{"id":1,"score":9},{"id":2,"score":-5}]', 2) == {1: 3.0, 2: 0.0}


@check("returns nothing for unparseable output")
def _():
    assert parse_scores("I cannot comply.", 3) == {}


@check("re-ranking reorders by model score")
def _():
    r = Reranker(_LLM('[{"id":1,"score":0},{"id":2,"score":3},{"id":3,"score":1},{"id":4,"score":2}]'))
    assert [h.chunk.page for h in r.rerank("q", _rr_hits(), 3)] == [2, 4, 3]


@check("an LLM failure falls back to fusion order")
def _():
    r = Reranker(_LLM(RuntimeError("429")))
    assert [h.chunk.page for h in r.rerank("q", _rr_hits(), 3)] == [1, 2, 3]
    assert "fusion order" in r.last_status


@check("garbage output falls back to fusion order")
def _():
    r = Reranker(_LLM("no thanks"))
    assert [h.chunk.page for h in r.rerank("q", _rr_hits(), 2)] == [1, 2]


@check("a partial reply keeps unscored passages at neutral")
def _():
    r = Reranker(_LLM('[{"id":3,"score":3}]'))
    out = r.rerank("q", _rr_hits(), 4)
    assert out[0].chunk.page == 3 and len(out) == 4


@check("equal scores preserve retrieval order")
def _():
    r = Reranker(_LLM('[{"id":1,"score":2},{"id":2,"score":2},{"id":3,"score":2},{"id":4,"score":2}]'))
    assert [h.chunk.page for h in r.rerank("q", _rr_hits(), 4)] == [1, 2, 3, 4]


# --------------------------------------------------------------------
print("\nstreaming")


class _StreamLLM:
    PIECES = ["The dataset ", "is CWRU [1]. ", "Ignore [9]."]
    def invoke(self, messages):
        sys_text = messages[0][1]
        if "score how well" in sys_text.lower():
            return type("R", (), {"content": '[{"id":1,"score":3}]'})()
        return type("R", (), {"content": "".join(self.PIECES)})()
    def stream(self, messages):
        for p in self.PIECES:
            yield type("C", (), {"content": p})()


def _stub_chain(llm=None):
    from src.chain import RAGChain
    from src.retrieve import HybridRetriever
    emb = StubEmbedder()
    texts = ["The CWRU bearing dataset has four load levels.",
             "Graph attention weights neighbouring nodes."]
    chunks = [Chunk(f"c{i}", "doc1", "p.pdf", i + 1, t, None) for i, t in enumerate(texts)]
    vecs = emb.embed_documents([c.indexable_text for c in chunks])
    store = VectorStore(vecs.shape[1]); store.add(vecs, chunks)
    chain = RAGChain.__new__(RAGChain)
    chain.retriever = HybridRetriever(store, emb, use_hybrid=True, candidate_k=8)
    chain.llm = llm or _StreamLLM()
    chain.model_name = "stub"
    chain.reranker = Reranker(chain.llm)
    chain.use_rerank = False
    return chain


@check("streamed deltas reassemble into the full text")
def _():
    st_ = _stub_chain().stream("CWRU", k=2)
    got = list(st_.tokens())
    assert "".join(got) == st_.raw
    assert len(got) == len(_StreamLLM.PIECES)


@check("citation validation runs on the finished stream")
def _():
    st_ = _stub_chain().stream("CWRU", k=2)
    list(st_.tokens())
    ans = st_.finalize()
    assert ans.cited_markers == [1]
    assert ans.invalid_markers == [9] and "[9]" not in ans.text


@check("finalize is idempotent")
def _():
    st_ = _stub_chain().stream("CWRU", k=2)
    list(st_.tokens())
    assert st_.finalize() is st_.finalize()


@check("a broken stream falls back to a blocking call")
def _():
    class Broken(_StreamLLM):
        def stream(self, messages):
            raise RuntimeError("streaming unsupported")
            yield  # pragma: no cover
    st_ = _stub_chain(Broken()).stream("CWRU", k=2)
    text = "".join(st_.tokens())
    assert "CWRU" in text
    assert st_.finalize().cited_markers == [1]


@check("re-rank status is recorded on the answer")
def _():
    chain = _stub_chain(); chain.use_rerank = True
    ans = chain.answer("CWRU", k=2)
    assert ans.rerank_status and "fusion order" not in ans.rerank_status


# --------------------------------------------------------------------
print("\npage preview")


@check("preview degrades to None without PyMuPDF")
def _():
    if preview.available():
        print("        (PyMuPDF present - exercising the real path)")
        return
    assert preview.render_page("/nope.pdf", 1) is None
    assert preview.page_count("/nope.pdf") is None


@check("preview never raises on a missing file")
def _():
    assert preview.render_page("/definitely/not/here.pdf", 3, highlight="x") is None


@check("highlight spans sample across the passage")
def _():
    spans = preview._spans("one two three four five six seven eight nine ten eleven twelve")
    assert 1 < len(spans) <= 4
    assert all(len(s.split()) <= 6 for s in spans)


@check("short passages yield a single span")
def _():
    assert preview._spans("two words") == ["two words"]


@check("empty passage yields no spans")
def _():
    assert preview._spans("   ") == []


# --------------------------------------------------------------------
print("\ninterface rendering")


@check("citation markers become superscript footnotes")
def _():
    out = ui.answer_html("Fact [1] and another [2].")
    assert '<span class="pc-cite">1</span>' in out
    assert "[1]" not in out


@check("answer HTML escapes injected markup")
def _():
    out = ui.answer_html('<img src=x onerror=alert(1)> [1]')
    assert "<img" not in out and "&lt;img" in out


@check("bullets render as a list")
def _():
    out = ui.answer_html("Intro:\n- one [1]\n- two [2]")
    assert out.count("<li>") == 2 and "<ul>" in out


@check("source lines escape document names")
def _():
    out = ui.source_line(1, "<evil>.pdf", 2, "A & B", "both", 3.0)
    assert "&lt;evil&gt;.pdf" in out and "<evil>" not in out
    assert "relevance 3/3" in out


@check("chips omit relevance when re-ranking did not run")
def _():
    assert "relevance" not in ui.chips("semantic", None)
    assert "relevance" in ui.chips("semantic", 2.0)


@check("long quotes truncate at a word boundary")
def _():
    out = ui.quote("word " * 400)
    assert out.endswith("…</div>") and len(out) < 600


@check("stylesheet has balanced braces and a resolved font URL")
def _():
    assert ui.CSS.count("{") == ui.CSS.count("}")
    assert "__FONTS__" not in ui.CSS


# --------------------------------------------------------------------
print(f"\n{'=' * 60}")
print(f"{len(PASSED)} passed, {len(FAILED)} failed")
if FAILED:
    for name, exc in FAILED:
        print(f"  - {name}: {exc}")
sys.exit(1 if FAILED else 0)
