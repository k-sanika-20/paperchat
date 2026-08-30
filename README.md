# PaperChat

Ask questions about research papers and get answers where every claim cites a
specific page — with that page rendered beside the answer so you can check it.

**Live app:** `https://<your-app>.streamlit.app`
**Stack:** Python · LangChain · Google Gemini · FAISS · Streamlit

> The hosted demo starts empty — upload a PDF in the sidebar and build an index
> to try it. Nothing is stored between container restarts, and no paper is
> redistributed with this repository.

<!-- Add a screenshot or short GIF here. It is the first thing anyone sees. -->

---

## What it does

- Upload PDFs and ask questions in plain language
- Every factual sentence carries a citation to a page number
- Citations are **verified in code** — references the model invents are stripped before you see them
- **Hybrid retrieval**: keyword search (BM25) fused with vector search (FAISS)
- **Re-ranking**: a second pass scores candidates for actual relevance
- **Page previews**: the cited page renders with the quoted passage highlighted
- Answers stream as they generate, and follow-up questions resolve against the conversation

---

## Setup

```bash
git clone https://github.com/<you>/paperchat.git
cd paperchat
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # paste a free Gemini API key into it
python scripts/check_setup.py      # verifies packages, key, and a live API call

streamlit run app.py
```

Get a free API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

If `check_setup.py` reports a model error, run `python scripts/list_models.py`
to see what your key can reach and set `CHAT_MODEL` in `.env` accordingly.

Command line alternative:

```bash
python scripts/build_index.py data/pdfs/*.pdf --name thesis
python scripts/evaluate.py --index thesis --compare
```

---

## How it works

```
PDFs → clean & chunk → embed → FAISS
                                  ↘
Question → BM25 ─────────────────→ Reciprocal Rank Fusion → re-rank → answer
                                                                        ↓
                                                        validate citations → render page
```

Six decisions that shape the results:

**Page-bounded chunking.** A chunk never spans two pages, so a page citation
can never be wrong. Ingestion also strips running headers, rejoins hyphenated
line breaks, and tracks section headings.

**Hybrid retrieval.** Embeddings handle paraphrase well but lose rare literal
tokens — dataset names, acronyms, equation numbers — which is exactly what
people search papers for. BM25 scores those highly because they are rare.

**Reciprocal Rank Fusion.** Cosine similarity and BM25 scores live on different
scales, so averaging them needs fragile per-query normalisation. RRF uses only
rank position: `score(d) = Σ w / (60 + rank(d))`.

**Re-ranking.** Retrieval optimises recall — get the right passage into the top
18. A second pass has the model score each candidate 0–3 for whether it actually
answers the question, and keeps the best few. Falls back to fusion order on any
failure.

**Verified citations.** Passages go to the model pre-numbered. Afterwards the
`[n]` markers are parsed back out and checked against the passage list; anything
out of range is deleted from the answer and reported.

**Page previews.** Stored text is normalised and never matches the raw PDF
byte-for-byte, so the renderer samples several short word-runs from the passage
and highlights whichever lands first.

BM25 is implemented from scratch in `src/bm25.py` (~60 lines) rather than
imported, so the lexical half of retrieval is fully inspectable.

---

## Results

15-question benchmark (12 answerable, 3 abstention probes) over 3 papers,
240 chunks. Retrieval metrics via `scripts/evaluate.py --retrieval-only`,
which makes no generation calls.

| Retrieval | Recall@5 | MRR |
|---|---|---|
| Keyword only (BM25) | 0.583 | 0.558 |
| Hybrid (RRF, w = 0.5) | 0.792 | 0.715 |
| Semantic only | **0.875** | **0.889** |

Sweeping the RRF fusion weight with `scripts/sweep_weight.py` gives a
monotonic curve — recall climbs from 0.583 at pure BM25 to 0.875 at pure
semantic across nine points, with no weight where hybrid wins:

```
w=0.0  0.583    w=0.4  0.792    w=0.7  0.833    w=0.9  0.875
w=0.2  0.667    w=0.5  0.792    w=0.8  0.833    w=1.0  0.875
```

**Interpretation.** BM25 adds no measurable value here. The three papers all
concern vibration and structural monitoring, so they share nearly all their
technical vocabulary and inverse document frequency has little to discriminate
on; the question set is also paraphrase-heavy, with few rare literal tokens for
lexical matching to exploit. Hybrid retrieval stays available as a mode — it
should help on a lexically diverse corpus, or with exact-match queries like part
numbers and dataset names — but `DENSE_WEIGHT` defaults to 1.0 because that is
what measured best.

Every gap in the table exceeds one question (1/12 = 0.083), and the monotonic
trend across the sweep makes the direction trustworthy even though individual
differences are small.

Citation precision and abstention rate need answer generation and are not yet
measured: the Gemini free tier allows 20 generate requests per day.

Reproduce:

```bash
python scripts/check_questions.py --index thesis      # validate the question set
python scripts/evaluate.py --index thesis --compare --retrieval-only
python scripts/sweep_weight.py --index thesis
```

---

## Configuration

Everything lives in `.env` (see `.env.example`):

| Variable | Default | |
|---|---|---|
| `CHAT_MODEL` | `gemini-3.6-flash` | Change if your key lacks access |
| `EMBED_MODEL` | `gemini-embedding-001` | Changing it rebuilds the index |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1000 / 150 | Smaller chunks, sharper citations |
| `TOP_K` | 5 | Passages shown to the model |
| `DENSE_WEIGHT` | 0.5 | 0 = pure BM25, 1 = pure embeddings |
| `USE_RERANK` | true | The relevance pass; one extra call |

---

## Project layout

```
app.py              Streamlit app
src/
  ingest.py         PDF → clean, page-tagged text
  chunking.py       page-bounded splitting
  bm25.py           Okapi BM25, from scratch
  embeddings.py     batched Gemini embeddings
  store.py          FAISS index + manifest
  retrieve.py       hybrid retrieval, RRF
  rerank.py         LLM relevance re-ranking
  preview.py        cited-page rendering
  chain.py          generation, streaming, citation validation
  ui.py             stylesheet and render helpers
scripts/            setup checks, indexing, evaluation
tests/              54 offline tests
```

---

## Tests

```bash
python tests/test_pipeline.py
```

54 tests, no API key needed — the embedder is stubbed, so the suite burns no quota.

---

## Notes

- Scanned PDFs need OCR first (`ocrmypdf in.pdf out.pdf`)
- If `faiss-cpu` will not install, the project falls back to an exact numpy
  search automatically — same results, slower past ~50k chunks
- Page previews need `pymupdf`; without it everything else works unchanged
