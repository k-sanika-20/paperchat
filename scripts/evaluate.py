#!/usr/bin/env python3
"""Evaluation harness.
 
Measures three things a RAG interviewer will ask about:
 
  RETRIEVAL   Recall@k and MRR against gold page numbers. Answers the
              question "did the right passage even reach the model?"
  CITATION    Fraction of answers where every [n] marker is valid AND at
              least one cited passage is a gold page. Catches the failure
              where retrieval worked but the model cited the wrong passage.
  ABSTENTION  On questions deliberately not answerable from the corpus,
              how often the system refuses instead of confabulating.
 
Run:
    python scripts/evaluate.py --index default --questions eval/questions.json
    python scripts/evaluate.py --index default --compare        # all 3 modes
"""
from __future__ import annotations
 
import argparse
import json
import statistics
import sys
import time
from types import SimpleNamespace
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
from src.pipeline import load_chain  # noqa: E402
 
 
def recall_and_rr(hits, gold_pages: set[int], gold_source: str | None):
    """Recall@k over gold pages, plus reciprocal rank of the first gold hit."""
    if not gold_pages:
        return None, None
    found, rr = set(), 0.0
    for rank, hit in enumerate(hits, start=1):
        if gold_source and hit.chunk.source != gold_source:
            continue
        if hit.chunk.page in gold_pages:
            found.add(hit.chunk.page)
            if rr == 0.0:
                rr = 1.0 / rank
    return len(found) / len(gold_pages), rr
 
 
class QuotaExhausted(RuntimeError):
    """The per-DAY generate quota is gone. Waiting will not help today."""
 
 
def _is_daily_quota(exc: Exception) -> bool:
    text = str(exc)
    return "PerDay" in text or "RequestsPerDay" in text
 
 
def evaluate(chain, questions: list[dict], k: int, mode: str, verbose: bool,
             rerank: bool | None = None, label: str | None = None,
             retrieval_only: bool = False) -> dict:
    recalls, rrs, latencies = [], [], []
    errors = 0
    citation_ok, citation_total = 0, 0
    invalid_marker_answers = 0
    abstain_correct, abstain_total = 0, 0
 
    for item in questions:
        question = item["question"]
        gold_pages = set(item.get("gold_pages", []))
        gold_source = item.get("source")
        unanswerable = bool(item.get("unanswerable", False))
 
        started = time.time()
 
        if retrieval_only:
            # Recall and MRR are properties of RETRIEVAL. Measuring them
            # needs no generation at all, which matters enormously on a
            # free tier metered at 20 generate calls per DAY: this path
            # spends only cheap embedding calls.
            hits = chain.retriever.retrieve(question, k=k, mode=mode)
            answer = SimpleNamespace(
                all_retrieved=hits, sources=[], is_refusal=False,
                invalid_markers=[], rerank_status=None,
            )
        else:
            try:
                answer = chain.answer(question, k=k, mode=mode, rerank=rerank)
            except Exception as exc:  # noqa: BLE001
                if _is_daily_quota(exc):
                    raise QuotaExhausted(str(exc)) from exc
                errors += 1
                if verbose:
                    print(f"  - {question[:64]:<66} ERROR {type(exc).__name__}")
                continue
        latencies.append(time.time() - started)
 
        if unanswerable:
            abstain_total += 1
            if answer.is_refusal:
                abstain_correct += 1
            status = "REFUSED" if answer.is_refusal else "ANSWERED (bad)"
        else:
            recall, rr = recall_and_rr(answer.all_retrieved, gold_pages, gold_source)
            if recall is not None:
                recalls.append(recall)
                rrs.append(rr)
            citation_total += 1
            cited_pages = {s.page for s in answer.sources}
            if answer.sources and not answer.invalid_markers and (
                not gold_pages or cited_pages & gold_pages
            ):
                citation_ok += 1
            status = f"recall={recall:.2f} rr={rr:.2f} cited={sorted(cited_pages)}"
 
        if answer.invalid_markers:
            invalid_marker_answers += 1
 
        if verbose:
            print(f"  - {question[:64]:<66} {status}")
 
    def mean(xs):
        return round(statistics.mean(xs), 3) if xs else None
 
    return {
        "mode": label or mode,
        "rerank": bool(rerank) and not retrieval_only,
        "retrieval_only": retrieval_only,
        "errors": errors,
        "k": k,
        "n_questions": len(questions),
        f"recall@{k}": mean(recalls),
        "mrr": mean(rrs),
        "citation_precision": (None if retrieval_only else
                               (round(citation_ok / citation_total, 3) if citation_total else None)),
        "answers_with_invalid_markers": invalid_marker_answers,
        "abstention_rate": (None if retrieval_only else
                            (round(abstain_correct / abstain_total, 3) if abstain_total else None)),
        "median_latency_s": round(statistics.median(latencies), 2) if latencies else None,
    }
 
 
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="default")
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "semantic", "keyword"])
    parser.add_argument("--compare", action="store_true", help="run all three retrieval modes")
    parser.add_argument("--rerank", action="store_true",
                        help="run the chosen mode with AND without LLM re-ranking, "
                             "so you can quote the delta")
    parser.add_argument("--no-rerank", action="store_true",
                        help="force re-ranking off for this run")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="measure Recall@k and MRR without generating any "
                             "answers. Spends ZERO chat quota - use this when the "
                             "daily generate limit is gone.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--out", default=None, help="write results JSON here")
    args = parser.parse_args()
 
    path = Path(args.questions)
    if not path.exists():
        parser.error(
            f"{path} not found. Copy eval/questions.example.json to {path} "
            "and fill in questions from your own PDFs."
        )
    questions = json.loads(path.read_text(encoding="utf-8"))
 
    chain, manifest = load_chain(args.index)
    print(f"Index '{args.index}': {manifest.get('n_chunks')} chunks, "
          f"{len(manifest.get('documents', {}))} document(s)\n")
 
    # Each run is (mode, rerank_flag, display_label).
    if args.rerank:
        runs = [(args.mode, False, args.mode + " / no rerank"),
                (args.mode, True, args.mode + " / rerank")]
    elif args.compare:
        rr = None if not args.no_rerank else False
        runs = [(m, rr, m) for m in ("hybrid", "semantic", "keyword")]
    else:
        runs = [(args.mode, False if args.no_rerank else None, args.mode)]
 
    if args.retrieval_only:
        runs = [(m, False, m) for m in (["hybrid", "semantic", "keyword"]
                                        if args.compare else [args.mode])]
 
    results = []
    exhausted = False
    for mode, rr, label in runs:
        print(f"=== {label} ===")
        try:
            result = evaluate(chain, questions, args.k, mode, not args.quiet,
                              rerank=rr, label=label,
                              retrieval_only=args.retrieval_only)
        except QuotaExhausted as exc:
            exhausted = True
            print(f"\n  DAILY generate quota exhausted - stopping here.\n"
                  f"  {str(exc)[:160]}\n"
                  f"  Results collected so far are kept below.\n"
                  f"  Options:\n"
                  f"    1. python scripts/evaluate.py --index {args.index} "
                  f"--compare --retrieval-only   (no chat calls at all)\n"
                  f"    2. switch CHAT_MODEL in .env to a model with a larger "
                  f"free daily allowance - see: python scripts/list_models.py\n"
                  f"    3. wait for the quota to reset (midnight US Pacific)\n")
            break
        results.append(result)
        print(json.dumps(result, indent=2), "\n")
 
    if len(results) > 1:
        print("=== summary ===")
        header = f"{'run':<22}{'recall':>9}{'mrr':>8}{'cite-prec':>11}{'abstain':>9}{'lat(s)':>9}"
        print(header)
        print("-" * len(header))
        for r in results:
            print(
                f"{r['mode']:<22}"
                f"{str(r[f'recall@{args.k}']):>9}"
                f"{str(r['mrr']):>8}"
                f"{str(r['citation_precision']):>11}"
                f"{str(r['abstention_rate']):>9}"
                f"{str(r['median_latency_s']):>9}"
            )
 
    if args.out and results:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}"
              + (" (partial - quota ran out)" if exhausted else ""))
    return 1 if exhausted and not results else 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
 