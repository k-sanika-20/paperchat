#!/usr/bin/env python3
"""Sweep the fusion weight and see where hybrid retrieval actually helps.

Reciprocal Rank Fusion blends two rankings:

    score(d) = w / (60 + dense_rank(d))  +  (1-w) / (60 + lexical_rank(d))

w = 1.0 is pure embeddings, w = 0.0 is pure BM25. The default 0.5 assumes
both retrievers are about equally good on your corpus, which is an
assumption, not a fact. This measures it.

Costs NO chat quota - retrieval metrics need no generation at all. Only
cheap query embeddings, one per question per weight.

    python scripts/sweep_weight.py --index thesis
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import load_retriever  # noqa: E402

_spec = importlib.util.spec_from_file_location("_ev", ROOT / "scripts" / "evaluate.py")
_ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ev)


def score(retriever, questions, k, mode="hybrid"):
    recalls, rrs = [], []
    for item in questions:
        if item.get("unanswerable"):
            continue
        gold = set(item.get("gold_pages") or [])
        if not gold:
            continue
        hits = retriever.retrieve(item["question"], k=k, mode=mode)
        recall, rr = _ev.recall_and_rr(hits, gold, item.get("source"))
        if recall is not None:
            recalls.append(recall)
            rrs.append(rr)
    n = len(recalls)
    return (statistics.mean(recalls) if n else 0.0,
            statistics.mean(rrs) if n else 0.0, n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="default")
    ap.add_argument("--questions", default="eval/questions.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--weights", default="0.0,0.2,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    qpath = Path(args.questions)
    if not qpath.exists():
        print(f"{qpath} not found.")
        return 1
    questions = json.loads(qpath.read_text(encoding="utf-8"))
    weights = [float(w) for w in args.weights.split(",")]

    retriever, manifest = load_retriever(args.index)
    n_answerable = sum(1 for q in questions
                       if not q.get("unanswerable") and q.get("gold_pages"))
    print(f"\nIndex '{args.index}': {manifest.get('n_chunks')} chunks")
    print(f"Sweeping DENSE_WEIGHT over {n_answerable} answerable questions, k={args.k}")
    print("No chat calls are made.\n")

    # Reference points from the pure modes.
    ref = {}
    for mode in ("semantic", "keyword"):
        r, m, _ = score(retriever, questions, args.k, mode=mode)
        ref[mode] = (r, m)
        print(f"  {mode:<22} recall={r:.3f}  mrr={m:.3f}")

    print(f"\n  {'hybrid DENSE_WEIGHT':<22} {'recall':>7} {'mrr':>7}   vs pure semantic")
    print("  " + "-" * 58)

    rows = []
    best = None
    for w in weights:
        retriever.dense_weight = w
        r, m, _ = score(retriever, questions, args.k, mode="hybrid")
        delta = r - ref["semantic"][0]
        arrow = "+" if delta > 0.0005 else ("-" if delta < -0.0005 else "=")
        # A difference smaller than one question is not a difference.
        one_q = 1.0 / max(1, n_answerable)
        note = "  (within one question)" if abs(delta) < one_q else ""
        print(f"  w={w:<20.1f} {r:>7.3f} {m:>7.3f}   {arrow}{abs(delta):.3f}{note}")
        rows.append({"dense_weight": w, "recall": round(r, 4), "mrr": round(m, 4)})
        if best is None or r > best["recall"]:
            best = {"dense_weight": w, "recall": r, "mrr": m}

    sem_r = ref["semantic"][0]
    one_q = 1.0 / max(1, n_answerable)
    print()
    sem_m = ref["semantic"][1]
    gap = best["recall"] - sem_r

    if gap > one_q:
        print(f"  Hybrid at w={best['dense_weight']:.1f} beats pure semantic by "
              f"{gap:.3f} recall - more than one question ({one_q:.3f}). Worth quoting.")
        print(f"  Set DENSE_WEIGHT={best['dense_weight']:.1f} in .env")
    elif gap > 0:
        print(f"  Hybrid peaks at w={best['dense_weight']:.1f}, ahead by only "
              f"{gap:.3f} - less than one question ({one_q:.3f}).")
        print("  That is noise, not a result. Report 'no measurable difference'.")
    elif abs(gap) < 1e-9:
        print(f"  Recall ties with pure semantic at w={best['dense_weight']:.1f}.")
        if best["mrr"] > sem_m + 1e-9:
            print(f"  But MRR is better ({best['mrr']:.3f} vs {sem_m:.3f}), so fusion is "
                  f"ranking the right passage higher even though it finds the same set.")
            print("  That is a defensible reason to keep hybrid on.")
        else:
            print("  No advantage either way on this question set.")
    else:
        print(f"  Pure semantic wins at every weight, by {abs(gap):.3f} at best.")
        print("  That is a real finding, not a failure - report it. It usually means")
        print("  the question set is paraphrase-heavy, so there are few rare literal")
        print("  tokens for BM25 to exploit. Adding lookup-style questions (dataset")
        print("  names, acronyms, exact figures) is what would change it.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"reference": {k: {"recall": v[0], "mrr": v[1]} for k, v in ref.items()},
             "sweep": rows, "n_questions": n_answerable}, indent=2), encoding="utf-8")
        print(f"\n  Wrote {args.out}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
