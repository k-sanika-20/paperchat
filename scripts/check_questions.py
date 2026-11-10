#!/usr/bin/env python3
#Validate an evaluation question set against a built index.
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from difflib import get_close_matches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings          # noqa: E402
from src.store import VectorStore, index_path  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[0m"


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate an eval question set.")
    ap.add_argument("--index", default="default")
    ap.add_argument("--questions", default="eval/questions.json")
    args = ap.parse_args()

    qpath = Path(args.questions)
    if not qpath.exists():
        print(f"{RED}{qpath} not found.{RESET}")
        print(f"  Start from the template:  cp eval/questions.example.json {qpath}")
        return 1

    try:
        questions = json.loads(qpath.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"{RED}{qpath} is not valid JSON.{RESET}")
        print(f"  {exc}")
        print(f"  {DIM}Usually a trailing comma after the last entry, "
              f"or a missing quote.{RESET}")
        return 1

    if not isinstance(questions, list):
        print(f"{RED}Expected a JSON array of question objects.{RESET}")
        return 1

    try:
        store, manifest = VectorStore.load(index_path(args.index, settings.storage_dir))
    except Exception as exc:
        print(f"{RED}Could not load index '{args.index}': {exc}{RESET}")
        print(f"  Build it first:  python scripts/build_index.py <pdfs> --name {args.index}")
        return 1

    # What the index actually contains -------------------------------
    docs = {d["source"]: d for d in manifest.get("documents", {}).values()}
    pages_with_chunks: dict[str, set[int]] = defaultdict(set)
    for chunk in store.chunks:
        pages_with_chunks[chunk.source].add(chunk.page)

    print(f"\nIndex {DIM}{args.index}{RESET}: {len(store.chunks)} chunks, "
          f"{len(docs)} document(s)")
    for name, meta in sorted(docs.items()):
        covered = len(pages_with_chunks.get(name, ()))
        print(f"  {name:<40} {meta['pages']:>3} pages, "
              f"{meta['chunks']:>4} chunks, {covered:>3} pages retrievable")

    # Per-question checks ---------------------------------------------
    errors: list[str] = []
    warnings: list[str] = []
    answerable = 0
    unanswerable = 0
    per_doc: Counter[str] = Counter()

    print(f"\nQuestions in {qpath}: {len(questions)}\n")

    for i, item in enumerate(questions, start=1):
        tag = f"[{i:>2}]"
        if not isinstance(item, dict):
            errors.append(f"{tag} is not an object")
            continue

        text = (item.get("question") or "").strip()
        if not text:
            errors.append(f"{tag} has no 'question'")
            continue

        short = text if len(text) <= 58 else text[:55] + "..."

        if item.get("unanswerable"):
            unanswerable += 1
            if item.get("gold_pages"):
                warnings.append(f"{tag} is marked unanswerable but has gold_pages")
            print(f"  {DIM}{tag} {short:<60} abstention probe{RESET}")
            continue

        answerable += 1
        source = item.get("source")
        gold = item.get("gold_pages") or []

        if not gold:
            errors.append(f"{tag} has no gold_pages and is not marked unanswerable")
            print(f"  {RED}{tag}{RESET} {short:<60} no gold_pages")
            continue

        if source and source not in docs:
            near = get_close_matches(source, docs.keys(), n=1)
            hint = f" Did you mean '{near[0]}'?" if near else ""
            errors.append(f"{tag} source '{source}' is not in this index.{hint}")
            print(f"  {RED}{tag}{RESET} {short:<60} unknown source")
            continue

        if not source:
            warnings.append(
                f"{tag} has no 'source' - recall will accept a matching page "
                f"from ANY document, which inflates the score")
            # With no source named, a page counts if it exists anywhere.
            retrievable = set().union(*pages_with_chunks.values()) if pages_with_chunks else set()
            where = "any indexed document"
        else:
            per_doc[source] += 1
            retrievable = pages_with_chunks.get(source, set())
            where = f"'{source}'"

        # pages_with_chunks is ground truth: it is exactly what retrieval can
        # return. The manifest's "pages" field counts pages that produced
        # text, which is NOT the highest page number when a blank page sits
        # in the middle - do not use it as a maximum.
        last = max(retrievable) if retrievable else 0
        malformed = [p for p in gold if not isinstance(p, int) or p < 1]
        beyond = [p for p in gold if p not in malformed and p > last]
        empty = [p for p in gold if p not in malformed and p not in beyond
                 and p not in retrievable]

        if malformed:
            errors.append(f"{tag} gold_pages {malformed} are not positive integers")
            print(f"  {RED}{tag}{RESET} {short:<60} bad page value")
        elif beyond:
            errors.append(
                f"{tag} gold page(s) {beyond} are past the end of {where} "
                f"(last page with text is {last})")
            print(f"  {RED}{tag}{RESET} {short:<60} past end of document")
        elif empty:
            errors.append(
                f"{tag} gold page(s) {empty} produced NO chunks in {where} - "
                f"probably a figure-only or near-empty page. Retrieval can "
                f"never return it, so this question scores 0 however good the "
                f"system is. Point at a page with body text on it.")
            print(f"  {RED}{tag}{RESET} {short:<60} page has no text")
        else:
            print(f"  {GREEN}{tag}{RESET} {short:<60} pages {gold}")

    # Shape of the set -------------------------------------------------
    print(f"\n{answerable} answerable, {unanswerable} abstention probe(s)")

    if answerable < 12:
        warnings.append(
            f"only {answerable} answerable questions - below ~12 the numbers "
            f"swing wildly and are not worth quoting")
    if unanswerable < 2:
        warnings.append(
            "fewer than 2 abstention probes - you will have no abstention rate "
            "to report")
    if per_doc and len(per_doc) < min(3, len(docs)):
        warnings.append(
            f"questions only cover {len(per_doc)} of {len(docs)} documents - "
            f"spread them so retrieval has to discriminate between papers")
    if per_doc:
        top, n = per_doc.most_common(1)[0]
        if answerable and n / answerable > 0.6:
            warnings.append(
                f"{n} of {answerable} questions target '{top}' - a lopsided set "
                f"measures one paper, not your system")

    # Report ------------------------------------------------------------
    if warnings:
        print(f"\n{YELLOW}Warnings{RESET}")
        for w in warnings:
            print(f"  - {w}")
    if errors:
        print(f"\n{RED}Errors ({len(errors)}) - fix these before running the eval{RESET}")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(f"\n{GREEN}Question set is valid.{RESET} Next:")
    print(f"  python scripts/evaluate.py --index {args.index} --compare")
    print(f"  python scripts/evaluate.py --index {args.index} --rerank")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
