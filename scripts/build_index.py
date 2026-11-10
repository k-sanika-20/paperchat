from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PDF_DIR, settings  # noqa: E402
from src.pipeline import build_index     # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a FAISS index from PDFs.")
    parser.add_argument("pdfs", nargs="*", help="PDF paths (default: everything in data/pdfs)")
    parser.add_argument("--name", default="default", help="index name")
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--rebuild", action="store_true", help="ignore existing index")
    args = parser.parse_args()

    paths = [Path(p) for p in args.pdfs]
    if not paths:
        paths = sorted(PDF_DIR.glob("*.pdf"))
    if not paths:
        parser.error(f"No PDFs given and {PDF_DIR} is empty.")

    missing = [p for p in paths if not p.exists()]
    if missing:
        parser.error(f"Not found: {', '.join(str(m) for m in missing)}")

    stats = build_index(
        paths,
        index_name=args.name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        rebuild=args.rebuild,
        progress=lambda msg, frac: print(f"[{frac:5.0%}] {msg}", flush=True),
    )
    print("\n--- build complete ---")
    for key, value in stats.items():
        if key == "warnings":
            continue
        print(f"{key:>20}: {value}")
    if stats.get("warnings"):
        print("\n--- warnings ---")
        for w in stats["warnings"]:
            print(f"  ! {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
