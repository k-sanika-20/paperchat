#!/usr/bin/env python3
"""List the Gemini models your key can actually reach.

Model IDs change and Google retires them for new keys without warning, so
this is the tool to reach for whenever you see a 404 NOT_FOUND.

Works with either Google SDK: `google-genai` (new, what langchain-google-genai
4.x installs) or `google-generativeai` (older).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402


def via_new_sdk(key: str):
    from google import genai  # google-genai

    client = genai.Client(api_key=key)
    rows = []
    for m in client.models.list():
        actions = list(getattr(m, "supported_actions", None) or [])
        rows.append((m.name, actions))
    return rows


def via_old_sdk(key: str):
    import google.generativeai as genai  # google-generativeai

    genai.configure(api_key=key)
    return [
        (m.name, list(getattr(m, "supported_generation_methods", []) or []))
        for m in genai.list_models()
    ]


def main() -> int:
    settings.validate()

    rows, source = None, ""
    for fn, label in ((via_new_sdk, "google-genai"), (via_old_sdk, "google-generativeai")):
        try:
            rows = fn(settings.google_api_key)
            source = label
            break
        except ImportError:
            continue
        except Exception as exc:
            print(f"{label}: {exc}", file=sys.stderr)
            continue

    if rows is None:
        print("Neither Google SDK is installed. Run:\n  pip install google-genai")
        return 1

    chat, embed, other = [], [], []
    for name, actions in sorted(rows):
        short = name.split("/")[-1]
        acts = " ".join(actions)
        if "embed" in short.lower() or "embedContent" in acts:
            embed.append((short, acts))
        elif "generateContent" in acts or not acts:
            chat.append((short, acts))
        else:
            other.append((short, acts))

    print(f"\nModels reachable with your key   (via {source})")
    for title, group, var in (
        ("CHAT  — set one of these as CHAT_MODEL in .env", chat, "CHAT_MODEL"),
        ("EMBED — set one of these as EMBED_MODEL in .env", embed, "EMBED_MODEL"),
        ("OTHER", other, None),
    ):
        if not group:
            continue
        print(f"\n  {title}")
        for short, acts in group:
            print(f"    {short:<42} {acts}")

    print(f"\nCurrently configured:\n    CHAT_MODEL  = {settings.chat_model}"
          f"\n    EMBED_MODEL = {settings.embed_model}")

    names = {s for s, _ in chat + embed + other}
    for var, val in (("CHAT_MODEL", settings.chat_model), ("EMBED_MODEL", settings.embed_model)):
        if val not in names:
            print(f"\n  ! {var}={val} is NOT in the list above — that is your 404.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
