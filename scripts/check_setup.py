#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REQUIRED = [
    ("streamlit", "pip install streamlit"),
    ("pypdf", "pip install pypdf"),
    ("numpy", "pip install numpy"),
    ("langchain_google_genai", "pip install langchain-google-genai"),
]
OPTIONAL = [
    ("faiss", "pip install faiss-cpu   (falls back to numpy if absent)"),
    ("dotenv", "pip install python-dotenv   (needed to read .env)"),
]

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"


def check(name: str, hint: str, required: bool) -> bool:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "")
        print(f"{GREEN}  ok  {RESET} {name} {version}")
        return True
    except ImportError:
        colour = RED if required else YELLOW
        tag = "MISSING" if required else "absent "
        print(f"{colour}  {tag}{RESET} {name}  ->  {hint}")
        return not required


def _hint(exc: Exception, var: str) -> None:
    """Point at the likely cause instead of guessing 'it must be the model'."""
    msg = str(exc).lower()
    if "could not locate" in msg or "no module named" in msg:
        print("           A package is missing. Run:")
        print("             pip install -r requirements.txt")
    elif "404" in msg or "not found" in msg or "not supported" in msg:
        # Google's own message usually names the replacement - surface it loudly.
        import re as _re
        hint = _re.search(r"use models/([\w.-]+)", str(exc))
        if hint:
            print(f"           Google says use '{hint.group(1)}' instead.")
            print(f"           Set this line in your .env:   {var}={hint.group(1)}")
        else:
            print(f"           Your key cannot reach that model. Set {var} in .env to one it can.")
        print("           To see everything your key can reach:  python scripts/list_models.py")
    elif "429" in msg or "quota" in msg or "rate" in msg:
        print("           Rate limited. Wait a minute and re-run - this is not a config problem.")
    elif "api key" in msg or "401" in msg or "403" in msg or "permission" in msg:
        print("           The key was rejected. Check for stray spaces or quotes in .env,")
        print("           and that the key is active at https://aistudio.google.com/apikey")
    else:
        print(f"           If this is a model-name error, set {var} in .env to a model")
        print("           your key can access.")


def main() -> int:
    print(f"\nPython {sys.version.split()[0]}\n")
    print("Required packages:")
    # list() first: `all(generator)` short-circuits and hides later failures
    ok = all([check(n, h, True) for n, h in REQUIRED])
    print("\nOptional packages:")
    for n, h in OPTIONAL:
        check(n, h, False)

    print("\nCredentials:")
    from src.config import settings
    if settings.google_api_key:
        key = settings.google_api_key
        print(f"{GREEN}  ok  {RESET} GOOGLE_API_KEY set ({key[:6]}...{key[-4:]})")
    else:
        ok = False
        print(f"{RED}  MISSING{RESET} GOOGLE_API_KEY -> copy .env.example to .env and add a key")
        print("           get one free at https://aistudio.google.com/apikey")

    print("\nModels configured:")
    print(f"        chat  : {settings.chat_model}")
    print(f"        embed : {settings.embed_model}")

    if ok:
        print("\nLive API test:")

        # 1. embeddings
        try:
            from src.embeddings import EmbeddingClient
            client = EmbeddingClient()
            print(f"{GREEN}  ok  {RESET} embeddings respond, dimension = {client.dim}")
        except Exception as exc:
            ok = False
            print(f"{RED}  FAILED{RESET} embeddings: {exc}")
            _hint(exc, "EMBED_MODEL")

        # 2. chat - a separate model and a separate quota, so test it separately
        if ok:
            try:
                from src.compat import ChatGoogleGenerativeAI
                from src.config import settings as cfg
                llm = ChatGoogleGenerativeAI(
                    model=cfg.chat_model,
                    temperature=0,
                    google_api_key=cfg.google_api_key,
                )
                reply = llm.invoke([("human", "Reply with the single word: ready")])
                text = str(getattr(reply, "content", reply)).strip()
                print(f"{GREEN}  ok  {RESET} chat model responds ({text[:40]!r})")
            except Exception as exc:
                ok = False
                print(f"{RED}  FAILED{RESET} chat: {exc}")
                _hint(exc, "CHAT_MODEL")

        # 3. vector backend actually in use
        if ok:
            try:
                from src.store import VectorStore
                print(f"{GREEN}  ok  {RESET} vector backend: {VectorStore(dim=4).backend}")
            except Exception as exc:
                print(f"{YELLOW}  warn {RESET} vector backend: {exc}")

    print(f"\n{'All good - run: streamlit run app.py' if ok else 'Fix the items above, then re-run.'}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
