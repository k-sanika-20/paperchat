"""Version-tolerant imports for the two LangChain classes this project uses.

Resolution is LAZY (PEP 562 module __getattr__): a name is looked up the
first time it is accessed, not when this module is imported. That matters
because an eager version made every import of `src` depend on packages the
project does not use - FAISS is driven directly in store.py, PDFs go
through pypdf in ingest.py, and the chunker is our own - so a missing
optional package broke import chains that never needed it.

Only these two genuinely require LangChain, and both live in
langchain-google-genai, which IS in requirements.txt.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

# name -> ordered (module, attribute) candidates, most likely first.
# Extra paths are cheap insurance against a future package reshuffle.
_CANDIDATES: dict[str, list[tuple[str, str]]] = {
    "ChatGoogleGenerativeAI": [
        ("langchain_google_genai", "ChatGoogleGenerativeAI"),
        ("langchain_google_genai.chat_models", "ChatGoogleGenerativeAI"),
    ],
    "GoogleGenerativeAIEmbeddings": [
        ("langchain_google_genai", "GoogleGenerativeAIEmbeddings"),
        ("langchain_google_genai.embeddings", "GoogleGenerativeAIEmbeddings"),
    ],
}

_cache: dict[str, Any] = {}

__all__ = list(_CANDIDATES)


def __getattr__(name: str) -> Any:
    """Resolve `name` on first access. Raises only if it is really missing."""
    if name in _cache:
        return _cache[name]

    candidates = _CANDIDATES.get(name)
    if candidates is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    tried: list[str] = []
    for module_path, attr in candidates:
        try:
            module = import_module(module_path)
        except ImportError as exc:
            tried.append(f"{module_path} - {exc}")
            continue
        obj = getattr(module, attr, None)
        if obj is not None:
            _cache[name] = obj
            return obj
        tried.append(f"{module_path}.{attr} - attribute not found")

    raise ImportError(
        f"Could not locate {name}.\n  Tried:\n    "
        + "\n    ".join(tried)
        + "\n  Fix with:  pip install -U langchain-google-genai"
    )


def __dir__() -> list[str]:
    return sorted(__all__)
