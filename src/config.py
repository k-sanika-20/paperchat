"""Central configuration. Everything tunable lives here or in .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional at runtime
    pass


def _secret(name: str, default: str = "") -> str:
    """Read config from Streamlit secrets first, then the environment.

    Local development uses a .env file. Streamlit Community Cloud has no
    filesystem to put one on, so it injects st.secrets instead. Checking
    secrets first means the same code deploys unchanged.
    """
    try:
        import streamlit as st  # noqa: PLC0415

        # Accessing st.secrets with no secrets file raises, so guard it.
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # not running under Streamlit, or no secrets configured
        pass
    return os.getenv(name, default)

ROOT = Path(__file__).resolve().parent.parent
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", ROOT / "storage"))
PDF_DIR = Path(os.getenv("PDF_DIR", ROOT / "data" / "pdfs"))

# Community Cloud gives every app a fresh container on restart, so anything
# written to disk is lost. The UI uses this to warn instead of pretending
# a built index will still be there tomorrow.
IS_EPHEMERAL = bool(os.getenv("STREAMLIT_RUNTIME_ENV") or os.getenv("HOSTNAME", "").startswith("streamlit"))


def _int(name: str, default: int) -> int:
    try:
        return int(_secret(name, str(default)))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_secret(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    # --- credentials -------------------------------------------------
    google_api_key: str = field(
        default_factory=lambda: _secret("GOOGLE_API_KEY")
    )
    # Optional. If set, the deployed app asks for it before doing anything.
    # A public Streamlit app with your key in it is a public Gemini quota.
    app_password: str = field(default_factory=lambda: _secret("APP_PASSWORD"))

    # --- models ------------------------------------------------------
    # Both are overridable so the project survives Gemini model renames.
    chat_model: str = field(
        default_factory=lambda: _secret("CHAT_MODEL", "gemini-3.6-flash")
    )
    embed_model: str = field(
        default_factory=lambda: _secret("EMBED_MODEL", "gemini-embedding-001")
    )
    temperature: float = field(default_factory=lambda: _float("TEMPERATURE", 0.1))

    # --- chunking ----------------------------------------------------
    chunk_size: int = field(default_factory=lambda: _int("CHUNK_SIZE", 1000))
    chunk_overlap: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 150))

    # --- retrieval ---------------------------------------------------
    top_k: int = field(default_factory=lambda: _int("TOP_K", 5))
    # How many candidates each retriever contributes before fusion.
    candidate_k: int = field(default_factory=lambda: _int("CANDIDATE_K", 20))
    # 0.0 = pure BM25, 1.0 = pure dense.
    # Default is 1.0 because that is what measured best on the benchmark in
    # the README: sweeping this weight showed recall rising monotonically
    # from 0.583 (pure BM25) to 0.875 (pure dense), with no crossover. On a
    # lexically diverse corpus the balanced 0.5 is the better starting point.
    dense_weight: float = field(default_factory=lambda: _float("DENSE_WEIGHT", 1.0))
    use_hybrid: bool = field(
        default_factory=lambda: _secret("USE_HYBRID", "true").lower() == "true"
    )
    # Reciprocal Rank Fusion smoothing constant (Cormack et al., 2009).
    rrf_k: int = field(default_factory=lambda: _int("RRF_K", 60))

    # --- re-ranking --------------------------------------------------
    # A precision pass over the recall pass: the model reads the fused
    # candidates and scores each for actual relevance. One extra call.
    use_rerank: bool = field(
        default_factory=lambda: _secret("USE_RERANK", "true").lower() == "true"
    )
    # How many fused candidates to hand the re-ranker.
    rerank_candidates: int = field(default_factory=lambda: _int("RERANK_CANDIDATES", 18))
    # Per-passage truncation when building the re-rank prompt.
    rerank_max_chars: int = field(default_factory=lambda: _int("RERANK_MAX_CHARS", 600))
    # Passages scoring below this are dropped entirely (0-3 scale).
    rerank_min_score: float = field(default_factory=lambda: _float("RERANK_MIN_SCORE", 1.0))

    # --- source previews ---------------------------------------------
    # Keep a copy of each PDF alongside the index so cited pages can be
    # rendered later. Costs disk; without it there is nothing to show.
    store_source_pdfs: bool = field(
        default_factory=lambda: _secret("STORE_SOURCE_PDFS", "true").lower() == "true"
    )
    preview_dpi: int = field(default_factory=lambda: _int("PREVIEW_DPI", 110))

    # --- embedding throughput ---------------------------------------
    embed_batch_size: int = field(default_factory=lambda: _int("EMBED_BATCH_SIZE", 32))
    embed_max_retries: int = field(default_factory=lambda: _int("EMBED_MAX_RETRIES", 8))
    # Gemini's free tier meters embeddings PER TEXT at 100/minute. Sit under
    # it deliberately rather than sprinting into a 429 and backing off.
    embed_rpm: int = field(default_factory=lambda: _int("EMBED_RPM", 85))

    # --- paths -------------------------------------------------------
    storage_dir: Path = field(default_factory=lambda: STORAGE_DIR)

    def validate(self) -> None:
        if not self.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Create a .env file (copy .env.example) "
                "and paste a key from https://aistudio.google.com/apikey"
            )
        # langchain-google-genai reads this env var directly in some versions.
        os.environ.setdefault("GOOGLE_API_KEY", self.google_api_key)


settings = Settings()
