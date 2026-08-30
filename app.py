"""PaperChat - retrieval-augmented question answering over research PDFs.

Run:  streamlit run app.py

This file is about behaviour. The stylesheet and the small HTML helpers
live in src/ui.py so the design can change without touching the logic.
"""
from __future__ import annotations

import hmac
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import preview, ui                            # noqa: E402
from src.config import IS_EPHEMERAL, settings          # noqa: E402
from src.pipeline import build_index, load_chain       # noqa: E402
from src.store import index_path, list_indexes         # noqa: E402

st.set_page_config(
    page_title=ui.APP_NAME,
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject(st)


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


# ---------------------------------------------------------------------
# state
# ---------------------------------------------------------------------
for key, default in (
    ("messages", []),
    ("chain", None),
    ("manifest", {}),
    ("active_index", None),
):
    st.session_state.setdefault(key, default)


def load_index_into_state(name: str) -> None:
    with st.spinner("Loading " + name):
        chain, manifest = load_chain(name)
    st.session_state.chain = chain
    st.session_state.manifest = manifest
    st.session_state.active_index = name
    st.session_state.messages = []


# ---------------------------------------------------------------------
# access gate
# ---------------------------------------------------------------------
def require_password() -> None:
    """Optional shared-password gate for public deployments.

    Skipped entirely when APP_PASSWORD is unset, so local development is
    unaffected. Not real auth - it exists so a public Streamlit URL found
    by a crawler cannot drain the Gemini quota.
    """
    if not settings.app_password or st.session_state.get("authed"):
        return

    html(ui.masthead())
    html('<div class="pc-empty">This instance is password protected '
         "so that its API quota stays with its owner.</div>")
    entered = st.text_input("Password", type="password", label_visibility="collapsed")
    if entered:
        if hmac.compare_digest(entered, settings.app_password):
            st.session_state.authed = True
            st.rerun()
        else:
            html('<div class="pc-warn">Incorrect password.</div>')
    st.stop()


require_password()


# ---------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    html('<div class="pc-name" style="font-size:21px">' + ui.APP_NAME + "</div>")

    if not settings.google_api_key:
        html('<div class="pc-warn">No <code>GOOGLE_API_KEY</code> found. '
             "Copy <code>.env.example</code> to <code>.env</code> and add a key from "
             '<a href="https://aistudio.google.com/apikey">Google AI Studio</a>.</div>')
        st.stop()

    if IS_EPHEMERAL:
        html('<div class="pc-warn">Hosted demo. The container restarts '
             "periodically and saved indexes go with it - rebuild from your PDFs "
             "after a restart.</div>")

    # -- add documents -------------------------------------------------
    html('<div class="pc-side-h">Add documents</div>')
    uploads = st.file_uploader(
        "PDFs", type="pdf", accept_multiple_files=True, label_visibility="collapsed"
    )
    index_name = st.text_input("Collection name", value="default")
    rebuild = st.checkbox(
        "Rebuild from scratch", value=False,
        help="Off means only new or changed PDFs are embedded.",
    )

    if st.button("Build index", type="primary", disabled=not uploads,
                 use_container_width=True):
        tmp_dir = Path(tempfile.mkdtemp(prefix="paperchat_"))
        paths = []
        for upload in uploads:
            dest = tmp_dir / upload.name
            dest.write_bytes(upload.getbuffer())
            paths.append(dest)

        bar = st.progress(0.0, text="Starting")
        try:
            stats = build_index(
                paths, index_name=index_name, rebuild=rebuild,
                progress=lambda m, f: bar.progress(min(max(f, 0.0), 1.0), text=m),
            )
        except Exception as exc:  # surface the real error
            bar.empty()
            html('<div class="pc-warn">Indexing failed: ' + ui.esc(exc) + "</div>")
        else:
            bar.empty()
            html('<div class="pc-facts"><b>' + str(stats["n_chunks"]) + "</b> chunks from <b>"
                 + str(stats["n_documents"]) + "</b> document(s)<br><b>"
                 + str(stats["new_chunks"]) + "</b> newly embedded in "
                 + str(stats["seconds"]) + "s</div>")
            if stats["skipped"]:
                html('<div class="pc-facts">unchanged: ' + ui.esc(", ".join(stats["skipped"]))
                     + "</div>")
            load_index_into_state(index_name)

    # -- open a collection ---------------------------------------------
    html('<div class="pc-side-h">Open a collection</div>')
    available = list_indexes(settings.storage_dir)
    if available:
        current = st.session_state.active_index
        chosen = st.selectbox(
            "Saved", available, label_visibility="collapsed",
            index=available.index(current) if current in available else 0,
        )
        if st.button("Open", use_container_width=True):
            load_index_into_state(chosen)
    else:
        html('<div class="pc-facts">Nothing saved yet.</div>')

    # -- retrieval ------------------------------------------------------
    html('<div class="pc-side-h">Retrieval</div>')
    mode = st.radio(
        "Mode", ["hybrid", "semantic", "keyword"], index=0, horizontal=True,
        help="hybrid fuses BM25 and embeddings with Reciprocal Rank Fusion.",
    )
    use_rerank = st.checkbox(
        "Re-rank with the model", value=settings.use_rerank,
        help="Fetch more candidates, then have the model score each for relevance "
             "and keep the best. One extra call; usually worth it.",
    )
    top_k = st.slider("Passages shown to the model", 3, 12, settings.top_k)
    dense_weight = st.slider(
        "Semantic weight in fusion", 0.0, 1.0, settings.dense_weight, 0.1,
        disabled=(mode != "hybrid"),
    )
    show_pages = st.checkbox(
        "Show cited pages", value=preview.available(),
        disabled=not preview.available(),
        help="Renders the cited PDF page with the passage highlighted."
             if preview.available() else "Install PyMuPDF to enable: pip install pymupdf",
    )

    # -- collection facts -----------------------------------------------
    manifest = st.session_state.manifest
    if manifest:
        html('<div class="pc-side-h">This collection</div>')
        html('<div class="pc-facts">'
             "chunks &nbsp;<b>" + str(manifest.get("n_chunks", "?")) + "</b><br>"
             "embed &nbsp;&nbsp;<b>" + ui.esc(manifest.get("embed_model", "?")) + "</b><br>"
             "chat &nbsp;&nbsp;&nbsp;<b>" + ui.esc(settings.chat_model) + "</b><br>"
             "built &nbsp;&nbsp;<b>" + ui.esc(manifest.get("built_at", "?")) + "</b>"
             "</div>")
        docs = list(manifest.get("documents", {}).values())
        if docs:
            with st.expander(str(len(docs)) + " document(s)"):
                for doc in docs:
                    html('<div class="pc-doc">' + ui.esc(doc["source"])
                         + " <span>" + str(doc["pages"]) + "p / "
                         + str(doc["chunks"]) + " chunks</span></div>")


# ---------------------------------------------------------------------
# main column
# ---------------------------------------------------------------------
html(ui.masthead())

if st.session_state.chain is None:
    html('<div class="pc-empty"><b>Add a paper to begin.</b><br>'
         "Upload PDFs in the sidebar and build an index, or open a collection "
         "you built earlier. Answers come back with a citation on every claim, "
         "and you can open the page each one came from.</div>")
    html('<div class="pc-empty" style="font-size:13px;margin-top:-28px">'
         "No sidebar? Use the arrow at the top left of the window to open it."
         "</div>")
    st.stop()

doc_names = sorted(
    {d["source"] for d in st.session_state.manifest.get("documents", {}).values()}
)
selected_sources = st.multiselect(
    "Restrict to particular documents", doc_names, default=[],
)


def render_sources(sources, index_dir: Path, show_previews: bool) -> None:
    """Sources as footnotes: rule, marker, provenance, evidence."""
    if not sources:
        return
    html('<div class="pc-sources-head"><span>Sources</span>'
         "<span>" + str(len(sources)) + " cited</span></div>")

    for src in sources:
        html('<div class="pc-src">'
             + ui.source_line(src["marker"], src["source"], src["page"],
                              src.get("section"), src["matched_by"],
                              src.get("rerank_score"))
             + ui.quote(src["text"]) + "</div>")

        if not (show_previews and src.get("doc_id")):
            continue
        pdf = preview.source_path(index_dir, src["doc_id"])
        if not pdf.exists():
            continue
        with st.expander("Open page " + str(src["page"]) + " of " + src["source"]):
            png = preview.render_page(pdf, src["page"], highlight=src["text"])
            if png:
                st.image(png, use_container_width=True)
            else:
                html('<div class="pc-facts">Could not render this page.</div>')


index_dir = index_path(st.session_state.active_index, settings.storage_dir)

# -- replay the conversation -------------------------------------------
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user" and i:
        st.divider()
    if message["role"] == "user":
        html('<div class="pc-q">' + ui.esc(message["content"]) + "</div>")
    else:
        html(ui.answer_html(message["content"]))
        render_sources(message.get("sources", []), index_dir, show_pages)
        if message.get("note"):
            st.caption(message["note"])

# -- new question -------------------------------------------------------
placeholder = "What dataset did they evaluate on, and why that one?"
if question := st.chat_input(placeholder):
    st.session_state.messages.append({"role": "user", "content": question})
    html('<div class="pc-q">' + ui.esc(question) + "</div>")

    chain = st.session_state.chain
    chain.retriever.dense_weight = dense_weight
    history = [(m["role"], m["content"]) for m in st.session_state.messages[:-1]]

    try:
        with st.spinner("Searching"):
            stream = chain.stream(
                question, k=top_k, sources=selected_sources or None,
                mode=mode, history=history, rerank=use_rerank,
            )
        # Stream the raw text for responsiveness, then replace it with the
        # citation-validated version - you cannot know a marker is invalid
        # until the text is complete.
        slot = st.empty()
        with slot.container():
            st.write_stream(stream.tokens())
        answer = stream.finalize()
        slot.empty()
        html(ui.answer_html(answer.text))
    except Exception as exc:
        html('<div class="pc-warn">Request failed: ' + ui.esc(exc) + "</div>")
        st.stop()

    payload = [
        {
            "marker": s.marker, "source": s.source, "page": s.page,
            "section": s.section, "text": s.text, "matched_by": s.matched_by,
            "doc_id": s.doc_id, "rerank_score": s.rerank_score,
        }
        for s in answer.sources
    ]
    render_sources(payload, index_dir, show_pages)

    notes = []
    if answer.rerank_status:
        notes.append("re-rank: " + answer.rerank_status)
    if answer.invalid_markers:
        notes.append("removed " + str(len(answer.invalid_markers))
                     + " unsupported citation marker(s): " + str(answer.invalid_markers))
    if answer.search_query and answer.search_query.strip() != question.strip():
        notes.append("searched for: " + answer.search_query)
    if answer.is_refusal and not answer.sources:
        notes.append("nothing matched - try different terms, raise k, "
                     "or switch retrieval mode")
    note = "   ·   ".join(notes)
    if note:
        st.caption(note)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer.text, "sources": payload, "note": note}
    )
