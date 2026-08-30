"""Presentation layer: the stylesheet and small render helpers app.py uses.

Kept out of app.py so that file stays about behaviour, and so the design
can change without touching the logic.

Design brief - a reading instrument. The answer is the only thing set in a
serif, at a comfortable measure and generous line height, because it is
prose meant to be read. Everything else (chrome, metadata, provenance) is
small, quiet, and set in sans or mono. Sources are footnotes rather than
cards: a hairline rule, a marker, the evidence.

The stylesheet is plain concatenation, not an f-string, because CSS is
mostly braces and doubling every one of them invites exactly the kind of
typo that stays invisible until it renders.
"""
from __future__ import annotations

import html as _html
import re as _re

APP_NAME = "PaperChat"
APP_TAGLINE = "Ask your papers. Every claim cites a page you can open."

_FONTS = (
    "https://fonts.googleapis.com/css2"
    "?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600"
    "&family=IBM+Plex+Sans:wght@400;500;600"
    "&family=IBM+Plex+Mono:wght@400;500"
    "&display=swap"
)

# The stylesheet is authored readably below and EMITTED AS ONE LINE.
#
# That is not cosmetic. Streamlit renders st.markdown through a CommonMark
# parser, and a raw-HTML block there terminates at the first blank line -
# so a stylesheet with blank lines between sections gets cut short and the
# remainder is rendered as prose (with `#MainMenu` becoming an <h1>).
# Minifying guarantees a single uninterrupted HTML block, which is also
# what makes the st.markdown fallback in inject() safe.

_CSS_SOURCE = """
:root {
  --ink:       #17191d;
  --ink-2:     #5a626d;
  --ink-3:     #8b939e;
  --paper:     #ffffff;
  --ground:    #f7f7f6;
  --rule:      #e2e2df;
  --rule-soft: #eeeeec;
  --accent:    #3a5570;
  --accent-2:  #2b4054;
  --wash:      #eef2f6;
  --flag:      #8c3b2e;
  --serif:  "Source Serif 4", Georgia, "Times New Roman", serif;
  --sans:   "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono:   "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
}

/* ---- quieten Streamlit chrome ----
   Do NOT hide header[data-testid="stHeader"]. The control that reopens a
   collapsed sidebar lives inside it, so hiding the header strands the user
   with no way to get the sidebar back. Hide the pieces, keep the frame. */
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stToolbarActions"], [data-testid="stStatusWidget"], #MainMenu { visibility: hidden; }
footer { visibility: hidden; height: 0; }

/* Belt and braces: whatever this Streamlit version calls the sidebar
   expand/collapse control, it stays visible and clickable. */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"],
[data-testid="collapsedControl"] {
  visibility: visible !important;
  opacity: 1 !important;
  pointer-events: auto !important;
  z-index: 999 !important;
}

.block-container { padding-top: 1.4rem !important; padding-bottom: 5rem !important; max-width: 1080px; }

html, body, [class*="css"] { font-family: var(--sans); color: var(--ink); }
.stApp { background: var(--ground); }

/* ---- masthead ---- */
.pc-mast { border-bottom: 1px solid var(--rule); padding-bottom: 16px; margin-bottom: 26px; }
.pc-name {
  font-family: var(--serif); font-weight: 600; font-size: 30px;
  letter-spacing: -.02em; color: var(--ink); margin: 0; line-height: 1.1;
}
.pc-tag {
  font-family: var(--mono); font-size: 11px; letter-spacing: .06em;
  color: var(--ink-3); margin-top: 7px; text-transform: uppercase;
}

/* ---- the answer: the one thing set for reading ---- */
.pc-answer {
  font-family: var(--serif); font-size: 18px; line-height: 1.66;
  color: var(--ink); max-width: 68ch; letter-spacing: .001em;
}
.pc-answer p { margin: 0 0 .85em; }
.pc-answer p:last-child { margin-bottom: 0; }
.pc-answer ul, .pc-answer ol { max-width: 66ch; padding-left: 1.3em; margin: .6em 0; }
.pc-answer li { margin-bottom: .35em; }
.pc-answer strong { font-weight: 600; }
.pc-answer code {
  font-family: var(--mono); font-size: .84em; background: var(--wash);
  padding: 1px 5px; border-radius: 2px;
}
/* citation markers ride high and quiet, like footnote numerals */
.pc-cite {
  font-family: var(--sans); font-size: .62em; font-weight: 600;
  vertical-align: super; line-height: 0; color: var(--accent);
  padding: 0 .5px; letter-spacing: .02em;
}

/* ---- the question, above its answer ---- */
.pc-q {
  font-family: var(--sans); font-weight: 500; font-size: 15px;
  color: var(--ink-2); padding-left: 13px; border-left: 2px solid var(--rule);
  margin-bottom: 18px; max-width: 68ch;
}

/* ---- sources set as footnotes ---- */
.pc-sources-head {
  font-family: var(--mono); font-size: 10px; letter-spacing: .13em;
  text-transform: uppercase; color: var(--ink-3);
  margin: 30px 0 0; padding-top: 13px; border-top: 1px solid var(--rule);
  display: flex; justify-content: space-between; align-items: baseline; gap: 12px;
}
.pc-src { padding: 13px 0; border-bottom: 1px solid var(--rule-soft); }
.pc-src:last-child { border-bottom: 0; }
.pc-src-line { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
.pc-marker {
  font-family: var(--sans); font-size: 11px; font-weight: 600; color: var(--accent);
  min-width: 17px;
}
.pc-src-doc { font-family: var(--sans); font-size: 13.5px; font-weight: 500; color: var(--ink); }
.pc-src-meta { font-family: var(--mono); font-size: 10.5px; color: var(--ink-3); letter-spacing: .02em; }
.pc-quote {
  font-family: var(--serif); font-size: 14.5px; line-height: 1.6; color: var(--ink-2);
  margin: 8px 0 0 27px; padding-left: 13px; border-left: 2px solid var(--rule);
  max-width: 74ch;
}

/* ---- provenance chips ---- */
.pc-chip {
  font-family: var(--mono); font-size: 9.5px; letter-spacing: .07em; text-transform: uppercase;
  padding: 1.5px 6px; border: 1px solid var(--rule); color: var(--ink-3);
  background: var(--paper); white-space: nowrap;
}
.pc-chip.both { color: var(--accent-2); border-color: #c3cfda; background: var(--wash); }
.pc-chip.rr   { color: var(--accent-2); border-color: #c3cfda; background: var(--wash); }

/* ---- notices ---- */
.pc-note {
  font-family: var(--mono); font-size: 11px; color: var(--ink-3);
  margin-top: 14px; padding-top: 10px; border-top: 1px dotted var(--rule);
}
.pc-warn {
  font-family: var(--sans); font-size: 13px; color: var(--flag);
  border-left: 2px solid var(--flag); padding: 2px 0 2px 12px; margin: 12px 0;
}
.pc-empty {
  font-family: var(--serif); font-size: 17px; line-height: 1.6;
  color: var(--ink-2); max-width: 56ch; margin: 46px auto; text-align: center;
}
.pc-empty b { color: var(--ink); font-weight: 600; }

/* ---- sidebar ---- */
[data-testid="stSidebar"] { background: var(--paper); border-right: 1px solid var(--rule); }
[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
.pc-side-h {
  font-family: var(--mono); font-size: 10px; letter-spacing: .13em; text-transform: uppercase;
  color: var(--ink-3); margin: 22px 0 9px; padding-bottom: 6px; border-bottom: 1px solid var(--rule-soft);
}
.pc-side-h:first-child { margin-top: 0; }
.pc-facts { font-family: var(--mono); font-size: 11px; color: var(--ink-2); line-height: 1.85; }
.pc-facts b { color: var(--ink); font-weight: 500; }
.pc-doc { font-family: var(--sans); font-size: 12.5px; color: var(--ink-2); padding: 3px 0; }
.pc-doc span { font-family: var(--mono); font-size: 10.5px; color: var(--ink-3); }

/* ---- widgets: quieten Streamlit's defaults ---- */
.stButton > button {
  font-family: var(--sans); font-size: 13px; font-weight: 500; border-radius: 2px;
  border: 1px solid var(--rule); background: var(--paper); color: var(--ink); transition: .12s;
}
.stButton > button:hover { border-color: var(--accent); color: var(--accent); }
.stButton > button[kind="primary"] { background: var(--accent); border-color: var(--accent); color: #fff; }
.stButton > button[kind="primary"]:hover { background: var(--accent-2); border-color: var(--accent-2); color: #fff; }
[data-testid="stChatInput"] textarea { font-family: var(--sans); font-size: 15px; }
[data-testid="stChatInput"] { border-radius: 2px; }
[data-testid="stExpander"] details { border: 1px solid var(--rule-soft); border-radius: 2px; background: var(--paper); }
[data-testid="stExpander"] summary { font-family: var(--mono); font-size: 11px; color: var(--ink-2); }
[data-testid="stFileUploader"] section { border-radius: 2px; border-color: var(--rule); }
.stSlider label, .stRadio label, .stSelectbox label, .stTextInput label,
.stMultiSelect label, .stCheckbox label {
  font-family: var(--mono) !important; font-size: 10px !important;
  letter-spacing: .09em !important; text-transform: uppercase !important; color: var(--ink-3) !important;
}
hr { border-color: var(--rule-soft); }
[data-testid="stImage"] img { border: 1px solid var(--rule); }

@media (max-width: 640px) {
  .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
  .pc-answer { font-size: 17px; }
  .pc-name { font-size: 25px; }
}
"""


def _one_line(css: str) -> str:
    """Strip comments and collapse every run of whitespace to one space.

    Deliberately conservative: it does NOT tighten around `:` `>` `,` and
    friends. Squeezing punctuation would save a few hundred bytes and risk
    mangling a selector in a way no test here would catch, since there is
    no browser in the loop. Removing newlines is the whole requirement.
    """
    css = _re.sub(r"/\*.*?\*/", "", css, flags=_re.S)   # comments
    css = _re.sub(r"\s+", " ", css)                     # newlines included
    return css.strip()


# @import rather than a <link> tag: injected markup lands in <body>, where
# a stylesheet link is not reliably honoured, and it must be the first rule
# inside <style> to be valid.
CSS = (
    "<style>@import url('" + _FONTS + "');"
    + _one_line(_CSS_SOURCE)
    + "</style>"
)


def inject(st) -> None:
    """Install the stylesheet.

    This must go through st.html (Streamlit >= 1.33). st.markdown, even
    with unsafe_allow_html=True, runs output through a sanitiser that
    strips <style> and <link> in current versions while keeping their text
    content - which is why the first cut of this file rendered its own CSS
    into the page as prose. st.html is the API that permits stylesheets.

    The markdown path remains only as a fallback for older Streamlit, and
    is safe there because CSS is emitted as a single line.
    """
    if hasattr(st, "html"):
        st.html(CSS)
    else:  # pragma: no cover - only on Streamlit < 1.33
        st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------
# render helpers
# ---------------------------------------------------------------------
_MARKER = _re.compile(r"\[(\d{1,2})\]")


def esc(text: str) -> str:
    return _html.escape(str(text), quote=False)


def answer_html(text: str) -> str:
    """Answer prose as HTML, with [n] markers set as superscript footnotes."""
    body = esc(text)
    body = _MARKER.sub(lambda m: '<span class="pc-cite">' + m.group(1) + "</span>", body)
    # Light markdown only: the model is told to answer in prose.
    body = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", body, flags=_re.S)
    body = _re.sub(r"`([^`]+)`", r"<code>\1</code>", body)

    blocks: list[str] = []
    bullets: list[str] = []

    def flush() -> None:
        if bullets:
            blocks.append("<ul>" + "".join("<li>" + b + "</li>" for b in bullets) + "</ul>")
            bullets.clear()

    for raw in body.split("\n"):
        line = raw.strip()
        if not line:
            flush()
            continue
        if _re.match(r"^[-*•]\s+", line):
            bullets.append(_re.sub(r"^[-*•]\s+", "", line))
        else:
            flush()
            blocks.append("<p>" + line + "</p>")
    flush()
    return '<div class="pc-answer">' + ("".join(blocks) or "<p></p>") + "</div>"


def chips(matched_by: str, rerank_score: float | None) -> str:
    """Provenance chips: how the passage was found, and how it was judged."""
    label = {"both": "keyword + semantic", "semantic": "semantic", "keyword": "keyword"}
    cls = "pc-chip both" if matched_by == "both" else "pc-chip"
    out = ['<span class="' + cls + '">' + label.get(matched_by, matched_by) + "</span>"]
    if rerank_score is not None:
        out.append('<span class="pc-chip rr">relevance %.0f/3</span>' % rerank_score)
    return " ".join(out)


def source_line(marker: int, source: str, page: int, section: str | None,
                matched_by: str, rerank_score: float | None) -> str:
    where = "page %d" % page
    if section:
        where += " &middot; " + esc(section)
    return (
        '<div class="pc-src-line">'
        '<span class="pc-marker">' + str(marker) + "</span>"
        '<span class="pc-src-doc">' + esc(source) + "</span>"
        '<span class="pc-src-meta">' + where + "</span>"
        + chips(matched_by, rerank_score)
        + "</div>"
    )


def quote(text: str, limit: int = 420) -> str:
    body = text.strip()
    if len(body) > limit:
        body = body[:limit].rsplit(" ", 1)[0] + " …"
    return '<div class="pc-quote">' + esc(body) + "</div>"


def masthead() -> str:
    return (
        '<div class="pc-mast">'
        '<div class="pc-name">' + APP_NAME + "</div>"
        '<div class="pc-tag">' + APP_TAGLINE + "</div>"
        "</div>"
    )
