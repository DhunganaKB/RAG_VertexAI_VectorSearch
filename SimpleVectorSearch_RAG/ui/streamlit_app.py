"""
RAG Streamlit UI
================
Frontend for the Vertex AI Vector Search + Firestore + Gemini RAG pipeline.

Usage:
    streamlit run ui/streamlit_app.py --server.port 8501

The app connects to the FastAPI backend running at API_BASE_URL (configured in
config.py or overridden via the RAG_API_URL environment variable).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config — read from root config.py (same source of truth as the backend)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    from config import API_BASE_URL as _CFG_API_URL, GEMINI_MODEL, TOP_K as _CFG_TOP_K
except Exception:
    _CFG_API_URL = "http://localhost:8000"
    GEMINI_MODEL = "gemini-2.5-flash"
    _CFG_TOP_K = 5

# Environment variable overrides config.py (handy for Docker / Cloud Run)
API_BASE_URL: str = os.environ.get("RAG_API_URL", _CFG_API_URL).rstrip("/")
DEFAULT_TOP_K: int = _CFG_TOP_K


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG — Vertex AI + Gemini",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS — subtle dark-ish card styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Answer card */
    .answer-card {
        background: #f0f4ff;
        border-left: 4px solid #4285F4;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        font-size: 1.02rem;
        line-height: 1.65;
    }
    /* Source chip */
    .source-rank {
        display: inline-block;
        background: #4285F4;
        color: white;
        border-radius: 12px;
        padding: 1px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .score-badge {
        display: inline-block;
        background: #34a853;
        color: white;
        border-radius: 12px;
        padding: 1px 8px;
        font-size: 0.75rem;
        margin-left: 4px;
    }
    /* Hide Streamlit footer */
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.divider()

    api_url = st.text_input(
        "FastAPI backend URL",
        value=API_BASE_URL,
        help="URL where `uvicorn app.main:app` is running",
    )

    top_k = st.slider(
        "Top-K sources to retrieve",
        min_value=1,
        max_value=20,
        value=DEFAULT_TOP_K,
        help="Number of document chunks retrieved from Vector Search per query.",
    )

    st.divider()

    # Backend health check
    st.subheader("Backend status")
    if st.button("🔄 Check connection"):
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            if r.status_code == 200:
                st.success("✅ Backend is reachable")
            else:
                st.error(f"❌ HTTP {r.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot reach backend — is uvicorn running?")
        except Exception as exc:
            st.error(f"❌ {exc}")

    st.divider()
    st.caption(
        f"Model: **{GEMINI_MODEL}**\n\n"
        "Architecture: Query → Embed → Vertex AI Vector Search → "
        "Firestore `get_all` → Gemini"
    )
    st.caption("📄 [API docs](http://localhost:8000/docs)")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("🔍 RAG — Vertex AI + Gemini")
st.markdown(
    "_Ask any question about your uploaded documents. "
    "The answer is grounded in retrieved source chunks._"
)
st.divider()

# Query input
with st.form("query_form", clear_on_submit=False):
    query = st.text_area(
        "Your question",
        placeholder="e.g. What are the main topics covered in the documents?",
        height=100,
    )
    submitted = st.form_submit_button("🚀 Ask", use_container_width=True)


# ---------------------------------------------------------------------------
# Call the RAG backend
# ---------------------------------------------------------------------------
def call_rag_api(question: str, k: int) -> dict:
    """POST /rag and return the parsed JSON response."""
    url = f"{api_url}/rag"
    payload = {"query": question, "top_k": k}
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


if submitted:
    question = query.strip()
    if not question:
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner("Retrieving sources and generating answer…"):
            t0 = time.perf_counter()
            try:
                result = call_rag_api(question, top_k)
                elapsed = time.perf_counter() - t0
                # Prepend to history (newest first)
                st.session_state.history.insert(
                    0,
                    {
                        "question": question,
                        "answer": result.get("answer", ""),
                        "sources": result.get("sources", []),
                        "model": result.get("model", GEMINI_MODEL),
                        "elapsed": elapsed,
                    },
                )
            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Could not connect to the FastAPI backend. "
                    "Make sure `uvicorn app.main:app --reload --port 8000` is running."
                )
            except requests.exceptions.HTTPError as exc:
                try:
                    detail = exc.response.json().get("detail", str(exc))
                except Exception:
                    detail = str(exc)
                st.error(f"❌ Backend error: {detail}")
            except Exception as exc:
                st.error(f"❌ Unexpected error: {exc}")


# ---------------------------------------------------------------------------
# Render history
# ---------------------------------------------------------------------------
def render_sources(sources: list[dict]) -> None:
    """Render the source chunks in an expandable section."""
    if not sources:
        st.info("No sources returned.")
        return

    for src in sources:
        rank = src.get("rank", "?")
        title = src.get("title") or src.get("id", "Unknown")
        uri = src.get("uri") or ""
        snippet = src.get("snippet") or ""
        score = src.get("score")

        score_html = (
            f'<span class="score-badge">score {score:.4f}</span>' if score is not None else ""
        )
        header_html = (
            f'<span class="source-rank">[{rank}]</span>'
            f"<strong>{title}</strong>{score_html}"
        )

        with st.expander(f"[{rank}] {title}", expanded=False):
            st.markdown(header_html, unsafe_allow_html=True)
            if uri:
                st.markdown(f"**Source:** `{uri}`")
            if snippet:
                st.markdown("**Excerpt:**")
                st.markdown(
                    f"<div style='background:#f8f8f8;border-radius:4px;"
                    f"padding:0.6rem 0.9rem;font-size:0.9rem;'>{snippet}</div>",
                    unsafe_allow_html=True,
                )


for entry in st.session_state.history:
    q = entry["question"]
    a = entry["answer"]
    sources = entry["sources"]
    model = entry.get("model", GEMINI_MODEL)
    elapsed = entry.get("elapsed", 0)

    # Question bubble
    st.markdown(f"#### 💬 {q}")

    # Answer card
    st.markdown(
        f'<div class="answer-card">{a}</div>',
        unsafe_allow_html=True,
    )

    # Metadata row
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        st.caption(f"🤖 Model: `{model}`")
    with col2:
        st.caption(f"📚 Sources retrieved: {len(sources)}")
    with col3:
        st.caption(f"⏱ {elapsed:.2f}s")

    # Sources
    render_sources(sources)

    st.divider()


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------
if not st.session_state.history and not submitted:
    st.markdown(
        """
        <div style="text-align:center; padding: 3rem 0; color: #888;">
            <h3>👆 Type a question above and press <em>Ask</em></h3>
            <p>Answers are grounded in your uploaded documents via Vertex AI Vector Search.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
