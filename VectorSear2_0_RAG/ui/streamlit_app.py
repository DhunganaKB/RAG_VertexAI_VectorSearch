"""
Streamlit Chatbot — RAG with Vector Search 2.0
================================================
Connects to the FastAPI backend at API_BASE_URL.

Start:
    streamlit run ui/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from config import API_BASE_URL as _CFG_URL, GEMINI_MODEL, TOP_K as _CFG_TOP_K
except Exception:
    _CFG_URL    = "http://localhost:8000"
    GEMINI_MODEL = "gemini-2.0-flash-001"
    _CFG_TOP_K  = 5

API_BASE_URL  = os.environ.get("RAG_API_URL", _CFG_URL).rstrip("/")
DEFAULT_TOP_K = _CFG_TOP_K

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG — Vector Search 2.0",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.answer-card {
    background:#f0f4ff; border-left:4px solid #4285F4;
    border-radius:6px; padding:1rem 1.25rem;
    margin-bottom:1rem; font-size:1.02rem; line-height:1.65;
}
.source-rank {
    display:inline-block; background:#4285F4; color:white;
    border-radius:12px; padding:1px 10px; font-size:0.78rem;
    font-weight:600; margin-right:6px;
}
.score-badge {
    display:inline-block; background:#34a853; color:white;
    border-radius:12px; padding:1px 8px; font-size:0.75rem; margin-left:4px;
}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    st.divider()
    api_url = st.text_input("FastAPI backend URL", value=API_BASE_URL)
    top_k   = st.slider("Top-K sources", 1, 20, DEFAULT_TOP_K)
    st.divider()
    st.subheader("Backend status")
    if st.button("🔄 Check connection"):
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            st.success("✅ Backend reachable") if r.ok else st.error(f"❌ HTTP {r.status_code}")
        except Exception as exc:
            st.error(f"❌ {exc}")
    st.divider()
    st.caption(
        f"Model: **{GEMINI_MODEL}**\n\n"
        "**VS2.0 Pipeline:**\n"
        "Query → Embed → `SearchDataObjects` → Gemini\n\n"
        "_No Firestore needed — metadata lives in the Collection._"
    )

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 RAG Chatbot — Vector Search 2.0")
st.markdown(
    "_Ask questions about your documents. Answers are grounded in chunks "
    "retrieved directly from the VS2.0 Collection._"
)
st.divider()

# ── Query form ────────────────────────────────────────────────────────────────
with st.form("query_form", clear_on_submit=False):
    query     = st.text_area("Your question", placeholder="What are the main topics covered?", height=100)
    submitted = st.form_submit_button("🚀 Ask", use_container_width=True)

# ── Call backend ──────────────────────────────────────────────────────────────
if submitted:
    question = query.strip()
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching collection and generating answer…"):
            t0 = time.perf_counter()
            try:
                resp = requests.post(
                    f"{api_url}/rag",
                    json={"query": question, "top_k": top_k},
                    timeout=120,
                )
                resp.raise_for_status()
                result  = resp.json()
                elapsed = time.perf_counter() - t0
                st.session_state.history.insert(0, {
                    "question": question,
                    "answer":   result.get("answer", ""),
                    "sources":  result.get("sources", []),
                    "model":    result.get("model", GEMINI_MODEL),
                    "elapsed":  elapsed,
                })
            except requests.ConnectionError:
                st.error("❌ Cannot reach backend. Is `uvicorn app.main:app --port 8000` running?")
            except requests.HTTPError as exc:
                detail = exc.response.json().get("detail", str(exc)) if exc.response else str(exc)
                st.error(f"❌ Backend error: {detail}")
            except Exception as exc:
                st.error(f"❌ {exc}")

# ── Render history ────────────────────────────────────────────────────────────
def render_sources(sources: list[dict]) -> None:
    for src in sources:
        rank    = src.get("rank", "?")
        title   = src.get("title") or src.get("id", "Unknown")
        source  = src.get("source", "")
        text    = src.get("text", "")
        score   = src.get("score")
        score_html = f'<span class="score-badge">score {score:.4f}</span>' if score is not None else ""
        with st.expander(f"[{rank}] {title}", expanded=False):
            st.markdown(
                f'<span class="source-rank">[{rank}]</span><strong>{title}</strong>{score_html}',
                unsafe_allow_html=True,
            )
            if source:
                st.markdown(f"**Source:** `{source}`")
            if text:
                st.markdown("**Excerpt:**")
                st.markdown(
                    f"<div style='background:#f8f8f8;border-radius:4px;padding:0.6rem;font-size:0.9rem;'>{text}</div>",
                    unsafe_allow_html=True,
                )


for entry in st.session_state.history:
    st.markdown(f"#### 💬 {entry['question']}")
    st.markdown(f'<div class="answer-card">{entry["answer"]}</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.caption(f"🤖 `{entry.get('model', GEMINI_MODEL)}`")
    c2.caption(f"📚 {len(entry['sources'])} sources")
    c3.caption(f"⏱ {entry.get('elapsed', 0):.2f}s")
    render_sources(entry["sources"])
    st.divider()

if not st.session_state.history and not submitted:
    st.markdown("""
    <div style="text-align:center;padding:3rem 0;color:#888;">
        <h3>👆 Type a question and press <em>Ask</em></h3>
        <p>Powered by Vertex AI Vector Search 2.0 — metadata stored directly in the Collection.</p>
    </div>
    """, unsafe_allow_html=True)
