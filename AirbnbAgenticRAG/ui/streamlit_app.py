"""
Streamlit Chatbot — Airbnb Agentic RAG
=======================================
Airbnb-themed chat interface deployed on Cloud Run.
Connects to the FastAPI backend via the RAG_API_URL environment variable.

Cloud Run env vars required:
    RAG_API_URL   — public URL of the airbnb-rag-api Cloud Run service
                    e.g. https://airbnb-rag-api-abc123-uc.a.run.app

Optional env vars (have sensible defaults):
    GEMINI_MODEL  — model name shown in UI  (default: gemini-2.5-flash)
    TOP_K         — default retrieval count (default: 10)
"""
from __future__ import annotations

import os
import time

import requests
import streamlit as st

# ── All config from environment variables (no local config.py dependency) ─────
API_BASE_URL  = os.environ.get("RAG_API_URL", "").rstrip("/")
GEMINI_MODEL  = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_TOP_K = int(os.environ.get("TOP_K", "10"))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airbnb Rental Agent — Austin TX",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Airbnb-themed CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Airbnb brand red as accent */
:root { --airbnb-red: #FF385C; --airbnb-dark: #222222; }

.hero-banner {
    background: linear-gradient(135deg, #FF385C 0%, #bd1e59 100%);
    border-radius: 12px; padding: 1.5rem 2rem; margin-bottom: 1.5rem;
    color: white;
}
.hero-banner h1 { font-size: 1.8rem; margin: 0; }
.hero-banner p  { margin: 0.3rem 0 0; opacity: 0.9; font-size: 1rem; }

.agent-answer {
    background: #fff7f8; border-left: 4px solid #FF385C;
    border-radius: 8px; padding: 1rem 1.25rem;
    margin-bottom: 0.75rem; font-size: 1.02rem; line-height: 1.7;
}
.user-bubble {
    background: #f0f0f0; border-radius: 8px;
    padding: 0.6rem 1rem; margin-bottom: 0.5rem;
    font-size: 1rem; color: #222;
}

.listing-card {
    background: white; border: 1px solid #e8e8e8;
    border-radius: 10px; padding: 0.9rem 1rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}
.price-tag {
    display: inline-block; background: #FF385C; color: white;
    border-radius: 14px; padding: 2px 10px;
    font-size: 0.85rem; font-weight: 700; margin-right: 6px;
}
.rating-badge {
    display: inline-block; background: #222; color: white;
    border-radius: 14px; padding: 2px 8px;
    font-size: 0.8rem; margin-left: 4px;
}
.instant-badge {
    display: inline-block; background: #00a699; color: white;
    border-radius: 14px; padding: 2px 8px; font-size: 0.78rem;
}
.tool-pill {
    display: inline-block; background: #f5f5f5; border: 1px solid #ddd;
    border-radius: 10px; padding: 2px 8px; font-size: 0.75rem; color: #555;
    margin-right: 4px;
}
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.divider()

    if not API_BASE_URL:
        st.error(
            "**RAG_API_URL environment variable is not set.**\n\n"
            "Set it on the Cloud Run service:\n"
            "`gcloud run services update airbnb-rag-ui --update-env-vars RAG_API_URL=https://...`"
        )

    api_url = st.text_input("FastAPI backend URL", value=API_BASE_URL)

    mode = st.radio(
        "Search mode",
        ["🤖 Agentic (recommended)", "🔍 Direct RAG"],
        help=(
            "**Agentic**: Gemini extracts filters and calls find_rentals() "
            "with the right parameters.\n\n"
            "**Direct RAG**: Embed query → VS2.0 → Gemini (no filter reasoning)."
        ),
    )
    use_agent = mode.startswith("🤖")

    if not use_agent:
        top_k = st.slider("Top-K listings", 1, 30, DEFAULT_TOP_K)

    st.divider()
    st.subheader("Backend status")
    if st.button("🔄 Check connection"):
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            if r.ok:
                st.success("✅ Backend reachable")
            else:
                st.error(f"❌ HTTP {r.status_code}")
        except Exception as exc:
            st.error(f"❌ {exc}")

    st.divider()
    st.caption(
        f"Model: **{GEMINI_MODEL}**\n\n"
        "**Pipeline:**\n"
        "Query → Gemini Agent → `find_rentals()` → VS2.0 → Answer\n\n"
        "_All listing metadata lives inside the VS2.0 Collection._"
    )
    st.caption("📍 Austin, TX • 10k+ listings")

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <h1>🏠 Airbnb Rental Agent — Austin, TX</h1>
  <p>AI-powered search using Vertex AI Vector Search 2.0 &amp; Gemini function calling</p>
</div>
""", unsafe_allow_html=True)

# ── Example prompts ───────────────────────────────────────────────────────────
with st.expander("💡 Example queries", expanded=False):
    examples = [
        "Find me a cozy studio near downtown under $80 per night",
        "I need an entire home for 6 people with at least 3 bedrooms",
        "Private room that's instantly bookable, max $60/night",
        "Family-friendly house in East Austin, good reviews, under $200",
        "Artist loft or creative space near live music venues",
        "Budget-friendly private room under $50, superhost preferred",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["prefill_query"] = ex

# ── Query form ────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill_query", "")
with st.form("query_form", clear_on_submit=False):
    query = st.text_area(
        "What are you looking for?",
        value=prefill,
        placeholder="e.g. cozy private room near downtown under $80 per night",
        height=90,
    )
    submitted = st.form_submit_button("🔍 Search", use_container_width=True)

# ── Call backend ──────────────────────────────────────────────────────────────
if submitted:
    question = query.strip()
    if not question:
        st.warning("Please describe what you're looking for.")
    else:
        with st.spinner("Searching Austin Airbnb listings…"):
            t0 = time.perf_counter()
            try:
                if use_agent:
                    resp = requests.post(
                        f"{api_url}/ask",
                        json={"query": question},
                        timeout=120,
                    )
                else:
                    resp = requests.post(
                        f"{api_url}/rag",
                        json={"query": question, "top_k": top_k},
                        timeout=120,
                    )
                resp.raise_for_status()
                result  = resp.json()
                elapsed = time.perf_counter() - t0

                st.session_state.history.insert(0, {
                    "question":   question,
                    "answer":     result.get("answer", ""),
                    "sources":    result.get("sources", []),
                    "tool_calls": result.get("tool_calls", []),
                    "model":      result.get("model", GEMINI_MODEL),
                    "elapsed":    elapsed,
                    "mode":       "agent" if use_agent else "rag",
                })
            except requests.ConnectionError:
                st.error(
                    f"❌ Cannot reach backend at `{api_url}`. "
                    "Check that the RAG_API_URL environment variable is set correctly "
                    "on this Cloud Run service."
                )
            except requests.HTTPError as exc:
                detail = (
                    exc.response.json().get("detail", str(exc))
                    if exc.response else str(exc)
                )
                st.error(f"❌ Backend error: {detail}")
            except Exception as exc:
                st.error(f"❌ {exc}")

# ── Render a single listing source card ───────────────────────────────────────
def render_listing_card(src: dict, rank: int) -> None:
    name    = src.get("name") or src.get("id", "Unknown")
    nbhd    = src.get("neighbourhood", "")
    r_type  = src.get("room_type", "")
    price   = src.get("price")
    rating  = src.get("rating")
    accom   = src.get("accommodates")
    beds    = src.get("bedrooms")
    ib      = src.get("instant_bookable", "false")
    url     = src.get("listing_url", "")
    host    = src.get("host_name", "")
    text    = src.get("text", "")[:250]
    score   = src.get("score")

    price_html  = f'<span class="price-tag">${price:.0f}/night</span>' if price else ""
    rating_html = f'<span class="rating-badge">⭐ {rating:.2f}</span>' if rating else ""
    ib_html     = '<span class="instant-badge">⚡ Instant</span>' if ib == "true" else ""
    score_html  = f"<small style='color:#999'>score: {score:.4f}</small>" if score else ""

    details = []
    if nbhd:
        details.append(f"📍 {nbhd}")
    if r_type:
        details.append(f"🏠 {r_type}")
    if accom:
        details.append(f"👥 {int(accom)} guests")
    if beds:
        details.append(f"🛏 {int(beds)} bed")
    if host:
        details.append(f"Host: {host}")
    detail_str = "  |  ".join(details)

    url_html = f'<a href="{url}" target="_blank" style="color:#FF385C;">View listing ↗</a>' if url else ""

    st.markdown(f"""
<div class="listing-card">
  <strong>[{rank}] {name}</strong><br/>
  {price_html}{rating_html}{ib_html} {score_html}<br/>
  <small>{detail_str}</small><br/>
  {url_html}
</div>
""", unsafe_allow_html=True)
    if text:
        with st.expander("📖 Description snippet", expanded=False):
            st.markdown(
                f"<div style='font-size:0.9rem;color:#444;'>{text}</div>",
                unsafe_allow_html=True,
            )


# ── Render conversation history ───────────────────────────────────────────────
for entry in st.session_state.history:
    st.markdown(
        f'<div class="user-bubble">💬 {entry["question"]}</div>',
        unsafe_allow_html=True,
    )

    # Tool calls log (agentic mode)
    if entry.get("tool_calls"):
        pills_html = "".join(
            f'<span class="tool-pill">🔧 {c["tool"]}()</span>'
            for c in entry["tool_calls"]
        )
        st.markdown(
            f"<div style='margin-bottom:0.4rem'>{pills_html}</div>",
            unsafe_allow_html=True,
        )

    # Agent / LLM answer
    st.markdown(
        f'<div class="agent-answer">{entry["answer"]}</div>',
        unsafe_allow_html=True,
    )

    # Metadata row
    c1, c2, c3, c4 = st.columns(4)
    c1.caption(f"🤖 `{entry.get('model', GEMINI_MODEL)}`")
    mode_label = "Agentic" if entry.get("mode") == "agent" else "Direct RAG"
    c2.caption(f"⚡ {mode_label}")
    c3.caption(f"📚 {len(entry.get('sources', []))} sources")
    c4.caption(f"⏱ {entry.get('elapsed', 0):.2f}s")

    # Sources / listings
    sources = entry.get("sources", [])
    if sources:
        with st.expander(f"📋 Retrieved listings ({len(sources)})", expanded=False):
            for src in sources:
                render_listing_card(src, src.get("rank", "?"))

    st.divider()

# ── Empty state ───────────────────────────────────────────────────────────────
if not st.session_state.history and not submitted:
    st.markdown("""
<div style="text-align:center; padding:3rem 0; color:#888;">
  <div style="font-size:3rem;">🏠</div>
  <h3>Find your perfect stay in Austin, TX</h3>
  <p>Describe what you're looking for in plain English.<br/>
  The AI agent will find the best matching listings from our VS2.0 collection.</p>
  <br/>
  <p style="font-size:0.9rem;">
    Powered by <strong>Vertex AI Vector Search 2.0</strong> + <strong>Gemini 2.5 Flash</strong>
  </p>
</div>
""", unsafe_allow_html=True)
