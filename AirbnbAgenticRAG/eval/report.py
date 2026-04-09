"""
eval/report.py — HTML Report Generator
=======================================
Produces a self-contained single-file HTML evaluation report as a string.
The string is uploaded to GCS by evaluate.py — no local file is written.

No external dependencies — everything is inline CSS + vanilla JS.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone


# ── Score coloring helpers ────────────────────────────────────────────────────

def _score_color(score: float | None, max_val: float = 4.0) -> str:
    """Return a hex colour from red → amber → green based on score ratio."""
    if score is None:
        return "#94A3B8"   # grey
    ratio = min(max(score / max_val, 0.0), 1.0)
    if ratio >= 0.75:
        return "#16A34A"   # green
    if ratio >= 0.50:
        return "#D97706"   # amber
    return "#DC2626"       # red


def _badge(val: float | None, max_val: float, label: str) -> str:
    col = _score_color(val, max_val)
    v   = f"{val:.2f}" if val is not None else "—"
    return (
        f'<span style="background:{col};color:white;border-radius:4px;'
        f'padding:2px 7px;font-size:12px;font-weight:600;">{label}: {v}</span>'
    )


def _bool_badge(val: bool | None) -> str:
    if val is None:
        return '<span style="color:#94A3B8">—</span>'
    col  = "#16A34A" if val else "#DC2626"
    text = "✓" if val else "✗"
    return f'<span style="color:{col};font-weight:700;font-size:16px">{text}</span>'


# ═══════════════════════════════════════════════════════════════════════════════
# HTML builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_html_report(results: list[dict], summary: dict) -> str:
    """Build the full HTML report and return it as a string (for GCS upload)."""
    return _render_html(results, summary)


def _render_html(results: list[dict], summary: dict) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── Summary cards ────────────────────────────────────────────────────────
    def card(title, value, sub="", col="#1D4ED8"):
        return f"""
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;
                    padding:16px 20px;text-align:center;min-width:140px">
          <div style="font-size:11px;color:#64748B;text-transform:uppercase;
                      letter-spacing:.5px;margin-bottom:4px">{title}</div>
          <div style="font-size:28px;font-weight:700;color:{col}">{value}</div>
          <div style="font-size:11px;color:#94A3B8;margin-top:2px">{sub}</div>
        </div>"""

    def metric(label, val, unit=""):
        v = f"{val}" if val is not None else "—"
        return f'<div style="margin:4px 0"><span style="color:#64748B">{label}:</span> <b>{v}{unit}</b></div>'

    rag_cards = ""
    ask_cards = ""
    if "rag_count" in summary:
        rag_cards = f"""
        <h3 style="margin:16px 0 8px;color:#0F766E">Simple RAG  <code>/rag</code></h3>
        <div style="display:flex;gap:12px;flex-wrap:wrap">
          {card("Queries", summary.get("rag_count","—"), col="#0F766E")}
          {card("Avg Relevance", f"{summary.get('rag_avg_relevance','—')}/4", "LLM judge", col="#0F766E")}
          {card("Keyword Hit", f"{(summary.get('rag_avg_keyword_hit_rate') or 0)*100:.0f}%", "", col="#0F766E")}
          {card("Has Results", f"{(summary.get('rag_has_results_pct') or 0)*100:.0f}%", "", col="#0F766E")}
          {card("Avg Latency", f"{summary.get('rag_avg_latency_ms','—'):.0f}", "ms", col="#0F766E")}
        </div>"""
    if "ask_count" in summary:
        ask_cards = f"""
        <h3 style="margin:16px 0 8px;color:#7C3AED">Agentic RAG  <code>/ask</code></h3>
        <div style="display:flex;gap:12px;flex-wrap:wrap">
          {card("Queries", summary.get("ask_count","—"), col="#7C3AED")}
          {card("Avg Relevance", f"{summary.get('ask_avg_relevance','—')}/4", "LLM judge", col="#7C3AED")}
          {card("Filter Accuracy", f"{(summary.get('ask_avg_filter_accuracy') or 0)*100:.0f}%", "", col="#7C3AED")}
          {card("Keyword Hit", f"{(summary.get('ask_avg_keyword_hit_rate') or 0)*100:.0f}%", "", col="#7C3AED")}
          {card("Avg Latency", f"{summary.get('ask_avg_latency_ms','—'):.0f}", "ms", col="#7C3AED")}
        </div>"""

    # ── Category breakdown ────────────────────────────────────────────────────
    cat_rows = ""
    for cat, cstats in summary.get("by_category", {}).items():
        rel = f"{cstats.get('avg_relevance','—'):.2f}" if cstats.get('avg_relevance') is not None else "—"
        lat = f"{cstats.get('avg_latency_ms','—'):.0f}" if cstats.get('avg_latency_ms') is not None else "—"
        pct = f"{cstats.get('has_results_pct',0)*100:.0f}%" if cstats.get('has_results_pct') is not None else "—"
        cat_rows += f"""
        <tr>
          <td><b>{cat}</b></td>
          <td style="text-align:center">{cstats.get('count','—')}</td>
          <td style="text-align:center">{rel}/4</td>
          <td style="text-align:center">{lat} ms</td>
          <td style="text-align:center">{pct}</td>
        </tr>"""

    # ── Per-query rows ────────────────────────────────────────────────────────
    ENDPOINT_COLORS = {"rag": "#0F766E", "ask": "#7C3AED"}
    result_rows = ""
    for r in results:
        ep    = r.get("endpoint", "")
        ecol  = ENDPOINT_COLORS.get(ep, "#334155")
        err   = r.get("error")
        rel_b = _badge(r.get("answer_relevance"), 4.0, "relevance")
        khr_b = _badge(r.get("keyword_hit_rate"),  1.0, "khr")
        fa_b  = _badge(r.get("filter_accuracy"),   1.0, "filter") if ep == "ask" else ""

        answer_escaped = (r.get("answer","") or "").replace("<","&lt;").replace(">","&gt;")
        err_html = f'<div style="color:#DC2626;font-size:11px">Error: {err}</div>' if err else ""

        result_rows += f"""
        <tr>
          <td style="font-size:12px;color:#64748B">{r.get('id','')}</td>
          <td><span style="background:{ecol};color:white;border-radius:3px;
              padding:1px 6px;font-size:11px">/{ep}</span></td>
          <td style="font-size:12px;color:#475569">{r.get('category','')}</td>
          <td style="max-width:280px;font-size:13px">{r.get('query','')}</td>
          <td style="text-align:center">{_bool_badge(r.get('has_results'))}</td>
          <td style="text-align:center">{r.get('retrieved_count','—')}</td>
          <td style="text-align:center">{r.get('latency_ms','—')} ms</td>
          <td>
            {rel_b} {khr_b} {fa_b}
            {err_html}
            <details style="margin-top:6px">
              <summary style="font-size:11px;color:#64748B;cursor:pointer">answer ▾</summary>
              <pre style="font-size:11px;white-space:pre-wrap;max-width:420px;
                          background:#F1F5F9;border-radius:4px;padding:6px;
                          margin:4px 0 0">{answer_escaped[:400]}</pre>
            </details>
          </td>
        </tr>"""

    # ── Full HTML ─────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAG Evaluation Report — {ts}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #F8FAFC; color: #1E293B; padding: 24px; }}
  h1   {{ font-size: 22px; font-weight: 700; margin-bottom: 4px; }}
  h2   {{ font-size: 16px; font-weight: 600; margin: 24px 0 12px; color: #334155; }}
  code {{ background: #E2E8F0; border-radius: 3px; padding: 1px 4px; font-size: 13px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
  th   {{ background: #1E293B; color: white; padding: 10px 12px; font-size: 12px;
           text-align: left; font-weight: 600; position: sticky; top: 0; }}
  td   {{ padding: 10px 12px; border-bottom: 1px solid #E2E8F0; vertical-align: top; }}
  tr:hover td {{ background: #F1F5F9; }}
  .section {{ background: white; border: 1px solid #E2E8F0; border-radius: 10px;
              padding: 20px 24px; margin-bottom: 20px; }}
  details summary {{ list-style: none; }}
  details summary::-webkit-details-marker {{ display: none; }}
</style>
</head>
<body>

<div class="section">
  <h1>🏡 Airbnb Agentic RAG — Evaluation Report</h1>
  <div style="color:#64748B;font-size:13px;margin-top:4px">
    Generated: {ts} &nbsp;|&nbsp; Queries: {summary.get('total_queries','—')} &nbsp;|&nbsp;
    Model: {getattr(__import__('config'), 'GEMINI_MODEL', 'gemini-2.5-flash')}
  </div>
</div>

<div class="section">
  <h2>Summary Metrics</h2>
  {rag_cards}
  {ask_cards}
</div>

<div class="section">
  <h2>Metrics by Category</h2>
  <table>
    <tr>
      <th>Category</th><th>Count</th><th>Avg Relevance</th>
      <th>Avg Latency</th><th>Has Results %</th>
    </tr>
    {cat_rows}
  </table>
</div>

<div class="section">
  <h2>Per-Query Results</h2>
  <table>
    <tr>
      <th>ID</th><th>Endpoint</th><th>Category</th><th>Query</th>
      <th>Results?</th><th>Count</th><th>Latency</th><th>Scores + Answer</th>
    </tr>
    {result_rows}
  </table>
</div>

<div style="text-align:center;color:#94A3B8;font-size:12px;margin-top:20px">
  Airbnb Agentic RAG · Vertex AI Vector Search 2.0 · Gemini 2.5 Flash
</div>

</body>
</html>"""

    return html
