"""
eval/evaluate.py — Offline Evaluation Runner
=============================================
Runs the evaluation dataset against both /rag and /ask endpoints,
scores each result, and uploads JSON + HTML reports to GCS.

Reports are stored at:
  gs://{BUCKET_NAME}/eval/eval_YYYYMMDD_HHMMSS.json
  gs://{BUCKET_NAME}/eval/eval_YYYYMMDD_HHMMSS.html
  gs://{BUCKET_NAME}/eval/latest.json   (overwritten each run)
  gs://{BUCKET_NAME}/eval/latest.html   (overwritten each run)

Usage:
    # Full evaluation (both endpoints)
    python eval/evaluate.py

    # Evaluate only /rag or /ask
    python eval/evaluate.py --endpoint rag
    python eval/evaluate.py --endpoint ask

    # Run a subset of queries (quick smoke-test)
    python eval/evaluate.py --ids q001,q002,q003

    # Skip LLM-as-judge scoring (faster, no Gemini API cost)
    python eval/evaluate.py --no-judge

    # Point at a different API base URL (e.g. Cloud Run)
    python eval/evaluate.py --api-url https://my-service-xyz.run.app

Prerequisites:
    1. FastAPI backend must be running (locally or on Cloud Run)
    2. eval/dataset.json must exist (checked into repo)
    3. GCP credentials: gcloud auth application-default login

What it measures:
    /rag  → answer_relevance (LLM judge 0-4), keyword_hit_rate,
             has_results, retrieved_count, latency_ms
    /ask  → all /rag metrics  +  filter_accuracy
             (how accurately Gemini extracted structured filters)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from google.cloud import storage as gcs

from eval.metrics import (
    aggregate,
    build_judge_prompt,
    filter_accuracy,
    keyword_hit_rate,
    parse_judge_score,
)
from eval.report import build_html_report

# ── Lazy Gemini model for LLM-as-judge ────────────────────────────────────────
_judge_model = None


def _get_judge():
    global _judge_model
    if _judge_model is None:
        import vertexai
        from vertexai.generative_models import GenerationConfig, GenerativeModel
        vertexai.init(project=config.PROJECT_ID, location=config.LOCATION)
        _judge_model = GenerativeModel(
            model_name=config.GEMINI_MODEL,
            generation_config=GenerationConfig(
                temperature=0.0,    # deterministic scoring
                max_output_tokens=64,
            ),
        )
    return _judge_model


def llm_judge(query: str, answer: str) -> int | None:
    """Call Gemini to score the answer 0-4. Returns None on error."""
    try:
        prompt = build_judge_prompt(query, answer)
        resp   = _get_judge().generate_content(prompt)
        return parse_judge_score(resp.text or "")
    except Exception as exc:
        print(f"    [Judge] Error: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint callers
# ═══════════════════════════════════════════════════════════════════════════════

def call_rag(query: str, api_url: str) -> tuple[dict, float]:
    """Call POST /rag. Returns (response_json, latency_ms)."""
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{api_url}/rag",
            json={"query": query, "top_k": config.EVAL_TOP_K},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        data = {"error": str(exc), "answer": "", "sources": []}
    latency = (time.perf_counter() - t0) * 1000
    return data, round(latency, 1)


def call_ask(query: str, api_url: str) -> tuple[dict, float]:
    """Call POST /ask. Returns (response_json, latency_ms)."""
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{api_url}/ask",
            json={"query": query},
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        data = {"error": str(exc), "answer": "", "tool_calls": []}
    latency = (time.perf_counter() - t0) * 1000
    return data, round(latency, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-query evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_rag_query(qdef: dict, api_url: str, use_judge: bool) -> dict:
    """Run one query through /rag and score it."""
    resp, latency = call_rag(qdef["query"], api_url)

    answer   = resp.get("answer", "")
    sources  = resp.get("sources", [])
    error    = resp.get("error")

    khr    = keyword_hit_rate(answer, qdef.get("expected_answer_keywords", []))
    judge  = llm_judge(qdef["query"], answer) if use_judge and answer else None

    return {
        "id":                 qdef["id"],
        "category":           qdef["category"],
        "query":              qdef["query"],
        "endpoint":           "rag",
        "answer":             answer[:500],
        "error":              error,
        "has_results":        len(sources) >= qdef.get("min_results_expected", 1),
        "retrieved_count":    len(sources),
        "latency_ms":         latency,
        "keyword_hit_rate":   khr,
        "answer_relevance":   judge,
        "filter_accuracy":    None,     # N/A for /rag
    }


def evaluate_ask_query(qdef: dict, api_url: str, use_judge: bool) -> dict:
    """Run one query through /ask and score it."""
    resp, latency = call_ask(qdef["query"], api_url)

    answer     = resp.get("answer", "")
    tool_calls = resp.get("tool_calls", [])
    error      = resp.get("error")

    # Count results from tool_calls (rough proxy — agent may embed count in answer)
    result_count = sum(
        len(tc.get("results", [])) for tc in tool_calls
        if isinstance(tc.get("results"), list)
    )
    # Fallback: check if answer mentions listings
    if result_count == 0 and "listing" in answer.lower():
        result_count = 1

    khr   = keyword_hit_rate(answer, qdef.get("expected_answer_keywords", []))
    fa    = filter_accuracy(tool_calls, qdef.get("expected_filters", {}))
    judge = llm_judge(qdef["query"], answer) if use_judge and answer else None

    return {
        "id":                 qdef["id"],
        "category":           qdef["category"],
        "query":              qdef["query"],
        "endpoint":           "ask",
        "answer":             answer[:500],
        "error":              error,
        "has_results":        result_count >= qdef.get("min_results_expected", 1),
        "retrieved_count":    result_count,
        "latency_ms":         latency,
        "keyword_hit_rate":   khr,
        "answer_relevance":   judge,
        "filter_accuracy":    fa,
        "tool_calls":         tool_calls,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    api_url:   str,
    endpoint:  str | None,    # "rag", "ask", or None (both)
    ids:       list[str] | None,
    use_judge: bool,
) -> list[dict]:
    dataset_path = ROOT / config.EVAL_DATASET
    with open(dataset_path) as f:
        dataset = json.load(f)

    queries = dataset["queries"]

    # Filter by id if requested
    if ids:
        queries = [q for q in queries if q["id"] in ids]

    results: list[dict] = []
    total = len(queries)

    for i, qdef in enumerate(queries, 1):
        ep    = qdef.get("endpoint", "both")   # "rag", "ask", or "both"
        qid   = qdef["id"]
        query = qdef["query"]

        print(f"\n[{i:02d}/{total:02d}] {qid} ({qdef['category']}) — \"{query[:60]}\"")

        # Decide which endpoints to hit
        run_rag = (endpoint in (None, "rag")) and ep in ("rag", "both")
        run_ask = (endpoint in (None, "ask")) and ep in ("ask", "both")

        if run_rag:
            print(f"  → /rag ...", end=" ", flush=True)
            r = evaluate_rag_query(qdef, api_url, use_judge)
            print(f"latency={r['latency_ms']:.0f}ms  "
                  f"results={r['retrieved_count']}  "
                  f"khr={r['keyword_hit_rate']:.2f}  "
                  f"judge={r['answer_relevance']}")
            results.append(r)

        if run_ask:
            print(f"  → /ask ...", end=" ", flush=True)
            r = evaluate_ask_query(qdef, api_url, use_judge)
            print(f"latency={r['latency_ms']:.0f}ms  "
                  f"filter_acc={r['filter_accuracy']:.2f}  "
                  f"khr={r['keyword_hit_rate']:.2f}  "
                  f"judge={r['answer_relevance']}")
            results.append(r)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# GCS upload helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _upload_bytes(bucket_name: str, gcs_path: str, data: bytes, content_type: str) -> str:
    """Upload bytes to GCS and return the public gs:// URI."""
    storage_client = gcs.Client(project=config.PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blob   = bucket.blob(gcs_path)
    blob.upload_from_string(data, content_type=content_type)
    return f"gs://{bucket_name}/{gcs_path}"


def save_results_to_gcs(results: list[dict], summary: dict) -> dict[str, str]:
    """
    Upload eval results (JSON + HTML) to GCS.

    Writes two sets of files:
      timestamped  → eval/eval_YYYYMMDD_HHMMSS.{json,html}
      latest       → eval/latest.{json,html}  (overwritten each run)

    Returns dict of {json_uri, html_uri, latest_json_uri, latest_html_uri}
    """
    ts     = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = getattr(config, "EVAL_GCS_PREFIX", "eval/")
    bucket = config.BUCKET_NAME

    payload = {"summary": summary, "results": results}

    # ── JSON ──────────────────────────────────────────────────────────────────
    json_bytes = json.dumps(payload, indent=2, ensure_ascii=False).encode()
    ts_json    = _upload_bytes(bucket, f"{prefix}eval_{ts}.json",
                               json_bytes, "application/json")
    latest_json = _upload_bytes(bucket, f"{prefix}latest.json",
                                json_bytes, "application/json")

    # ── HTML ──────────────────────────────────────────────────────────────────
    html_bytes = build_html_report(results, summary).encode("utf-8")
    ts_html    = _upload_bytes(bucket, f"{prefix}eval_{ts}.html",
                               html_bytes, "text/html; charset=utf-8")
    latest_html = _upload_bytes(bucket, f"{prefix}latest.html",
                                html_bytes, "text/html; charset=utf-8")

    return {
        "json_uri":        ts_json,
        "html_uri":        ts_html,
        "latest_json_uri": latest_json,
        "latest_html_uri": latest_html,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Airbnb RAG offline evaluator")
    p.add_argument("--endpoint",  choices=["rag", "ask"],
                   help="Evaluate only this endpoint (default: both)")
    p.add_argument("--ids",       type=str, default=None,
                   help="Comma-separated query IDs to run (e.g. q001,q005)")
    p.add_argument("--no-judge",  action="store_true",
                   help="Skip LLM-as-judge scoring (faster, no API cost)")
    p.add_argument("--api-url",   type=str, default=config.API_BASE_URL,
                   help=f"FastAPI base URL (default: {config.API_BASE_URL})")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    ids    = [x.strip() for x in args.ids.split(",")] if args.ids else None

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       Airbnb Agentic RAG — Offline Evaluation               ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  API URL   : {args.api_url}")
    print(f"  Endpoint  : {args.endpoint or 'both'}")
    print(f"  Query IDs : {', '.join(ids) if ids else 'all'}")
    print(f"  LLM judge : {'OFF' if args.no_judge else 'ON  (Gemini as judge)'}")
    print(f"  Top-K     : {config.EVAL_TOP_K}")
    print()

    # Health check
    try:
        r = requests.get(f"{args.api_url}/health", timeout=5)
        r.raise_for_status()
        print(f"  ✓  Backend healthy: {r.json()}\n")
    except Exception as exc:
        print(f"  ✗  Cannot reach backend at {args.api_url}: {exc}")
        print("     Start it with: uvicorn app.main:app --port 8000")
        sys.exit(1)

    t0      = time.time()
    results = run_evaluation(
        api_url   = args.api_url,
        endpoint  = args.endpoint,
        ids       = ids,
        use_judge = not args.no_judge,
    )
    elapsed = time.time() - t0

    summary = aggregate(results)

    print()
    print("═" * 62)
    print("  SUMMARY")
    print("═" * 62)
    print(f"  Queries evaluated : {summary['total_queries']}")
    if "rag_count" in summary:
        print(f"  /rag avg relevance   : {summary.get('rag_avg_relevance', 'N/A')} / 4")
        print(f"  /rag avg latency     : {summary.get('rag_avg_latency_ms', 'N/A')} ms")
    if "ask_count" in summary:
        print(f"  /ask avg relevance   : {summary.get('ask_avg_relevance', 'N/A')} / 4")
        print(f"  /ask filter accuracy : {summary.get('ask_avg_filter_accuracy', 'N/A')}")
        print(f"  /ask avg latency     : {summary.get('ask_avg_latency_ms', 'N/A')} ms")
    print(f"  Total eval time      : {elapsed:.1f}s")
    print()

    print("  Uploading reports to GCS...")
    uris = save_results_to_gcs(results, summary)
    print()
    print(f"  ✓  Reports uploaded:")
    print(f"     {uris['json_uri']}")
    print(f"     {uris['html_uri']}")
    print(f"     {uris['latest_json_uri']}  (latest)")
    print(f"     {uris['latest_html_uri']}  (latest)")
    print()
    print("  To view the HTML report:")
    print(f"    gsutil cp {uris['latest_html_uri']} /tmp/eval_latest.html && open /tmp/eval_latest.html")
    print()


if __name__ == "__main__":
    main()
