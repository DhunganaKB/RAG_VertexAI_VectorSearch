#!/usr/bin/env bash
# =============================================================================
# 07_run_loadtest.sh — Locust load testing against the Cloud Run API
# =============================================================================
# Runs load tests in stages — smoke → load → stress.
# All results saved to load_test/results/ as HTML + CSV reports.
#
# Usage:
#   ./run/07_run_loadtest.sh           # smoke → load → stress (full sequence)
#   ./run/07_run_loadtest.sh --smoke   # smoke only (5 users, 60s)
#   ./run/07_run_loadtest.sh --load    # load test only (20 users, 5 min)
#   ./run/07_run_loadtest.sh --stress  # stress test only (50 users, 10 min)
#   ./run/07_run_loadtest.sh --cache   # cache validation (miss vs hit)
#   ./run/07_run_loadtest.sh --ui      # open Locust web dashboard
#
# Run from the project root directory.
# =============================================================================
set -euo pipefail
[[ -f .env ]] && export $(grep -v '^#' .env | grep -v '^$' | xargs)

info()    { echo -e "\n\033[1;34m▶  $*\033[0m"; }
success() { echo -e "\033[1;32m✓  $*\033[0m"; }
warn()    { echo -e "\033[1;33m⚠  $*\033[0m"; }

PROJECT_ID="${GCP_PROJECT_ID:?'GCP_PROJECT_ID not set.'}"
REGION="${GCP_REGION:-us-central1}"
RESULTS_DIR="load_test/results"
LOCUSTFILE="load_test/locustfile.py"

MODE="all"
for arg in "$@"; do
  case "$arg" in
    --smoke)  MODE="smoke" ;;
    --load)   MODE="load" ;;
    --stress) MODE="stress" ;;
    --cache)  MODE="cache" ;;
    --ui)     MODE="ui" ;;
  esac
done

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         AirbnbAgenticRAG — Load Testing                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Get API URL ───────────────────────────────────────────────────────────────
API_URL=$(gcloud run services describe airbnb-rag-api \
  --region="$REGION" --project="$PROJECT_ID" --format="value(status.url)")
info "Testing against: $API_URL"

# ── Verify Locust is installed ────────────────────────────────────────────────
locust --version &>/dev/null || { warn "Locust not found. Installing..."; pip install "locust>=2.28.0"; }
mkdir -p "$RESULTS_DIR"

# ── Smoke test ────────────────────────────────────────────────────────────────
run_smoke() {
  info "Smoke Test — 5 users, 60 seconds"
  warn "Expected: all requests succeed, /rag latency 800–3000ms (cold)"
  locust -f "$LOCUSTFILE" --host="$API_URL" --headless \
    --users=5 --spawn-rate=1 --run-time=60s \
    --html="${RESULTS_DIR}/smoke_report.html" \
    --csv="${RESULTS_DIR}/smoke_stats"
  open "${RESULTS_DIR}/smoke_report.html"
  success "Smoke test complete → ${RESULTS_DIR}/smoke_report.html"
}

# ── Load test ─────────────────────────────────────────────────────────────────
run_load() {
  info "Load Test — 20 users, 5 minutes"
  warn "Expected: /rag p50 1000–4000ms, /ask p50 3000–8000ms, errors <2%"
  locust -f "$LOCUSTFILE" --host="$API_URL" --headless \
    --users=20 --spawn-rate=2 --run-time=5m \
    --html="${RESULTS_DIR}/load_report.html" \
    --csv="${RESULTS_DIR}/load_stats"
  open "${RESULTS_DIR}/load_report.html"
  success "Load test complete → ${RESULTS_DIR}/load_report.html"
}

# ── Stress test ───────────────────────────────────────────────────────────────
run_stress() {
  info "Stress Test — 50 users, 10 minutes"
  warn "Gemini tokens WILL be consumed. Watch for 429 quota errors."
  warn "Expected: Cloud Run auto-scales; errors <5%"
  locust -f "$LOCUSTFILE" --host="$API_URL" --headless \
    --users=50 --spawn-rate=5 --run-time=10m \
    --html="${RESULTS_DIR}/stress_report.html" \
    --csv="${RESULTS_DIR}/stress_stats"
  open "${RESULTS_DIR}/stress_report.html"
  success "Stress test complete → ${RESULTS_DIR}/stress_report.html"
}

# ── Cache validation ──────────────────────────────────────────────────────────
run_cache() {
  info "Cache Validation — Run 1: MISS (cold)"
  locust -f "$LOCUSTFILE" --host="$API_URL" --headless \
    --users=5 --spawn-rate=1 --run-time=90s \
    --html="${RESULTS_DIR}/cache_miss_report.html"

  info "Cache Validation — Run 2: HIT (warm, same queries)"
  warn "Expected: 10–50x faster than Run 1 (~20–80ms)"
  locust -f "$LOCUSTFILE" --host="$API_URL" --headless \
    --users=5 --spawn-rate=1 --run-time=90s \
    --html="${RESULTS_DIR}/cache_hit_report.html"

  open "${RESULTS_DIR}/cache_miss_report.html"
  open "${RESULTS_DIR}/cache_hit_report.html"

  echo "  Cache stats after test:"
  curl -s "${API_URL}/cache/stats" | python -c \
    "import json,sys; d=json.load(sys.stdin); print(f\"  hits={d.get('hits',0)}  misses={d.get('misses',0)}  keys={d.get('total_keys',0)}\")"
  success "Cache validation complete"
}

# ── Web UI mode ───────────────────────────────────────────────────────────────
run_ui() {
  info "Launching Locust web dashboard at http://localhost:8089"
  warn "Set users=20, spawn-rate=2 and click 'Start swarming'"
  open http://localhost:8089 &
  locust -f "$LOCUSTFILE" --host="$API_URL"
}

# ── Execute ───────────────────────────────────────────────────────────────────
case "$MODE" in
  smoke)  run_smoke ;;
  load)   run_load ;;
  stress) run_stress ;;
  cache)  run_cache ;;
  ui)     run_ui ;;
  all)
    run_smoke
    echo ""; warn "Pausing 30s between tests..."; sleep 30
    run_load
    echo ""; warn "Pausing 30s between tests..."; sleep 30
    run_stress
    ;;
esac

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Load Testing Complete                                      ║"
echo "║  Reports: load_test/results/                                ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Operations: ./run/08_ops.sh                                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
