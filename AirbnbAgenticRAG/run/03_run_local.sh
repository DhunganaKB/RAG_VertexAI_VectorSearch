#!/usr/bin/env bash
# =============================================================================
# 03_run_local.sh — Run API and UI locally
# =============================================================================
# Starts the FastAPI backend and Streamlit UI locally for development.
#
# Note: Redis cache is DISABLED locally. The app works without it —
#       cache checks silently return None and the full pipeline runs.
#       To test with Redis locally, install Redis and set REDIS_HOST_OVERRIDE.
#
# Usage:
#   ./run/03_run_local.sh          # start both API + UI
#   ./run/03_run_local.sh --api    # start API only (port 8000)
#   ./run/03_run_local.sh --ui     # start UI only (port 8501)
#
# Requires: pip install -r requirements.txt
#           gcloud auth application-default login
# Run from the project root directory.
# =============================================================================
set -euo pipefail

info()    { echo -e "\n\033[1;34m▶  $*\033[0m"; }
success() { echo -e "\033[1;32m✓  $*\033[0m"; }
warn()    { echo -e "\033[1;33m⚠  $*\033[0m"; }

API_ONLY=false
UI_ONLY=false

for arg in "$@"; do
  case "$arg" in
    --api) API_ONLY=true ;;
    --ui)  UI_ONLY=true ;;
  esac
done

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         AirbnbAgenticRAG — Local Development               ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Check ADC authentication ─────────────────────────────────────────────────
info "Checking GCP authentication..."
gcloud auth application-default print-access-token > /dev/null 2>&1 \
  && success "ADC authentication OK" \
  || { warn "Not authenticated. Running:"; gcloud auth application-default login; }

# ── Start API ─────────────────────────────────────────────────────────────────
if ! $UI_ONLY; then
  info "Starting FastAPI backend on http://localhost:8000"
  warn "Redis cache: DISABLED locally (REDIS_HOST is empty — this is expected)"
  warn "Docs: http://localhost:8000/docs"

  if $API_ONLY; then
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
  else
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
    API_PID=$!
    echo "  API PID: $API_PID"
    sleep 3
    # Quick health check
    HTTP=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null || echo "000")
    [[ "$HTTP" == "200" ]] && success "API health check passed" || warn "API not yet ready (HTTP $HTTP) — give it a moment"
  fi
fi

# ── Start UI ──────────────────────────────────────────────────────────────────
if ! $API_ONLY; then
  info "Starting Streamlit UI on http://localhost:8501"
  export RAG_API_URL="http://localhost:8000"
  streamlit run ui/app.py --server.port 8501 --server.headless true
fi

# ── Cleanup trap ──────────────────────────────────────────────────────────────
if ! $API_ONLY && ! $UI_ONLY; then
  trap "kill $API_PID 2>/dev/null; echo 'Stopped API and UI'" EXIT
  wait
fi
