#!/usr/bin/env bash
# =============================================================================
# 05_verify.sh — Verify deployment is working end-to-end
# =============================================================================
# Runs health check, tests both endpoints, and verifies Redis cache.
#
# Usage:
#   ./run/05_verify.sh
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

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         AirbnbAgenticRAG — Deployment Verification          ║"
echo "╚══════════════════════════════════════════════════════════════╝"

API_URL=$(gcloud run services describe airbnb-rag-api \
  --region="$REGION" --project="$PROJECT_ID" --format="value(status.url)")
info "Testing API at: $API_URL"

# ── 1. Health check ───────────────────────────────────────────────────────────
info "1. Health check..."
HTTP=$(curl -s -o /tmp/health.json -w "%{http_code}" "${API_URL}/health")
[[ "$HTTP" == "200" ]] && success "Health check passed (HTTP 200)" || warn "HTTP $HTTP"
cat /tmp/health.json | python -m json.tool

# ── 2. Test Simple RAG ────────────────────────────────────────────────────────
info "2. Testing /rag endpoint..."
time curl -s -X POST "${API_URL}/rag" \
  -H "Content-Type: application/json" \
  -d '{"query": "cozy studio near downtown Austin", "top_k": 3}' \
  | python -m json.tool | head -30
success "/rag responded"

# ── 3. Test Agentic RAG ───────────────────────────────────────────────────────
info "3. Testing /ask endpoint..."
time curl -s -X POST "${API_URL}/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "2-bedroom entire home under $120 in East Austin"}' \
  | python -m json.tool | head -30
success "/ask responded"

# ── 4. Cache stats ────────────────────────────────────────────────────────────
info "4. Cache stats..."
curl -s "${API_URL}/cache/stats" | python -m json.tool

# ── 5. Cache HIT test (same query, second call should be much faster) ─────────
info "5. Cache HIT test (calling /rag twice with same query)..."
echo "  First call (MISS — expect 800–3000ms):"
time curl -s -X POST "${API_URL}/rag" \
  -H "Content-Type: application/json" \
  -d '{"query": "quiet place with parking near SoCo"}' > /dev/null

echo "  Second call (HIT — expect <80ms):"
time curl -s -X POST "${API_URL}/rag" \
  -H "Content-Type: application/json" \
  -d '{"query": "quiet place with parking near SoCo"}' > /dev/null

echo ""
echo "  Updated cache stats:"
curl -s "${API_URL}/cache/stats" | python -c \
  "import json,sys; d=json.load(sys.stdin); print(f\"  hits={d.get('hits',0)}  misses={d.get('misses',0)}  keys={d.get('total_keys',0)}\")"

success "Verification complete"
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  API docs:  ${API_URL}/docs"
echo "║  Next step: ./run/06_run_eval.sh                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
