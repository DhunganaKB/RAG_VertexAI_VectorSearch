#!/usr/bin/env bash
# =============================================================================
# 06_run_eval.sh — Run RAGAS evaluation against the deployed Cloud Run API
# =============================================================================
# Runs 25 labeled queries against the live API, scores them, uploads
# JSON + HTML reports to GCS, and opens the report locally.
#
# Usage:
#   ./run/06_run_eval.sh              # full eval — both endpoints (3–8 min)
#   ./run/06_run_eval.sh --rag        # /rag endpoint only
#   ./run/06_run_eval.sh --ask        # /ask endpoint only
#   ./run/06_run_eval.sh --smoke      # 5 queries only, no LLM judge (fast)
#   ./run/06_run_eval.sh --view       # just download + open latest report
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
ENDPOINT_FLAG=""
SMOKE=false
VIEW_ONLY=false

for arg in "$@"; do
  case "$arg" in
    --rag)   ENDPOINT_FLAG="--endpoint rag" ;;
    --ask)   ENDPOINT_FLAG="--endpoint ask" ;;
    --smoke) SMOKE=true ;;
    --view)  VIEW_ONLY=true ;;
  esac
done

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           AirbnbAgenticRAG — Evaluation                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Get Cloud Run API URL ─────────────────────────────────────────────────────
export RAG_API_URL=$(gcloud run services describe airbnb-rag-api \
  --region="$REGION" --project="$PROJECT_ID" --format="value(status.url)")
info "Evaluating against: $RAG_API_URL"

# ── Verify it's our API ───────────────────────────────────────────────────────
HEALTH=$(curl -s "${RAG_API_URL}/health")
echo "$HEALTH" | python -c "import json,sys; d=json.load(sys.stdin); assert 'collection' in d, 'Wrong API!'" \
  && success "API identity confirmed" \
  || { warn "Health response does not look like our API:"; echo "$HEALTH"; exit 1; }

# ── Download + view latest report only ───────────────────────────────────────
if $VIEW_ONLY; then
  BUCKET=$(python -c "import config; print(config.BUCKET_NAME)")
  PREFIX=$(python -c "import config; print(config.EVAL_GCS_PREFIX)")
  info "Downloading latest eval report..."
  gsutil cp "gs://${BUCKET}/${PREFIX}latest.html" /tmp/eval_latest.html
  open /tmp/eval_latest.html
  exit 0
fi

# ── Run evaluation ────────────────────────────────────────────────────────────
if $SMOKE; then
  info "Smoke eval — 5 queries, no LLM judge..."
  python eval/evaluate.py \
    --api-url "$RAG_API_URL" \
    --ids q001,q006,q011,q016,q021 \
    --no-judge \
    $ENDPOINT_FLAG
else
  info "Full evaluation — all 25 queries with LLM judge (~3–8 min)..."
  python eval/evaluate.py \
    --api-url "$RAG_API_URL" \
    $ENDPOINT_FLAG
fi

# ── Download and open HTML report ─────────────────────────────────────────────
BUCKET=$(python -c "import config; print(config.BUCKET_NAME)")
PREFIX=$(python -c "import config; print(config.EVAL_GCS_PREFIX)")
info "Downloading eval report..."
gsutil cp "gs://${BUCKET}/${PREFIX}latest.html" /tmp/eval_report.html
open /tmp/eval_report.html

success "Evaluation complete — report opened in browser"
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Next step: ./run/07_run_loadtest.sh                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
