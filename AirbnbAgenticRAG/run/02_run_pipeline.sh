#!/usr/bin/env bash
# =============================================================================
# 02_run_pipeline.sh — Offline Data Ingestion Pipeline
# =============================================================================
# Runs all three pipeline stages in order:
#   Stage 1 — Create VS2.0 Collection with typed schema
#   Stage 2 — Embed + ingest 3,000 Airbnb listings as DataObjects
#   Stage 3 — Build ScaNN ANN index on the embedding field
#
# Usage:
#   ./run/02_run_pipeline.sh              # full pipeline (~15–45 min total)
#   ./run/02_run_pipeline.sh --quick      # ingest only 100 listings (dev/test)
#   ./run/02_run_pipeline.sh --skip-index # skip index build (use exact kNN)
#   ./run/02_run_pipeline.sh --verify     # inspect collection after run
#
# Idempotent — stages are safe to re-run.
# Run from the project root directory.
# =============================================================================
set -euo pipefail
[[ -f .env ]] && export $(grep -v '^#' .env | grep -v '^$' | xargs)

info()    { echo -e "\n\033[1;34m▶  $*\033[0m"; }
success() { echo -e "\033[1;32m✓  $*\033[0m"; }
warn()    { echo -e "\033[1;33m⚠  $*\033[0m"; }

QUICK=false
SKIP_INDEX=false
VERIFY=false

for arg in "$@"; do
  case "$arg" in
    --quick)      QUICK=true ;;
    --skip-index) SKIP_INDEX=true ;;
    --verify)     VERIFY=true ;;
  esac
done

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         AirbnbAgenticRAG — Data Pipeline                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Verify GCS data file is in place ─────────────────────────────────────────
info "Verifying data file in GCS..."
GCS_URI=$(python -c "import config; print(config.GCS_DATA_URI)")
gsutil ls -lh "$GCS_URI" && success "Data file found: $GCS_URI" \
  || { echo "Data not found. Uploading..."; python scripts/00_upload_data.py; }

# ── Stage 1: Create VS2.0 Collection ─────────────────────────────────────────
info "Stage 1 — Creating VS2.0 Collection..."
python scripts/01_setup_collection.py
success "Stage 1 complete — Collection ready"

# ── Stage 2: Ingest listings ──────────────────────────────────────────────────
info "Stage 2 — Embedding and ingesting listings..."
if $QUICK; then
  warn "Quick mode: ingesting only 100 listings"
  python scripts/04_build_pipeline.py --max-listings 100 --skip-collection --skip-index
else
  python scripts/02_ingest.py
fi
success "Stage 2 complete — Listings ingested"

# ── Stage 3: Build ScaNN ANN index ───────────────────────────────────────────
if $SKIP_INDEX; then
  warn "Skipping index build (--skip-index). Searches will use exact kNN."
else
  info "Stage 3 — Building ScaNN ANN index (server-side, ~5–20 min)..."
  python scripts/03_create_index.py
  success "Stage 3 complete — ScaNN index built"
fi

# ── Optional: Verify collection ───────────────────────────────────────────────
if $VERIFY; then
  info "Verifying collection..."
  python scripts/inspect_collection.py
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║            Pipeline Complete                                ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Run locally:   ./run/03_run_local.sh                       ║"
echo "║  Deploy to GCP: ./run/04_deploy.sh                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
