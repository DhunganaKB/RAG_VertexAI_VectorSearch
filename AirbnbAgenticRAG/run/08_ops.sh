#!/usr/bin/env bash
# =============================================================================
# 08_ops.sh — Day-to-day operations (cache, Redis, re-ingestion)
# =============================================================================
# Usage:
#   ./run/08_ops.sh cache-stats     # view cache hit/miss stats
#   ./run/08_ops.sh cache-flush     # flush all cache entries
#   ./run/08_ops.sh redis-stop      # delete Redis + VPC connector (save cost)
#   ./run/08_ops.sh redis-start     # re-create Redis + VPC connector
#   ./run/08_ops.sh reindex         # delete collection + re-run full pipeline
#   ./run/08_ops.sh logs            # tail Cloud Run API logs
# =============================================================================
set -euo pipefail
[[ -f .env ]] && export $(grep -v '^#' .env | grep -v '^$' | xargs)

info()    { echo -e "\n\033[1;34m▶  $*\033[0m"; }
success() { echo -e "\033[1;32m✓  $*\033[0m"; }
warn()    { echo -e "\033[1;33m⚠  $*\033[0m"; }
err()     { echo -e "\033[1;31m✗  $*\033[0m"; exit 1; }

PROJECT_ID="${GCP_PROJECT_ID:?'GCP_PROJECT_ID not set.'}"
REGION="${GCP_REGION:-us-central1}"
REDIS_INSTANCE="airbnb-rag-cache"
VPC_CONNECTOR="airbnb-rag-connector"

[[ $# -eq 0 ]] && { echo "Usage: ./run/08_ops.sh <command>"; echo "Commands: cache-stats | cache-flush | redis-stop | redis-start | reindex | logs"; exit 0; }

API_URL=$(gcloud run services describe airbnb-rag-api \
  --region="$REGION" --project="$PROJECT_ID" --format="value(status.url)" 2>/dev/null || echo "")

case "$1" in

  # ── Cache stats ──────────────────────────────────────────────────────────────
  cache-stats)
    info "Cache stats..."
    [[ -z "$API_URL" ]] && err "API not deployed yet"
    curl -s "${API_URL}/cache/stats" | python -m json.tool
    ;;

  # ── Cache flush ──────────────────────────────────────────────────────────────
  cache-flush)
    info "Flushing all cache entries..."
    [[ -z "$API_URL" ]] && err "API not deployed yet"
    curl -s -X POST "${API_URL}/cache/flush" | python -m json.tool
    success "Cache flushed"
    ;;

  # ── Stop Redis (cost saving) ──────────────────────────────────────────────────
  redis-stop)
    warn "This will DELETE the Redis instance and stop billing (~\$40-55/month saved)."
    warn "All cached data will be lost. The app will keep working without cache."
    read -p "Continue? [y/N] " confirm
    [[ "$confirm" != "y" ]] && { echo "Aborted."; exit 0; }

    info "Removing REDIS_HOST from Cloud Run service..."
    gcloud run services update airbnb-rag-api \
      --region="$REGION" --project="$PROJECT_ID" \
      --remove-env-vars=REDIS_HOST \
      --vpc-egress=all-traffic

    info "Deleting Memorystore Redis instance..."
    gcloud redis instances delete "$REDIS_INSTANCE" \
      --region="$REGION" --project="$PROJECT_ID" --quiet
    success "Redis instance deleted"

    info "Deleting VPC Access connector..."
    gcloud run services update airbnb-rag-api \
      --region="$REGION" --project="$PROJECT_ID" --clear-vpc-connector
    gcloud compute networks vpc-access connectors delete "$VPC_CONNECTOR" \
      --region="$REGION" --project="$PROJECT_ID" --quiet
    success "VPC connector deleted"
    success "Redis stopped — billing will stop within a few minutes"
    ;;

  # ── Start Redis ───────────────────────────────────────────────────────────────
  redis-start)
    info "Re-creating Memorystore Redis instance..."
    gcloud redis instances create "$REDIS_INSTANCE" \
      --size=1 --region="$REGION" --network=default \
      --redis-version=redis_7_0 --tier=BASIC \
      --project="$PROJECT_ID"

    info "Waiting for Redis to reach READY state (~5 min)..."
    for i in $(seq 1 30); do
      STATE=$(gcloud redis instances describe "$REDIS_INSTANCE" \
        --region="$REGION" --project="$PROJECT_ID" --format="value(state)")
      [[ "$STATE" == "READY" ]] && { success "Redis is READY"; break; }
      echo "  State: $STATE — waiting 20s ($i/30)..."
      sleep 20
    done

    REDIS_HOST=$(gcloud redis instances describe "$REDIS_INSTANCE" \
      --region="$REGION" --project="$PROJECT_ID" --format="value(host)")

    info "Re-creating VPC Access connector..."
    gcloud compute networks vpc-access connectors create "$VPC_CONNECTOR" \
      --region="$REGION" --network=default \
      --range=10.8.0.0/28 --min-instances=2 --max-instances=3 \
      --machine-type=e2-micro --project="$PROJECT_ID"

    info "Re-wiring Cloud Run service to Redis..."
    gcloud run services update airbnb-rag-api \
      --region="$REGION" --project="$PROJECT_ID" \
      --set-env-vars="REDIS_HOST=${REDIS_HOST},REDIS_PORT=6379" \
      --vpc-connector="$VPC_CONNECTOR" \
      --vpc-egress=private-ranges-only
    success "Redis re-connected: $REDIS_HOST:6379"
    ;;

  # ── Re-ingestion ──────────────────────────────────────────────────────────────
  reindex)
    warn "This will DELETE the existing collection and re-run the full pipeline."
    read -p "Continue? [y/N] " confirm
    [[ "$confirm" != "y" ]] && { echo "Aborted."; exit 0; }

    info "Deleting existing collection..."
    python scripts/00_delete_collections.py

    info "Running full pipeline..."
    ./run/02_run_pipeline.sh

    info "Flushing stale cache entries..."
    [[ -n "$API_URL" ]] && curl -s -X POST "${API_URL}/cache/flush" | python -m json.tool
    success "Re-ingestion complete"
    ;;

  # ── Tail logs ─────────────────────────────────────────────────────────────────
  logs)
    info "Tailing Cloud Run API logs (Ctrl+C to stop)..."
    gcloud run services logs tail airbnb-rag-api \
      --region="$REGION" --project="$PROJECT_ID"
    ;;

  *)
    err "Unknown command: $1. Use: cache-stats | cache-flush | redis-stop | redis-start | reindex | logs"
    ;;
esac
