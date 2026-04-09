#!/usr/bin/env bash
# =============================================================================
# 04_deploy.sh — Build Docker images and deploy both Cloud Run services
# =============================================================================
# Builds the API and UI Docker images using Cloud Build (AMD64, no local
# Docker needed), then deploys both services to Cloud Run.
#
# Usage:
#   ./run/04_deploy.sh          # build + deploy both API and UI
#   ./run/04_deploy.sh --api    # build + deploy API only
#   ./run/04_deploy.sh --ui     # build + deploy UI only
#
# Requires: ./run/01_setup_gcp.sh must have been run first.
# Run from the project root directory.
#
# NOTE: Uses Cloud Build to avoid local Docker proxy issues on Apple Silicon.
# =============================================================================
set -euo pipefail

info()    { echo -e "\n\033[1;34m▶  $*\033[0m"; }
success() { echo -e "\033[1;32m✓  $*\033[0m"; }
warn()    { echo -e "\033[1;33m⚠  $*\033[0m"; }
err()     { echo -e "\033[1;31m✗  $*\033[0m"; exit 1; }

# ── Load .env if present ─────────────────────────────────────────────────────
[[ -f .env ]] && export $(grep -v '^#' .env | grep -v '^$' | xargs)

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID="${GCP_PROJECT_ID:?'GCP_PROJECT_ID not set. Copy .env.example to .env.'}"
REGION="${GCP_REGION:-us-central1}"
SA_EMAIL="airbnb-rag-sa@${PROJECT_ID}.iam.gserviceaccount.com"
REDIS_INSTANCE="airbnb-rag-cache"
VPC_CONNECTOR="airbnb-rag-connector"
API_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/airbnb-rag/api:latest"
UI_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/airbnb-rag/ui:latest"

API_ONLY=false
UI_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --api) API_ONLY=true ;;
    --ui)  UI_ONLY=true ;;
  esac
done

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         AirbnbAgenticRAG — Deploy to Cloud Run              ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Grant Cloud Build SA artifact write access (idempotent) ──────────────────
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
CB_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${CB_SA}" --role="roles/artifactregistry.writer" \
  --condition=None --quiet 2>/dev/null || true

# ── Helper: build image via Cloud Build ──────────────────────────────────────
build_image() {
  local image="$1" dockerfile="$2"
  local tmp; tmp=$(mktemp /tmp/cloudbuild-XXXXXX.yaml)
  cat > "$tmp" <<EOF
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-f', '${dockerfile}', '-t', '${image}', '.']
images: ['${image}']
options:
  logging: CLOUD_LOGGING_ONLY
EOF
  gcloud builds submit --config="$tmp" --project="$PROJECT_ID" .
  rm -f "$tmp"
}

# ── Get Redis host ────────────────────────────────────────────────────────────
REDIS_HOST=$(gcloud redis instances describe "$REDIS_INSTANCE" \
  --region="$REGION" --project="$PROJECT_ID" --format="value(host)" 2>/dev/null \
  || err "Redis instance '$REDIS_INSTANCE' not found. Run ./run/01_setup_gcp.sh first.")

# ── Build + deploy API ────────────────────────────────────────────────────────
if ! $UI_ONLY; then
  info "Building API image via Cloud Build..."
  build_image "$API_IMAGE" "Dockerfile.api"
  success "API image built: $API_IMAGE"

  info "Deploying API service to Cloud Run..."
  gcloud run deploy airbnb-rag-api \
    --image="$API_IMAGE" \
    --region="$REGION" \
    --service-account="$SA_EMAIL" \
    --memory=2Gi --cpu=2 \
    --min-instances=0 --max-instances=10 \
    --timeout=120 --concurrency=80 \
    --cpu-boost \
    --vpc-connector="$VPC_CONNECTOR" \
    --vpc-egress=private-ranges-only \
    --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCS_BUCKET_NAME=${GCS_BUCKET_NAME},GCP_REGION=${REGION},REDIS_HOST=${REDIS_HOST},REDIS_PORT=6379,COLLECTION_ID=${COLLECTION_ID:-airbnb-listings-collection},EMBEDDING_MODEL=${EMBEDDING_MODEL:-text-embedding-005},GEMINI_MODEL=${GEMINI_MODEL:-gemini-2.5-flash}" \
    --allow-unauthenticated \
    --project="$PROJECT_ID"
  success "API deployed"
fi

API_URL=$(gcloud run services describe airbnb-rag-api \
  --region="$REGION" --project="$PROJECT_ID" --format="value(status.url)")

# ── Build + deploy UI ─────────────────────────────────────────────────────────
if ! $API_ONLY; then
  info "Building UI image via Cloud Build..."
  build_image "$UI_IMAGE" "Dockerfile.ui"
  success "UI image built: $UI_IMAGE"

  info "Deploying UI service to Cloud Run..."
  gcloud run deploy airbnb-rag-ui \
    --image="$UI_IMAGE" \
    --region="$REGION" \
    --memory=1Gi --cpu=1 \
    --min-instances=0 --max-instances=5 \
    --timeout=60 \
    --set-env-vars="RAG_API_URL=${API_URL}" \
    --allow-unauthenticated \
    --project="$PROJECT_ID"
  success "UI deployed"
fi

UI_URL=$(gcloud run services describe airbnb-rag-ui \
  --region="$REGION" --project="$PROJECT_ID" --format="value(status.url)" 2>/dev/null || echo "N/A")

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              Deployment Complete                            ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  API  : %-52s ║\n" "$API_URL"
printf "║  UI   : %-52s ║\n" "$UI_URL"
printf "║  Docs : %-52s ║\n" "${API_URL}/docs"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Next step:  ./run/05_verify.sh                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
