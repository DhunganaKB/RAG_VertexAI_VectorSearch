#!/usr/bin/env bash
# =============================================================================
# 01_setup_gcp.sh — One-time GCP project setup
# =============================================================================
# Run ONCE per project to:
#   - Enable all required GCP APIs
#   - Create the service account
#   - Create the custom Vector Search IAM role
#   - Grant all required IAM permissions
#   - Create Memorystore Redis instance
#   - Create Serverless VPC Access connector
#
# Usage:
#   chmod +x run/01_setup_gcp.sh
#   ./run/01_setup_gcp.sh
#
# Idempotent — safe to re-run. Already-existing resources are skipped.
# =============================================================================
set -euo pipefail

# ── Load .env if present ─────────────────────────────────────────────────────
[[ -f .env ]] && export $(grep -v '^#' .env | grep -v '^$' | xargs)

# ── Config — reads from .env or environment variables ────────────────────────
PROJECT_ID="${GCP_PROJECT_ID:?'GCP_PROJECT_ID is not set. Copy .env.example to .env and fill in your project ID.'}"
REGION="${GCP_REGION:-us-central1}"
SA_NAME="airbnb-rag-sa"
REDIS_INSTANCE="airbnb-rag-cache"
VPC_CONNECTOR="airbnb-rag-connector"
VPC_CONNECTOR_RANGE="10.8.0.0/28"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# ── Helpers ───────────────────────────────────────────────────────────────────
info()    { echo -e "\n\033[1;34m▶  $*\033[0m"; }
success() { echo -e "\033[1;32m✓  $*\033[0m"; }
warn()    { echo -e "\033[1;33m⚠  $*\033[0m"; }
err()     { echo -e "\033[1;31m✗  $*\033[0m"; exit 1; }

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           AirbnbAgenticRAG — GCP Setup                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── 1. Set active project ─────────────────────────────────────────────────────
info "Setting active project: $PROJECT_ID"
gcloud config set project "$PROJECT_ID"

# ── 2. Enable all required APIs ───────────────────────────────────────────────
info "Enabling required GCP APIs..."
gcloud services enable \
  aiplatform.googleapis.com \
  vectorsearch.googleapis.com \
  redis.googleapis.com \
  vpcaccess.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  cloudbuild.googleapis.com \
  --project="$PROJECT_ID"
success "All APIs enabled"

# ── 3. Artifact Registry repository ──────────────────────────────────────────
info "Creating Artifact Registry Docker repository..."
if gcloud artifacts repositories describe airbnb-rag \
    --location="$REGION" --project="$PROJECT_ID" &>/dev/null; then
  warn "Repository 'airbnb-rag' already exists — skipping"
else
  gcloud artifacts repositories create airbnb-rag \
    --repository-format=docker \
    --location="$REGION" \
    --project="$PROJECT_ID"
  success "Artifact Registry repository created"
fi

# ── 4. Service account ────────────────────────────────────────────────────────
info "Creating service account: $SA_NAME"
if gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJECT_ID" &>/dev/null; then
  warn "Service account already exists — skipping"
else
  gcloud iam service-accounts create "$SA_NAME" \
    --display-name="Airbnb RAG Service Account" \
    --project="$PROJECT_ID"
  success "Service account created: $SA_EMAIL"
fi

# ── 5. Custom Vector Search IAM role ─────────────────────────────────────────
# IMPORTANT: roles/aiplatform.user has ZERO vectorsearch.* permissions.
# You MUST create this custom role or you will get 403 errors.
info "Creating custom Vector Search IAM role..."
if gcloud iam roles describe vectorSearchUser --project="$PROJECT_ID" &>/dev/null; then
  warn "Custom role 'vectorSearchUser' already exists — skipping"
else
  gcloud iam roles create vectorSearchUser \
    --project="$PROJECT_ID" \
    --title="Vector Search User" \
    --description="Read/write/search Vector Search 2.0 DataObjects" \
    --permissions="vectorsearch.dataObjects.search,vectorsearch.dataObjects.get,vectorsearch.dataObjects.create,vectorsearch.dataObjects.update,vectorsearch.dataObjects.delete,vectorsearch.dataObjects.query,vectorsearch.collections.get,vectorsearch.indexes.get,vectorsearch.indexes.list,vectorsearch.locations.get" \
    --stage="GA"
  success "Custom role 'vectorSearchUser' created"
fi

# ── 6. Grant IAM roles ────────────────────────────────────────────────────────
info "Granting IAM roles to service account..."
for ROLE in \
  roles/aiplatform.user \
  "projects/${PROJECT_ID}/roles/vectorSearchUser" \
  roles/storage.objectViewer \
  roles/storage.objectCreator \
  roles/artifactregistry.reader; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="$ROLE" \
    --condition=None --quiet
  echo "  granted: $ROLE"
done
success "All IAM roles granted"

# ── 7. Memorystore Redis instance ─────────────────────────────────────────────
info "Creating Memorystore Redis instance: $REDIS_INSTANCE"
if gcloud redis instances describe "$REDIS_INSTANCE" \
    --region="$REGION" --project="$PROJECT_ID" &>/dev/null; then
  warn "Redis instance already exists — skipping"
else
  gcloud redis instances create "$REDIS_INSTANCE" \
    --size=1 --region="$REGION" --network=default \
    --redis-version=redis_7_0 --tier=BASIC \
    --project="$PROJECT_ID"
  success "Redis instance created (takes ~5 min to reach READY)"
fi

info "Waiting for Redis to reach READY state..."
for i in $(seq 1 30); do
  STATE=$(gcloud redis instances describe "$REDIS_INSTANCE" \
    --region="$REGION" --project="$PROJECT_ID" --format="value(state)" 2>/dev/null || echo "UNKNOWN")
  [[ "$STATE" == "READY" ]] && { success "Redis is READY"; break; }
  [[ "$i" -eq 30 ]] && err "Redis timed out. Check GCP Console."
  echo "  State: $STATE — waiting 20s ($i/30)..."
  sleep 20
done

REDIS_HOST=$(gcloud redis instances describe "$REDIS_INSTANCE" \
  --region="$REGION" --project="$PROJECT_ID" --format="value(host)")
success "Redis host: $REDIS_HOST:6379"

# ── 8. Serverless VPC Access connector ───────────────────────────────────────
info "Creating Serverless VPC Access connector: $VPC_CONNECTOR"
if gcloud compute networks vpc-access connectors describe "$VPC_CONNECTOR" \
    --region="$REGION" --project="$PROJECT_ID" &>/dev/null; then
  warn "VPC connector already exists — skipping"
else
  gcloud compute networks vpc-access connectors create "$VPC_CONNECTOR" \
    --region="$REGION" --network=default \
    --range="$VPC_CONNECTOR_RANGE" \
    --min-instances=2 --max-instances=3 \
    --machine-type=e2-micro \
    --project="$PROJECT_ID"
  success "VPC connector created"
fi

info "Waiting for VPC connector to reach READY state..."
for i in $(seq 1 15); do
  STATE=$(gcloud compute networks vpc-access connectors describe "$VPC_CONNECTOR" \
    --region="$REGION" --project="$PROJECT_ID" --format="value(state)" 2>/dev/null || echo "UNKNOWN")
  [[ "$STATE" == "READY" ]] && { success "VPC connector is READY"; break; }
  [[ "$i" -eq 15 ]] && err "VPC connector timed out. Check GCP Console."
  echo "  State: $STATE — waiting 15s ($i/15)..."
  sleep 15
done

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║            GCP Setup Complete                               ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Next step:  ./run/02_run_pipeline.sh                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
