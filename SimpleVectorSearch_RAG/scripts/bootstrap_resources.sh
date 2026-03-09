#!/usr/bin/env bash
# =============================================================================
# bootstrap_resources.sh
#
# One-time (idempotent) setup for ALL GCP resources required by this RAG app.
# Safe to re-run — every step checks whether the resource already exists.
#
# What this script creates / configures:
#   1.  Enables required GCP APIs
#   2.  Creates a service account (if missing)
#   3.  Grants IAM roles to the service account
#   4.  Creates the GCS bucket (if missing)
#   5.  Creates the Firestore database in Native mode (if missing)
#   6.  Creates local working directories
#
# Usage:
#   ./scripts/bootstrap_resources.sh
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Read config values from config.py (single source of truth)
# ---------------------------------------------------------------------------
CFG_OUTPUT="$(
python - <<'PY'
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import load_project_settings
s = load_project_settings()
print(s.project_id)
print(s.region)
print(s.bucket_name)
print(s.service_account_name)
print(s.firestore_database)
print(s.firestore_collection)
PY
)"

PROJECT_ID="$(printf '%s\n'       "$CFG_OUTPUT" | sed -n '1p')"
REGION="$(printf '%s\n'           "$CFG_OUTPUT" | sed -n '2p')"
BUCKET_NAME="$(printf '%s\n'      "$CFG_OUTPUT" | sed -n '3p')"
SA_NAME="$(printf '%s\n'          "$CFG_OUTPUT" | sed -n '4p')"
FS_DATABASE="$(printf '%s\n'      "$CFG_OUTPUT" | sed -n '5p')"
FS_COLLECTION="$(printf '%s\n'    "$CFG_OUTPUT" | sed -n '6p')"

if [[ -z "$PROJECT_ID" || -z "$REGION" || -z "$BUCKET_NAME" || -z "$SA_NAME" ]]; then
  echo "ERROR: Could not load values from config.py. Verify PROJECT_ID, REGION, BUCKET_NAME are set."
  exit 1
fi

SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "============================================================"
echo "  Bootstrap — GCP RAG Pipeline"
echo "============================================================"
echo "  Project    : ${PROJECT_ID}"
echo "  Region     : ${REGION}"
echo "  Bucket     : gs://${BUCKET_NAME}"
echo "  SA         : ${SA_EMAIL}"
echo "  Firestore  : db=${FS_DATABASE}  collection=${FS_COLLECTION}"
echo "============================================================"
echo ""

gcloud config set project "${PROJECT_ID}" --quiet

# ---------------------------------------------------------------------------
# 1. Enable required GCP APIs
# ---------------------------------------------------------------------------
echo "[1/6] Enabling GCP APIs..."
gcloud services enable \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  iam.googleapis.com \
  firestore.googleapis.com \
  --quiet
echo "      Done (already-enabled APIs are silently skipped)."
echo ""

# ---------------------------------------------------------------------------
# 2. Service account
# ---------------------------------------------------------------------------
echo "[2/6] Service account..."
if gcloud iam service-accounts describe "${SA_EMAIL}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "      Already exists: ${SA_EMAIL} — skipping."
else
  gcloud iam service-accounts create "${SA_NAME}" \
    --display-name="RAG Vector Search Service Account" \
    --project="${PROJECT_ID}"
  echo "      Created: ${SA_EMAIL}"
fi
echo ""

# ---------------------------------------------------------------------------
# 3. IAM roles (idempotent — add-iam-policy-binding is safe to repeat)
#
#   roles/aiplatform.user       Vertex AI: embeddings, Vector Search, Gemini
#   roles/storage.objectAdmin   GCS: read raw docs, write metadata backup
#   roles/datastore.user        Firestore: read/write chunk metadata
# ---------------------------------------------------------------------------
echo "[3/6] Granting IAM roles..."

grant_project_role() {
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="$1" \
    --condition=None \
    --quiet >/dev/null
  echo "      Granted $1"
}

grant_project_role "roles/aiplatform.user"
grant_project_role "roles/storage.objectAdmin"
grant_project_role "roles/datastore.user"
echo ""

# ---------------------------------------------------------------------------
# 4. GCS bucket
# ---------------------------------------------------------------------------
echo "[4/6] GCS bucket..."
if gcloud storage buckets describe "gs://${BUCKET_NAME}" >/dev/null 2>&1; then
  echo "      Already exists: gs://${BUCKET_NAME} — skipping."
else
  gcloud storage buckets create "gs://${BUCKET_NAME}" \
    --location="${REGION}" \
    --uniform-bucket-level-access \
    --quiet
  echo "      Created: gs://${BUCKET_NAME}"
fi
echo ""

# ---------------------------------------------------------------------------
# 5. Firestore database (Native mode)
#
#    Native mode is required for db.get_all() batched reads used by the app.
#    Datastore mode does NOT support the Python Firestore client library.
# ---------------------------------------------------------------------------
echo "[5/6] Firestore database (Native mode)..."
if gcloud firestore databases describe \
     --database="${FS_DATABASE}" \
     --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "      Already exists: database='${FS_DATABASE}' — skipping."
else
  gcloud firestore databases create \
    --database="${FS_DATABASE}" \
    --location="${REGION}" \
    --type=firestore-native \
    --project="${PROJECT_ID}" \
    --quiet
  echo "      Created Firestore database '${FS_DATABASE}' in ${REGION}."
fi
echo "      Collection '${FS_COLLECTION}' is auto-created on first ingest."
echo ""

# ---------------------------------------------------------------------------
# 6. Local folders
# ---------------------------------------------------------------------------
echo "[6/6] Creating local folders..."
mkdir -p artifacts input_docs ui
echo "      artifacts/  — runtime vector_resources.json + JSONL backup"
echo "      input_docs/ — place your source documents here before uploading"
echo "      ui/         — Streamlit app source"
echo ""

echo "============================================================"
echo "  Bootstrap complete!"
echo "============================================================"
echo ""
echo "  Next steps:"
echo ""
echo "  A) Put your documents in input_docs/ then upload:"
echo "     gcloud storage cp ./input_docs/* gs://${BUCKET_NAME}/raw/"
echo ""
echo "  B) Create Vector Search index + endpoint (idempotent):"
echo "     python scripts/create_vector_search.py"
echo ""
echo "  C) Ingest (chunk -> embed -> Vector Search + Firestore):"
echo "     python scripts/ingest_from_bucket.py"
echo ""
echo "  D) Start FastAPI backend (terminal 1):"
echo "     uvicorn app.main:app --reload --port 8000"
echo ""
echo "  E) Start Streamlit UI (terminal 2):"
echo "     streamlit run ui/streamlit_app.py --server.port 8501"
echo ""
