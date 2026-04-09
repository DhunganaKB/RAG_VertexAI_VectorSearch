"""
Airbnb Agentic RAG — Project Configuration
==========================================
Single source of truth for all scripts and the app.

Setup:
  1. Copy .env.example to .env
  2. Fill in your GCP project values
  3. Load with: source .env  (or use python-dotenv)

All project-specific values are read from environment variables.
Do NOT hardcode project IDs or bucket names here — they differ per user.
"""
from __future__ import annotations
import os as _os
import sys as _sys

# ── Optional: load .env file automatically ────────────────────────────────────
# Requires: pip install python-dotenv  (already in requirements.txt)
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(override=False)   # .env takes precedence over system env vars only if override=True
except ImportError:
    pass  # dotenv not installed — rely on shell env vars or export statements


def _require(key: str, default: str = "") -> str:
    """Read env var; warn if empty and no default was intended."""
    val = _os.environ.get(key, default)
    return val


# ─── GCP ─────────────────────────────────────────────────────────────────────
PROJECT_ID = _require("GCP_PROJECT_ID", "")   # e.g. "my-gcp-project"
LOCATION   = _require("GCP_REGION",    "us-central1")

# Warn at import time if PROJECT_ID is not configured
if not PROJECT_ID:
    print(
        "[config] WARNING: GCP_PROJECT_ID is not set. "
        "Copy .env.example → .env and fill in your project ID.",
        file=_sys.stderr,
    )

# ─── Vector Search 2.0 Collection ────────────────────────────────────────────
COLLECTION_ID = _require("COLLECTION_ID", "airbnb-listings-collection")

# ─── GCS ─────────────────────────────────────────────────────────────────────
BUCKET_NAME   = _require("GCS_BUCKET_NAME",    "")       # e.g. "my-rag-bucket"
INPUT_PREFIX  = _require("GCS_INPUT_PREFIX",   "data/")
GCS_FILENAME  = _require("GCS_FILENAME",       "listings.csv.gz")

# ─── AI Models ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = _require("EMBEDDING_MODEL", "text-embedding-005")   # 768-dim
GEMINI_MODEL    = _require("GEMINI_MODEL",    "gemini-2.5-flash")

# ─── Ingestion ────────────────────────────────────────────────────────────────
BATCH_SIZE    = int(_require("BATCH_SIZE",    "50"))
MAX_LISTINGS  = int(_require("MAX_LISTINGS",  "3000"))   # 0 = ingest all

# ─── RAG / Agent ──────────────────────────────────────────────────────────────
TOP_K             = int(_require("TOP_K",             "10"))
MAX_OUTPUT_TOKENS = int(_require("MAX_OUTPUT_TOKENS", "2048"))
TEMPERATURE       = float(_require("TEMPERATURE",     "0.2"))

# ─── Cloud Memorystore for Redis (Cache) ─────────────────────────────────────
# REDIS_HOST is the private VPC IP of your Memorystore instance.
# It is only reachable from Cloud Run via a Serverless VPC Access connector.
# Leave empty locally — the cache client detects this and disables itself.
#
# Set via environment variable (never hardcode an internal IP):
#   Cloud Run:  set REDIS_HOST as an env var in run/04_deploy.sh
#   Local test: export REDIS_HOST=localhost  (if running Redis locally)
CACHE_ENABLED = _require("CACHE_ENABLED", "true").lower() != "false"
REDIS_HOST    = _require("REDIS_HOST",    "")   # empty = cache disabled
REDIS_PORT    = int(_require("REDIS_PORT", "6379"))
REDIS_DB      = 0
CACHE_RAG_TTL = int(_require("CACHE_RAG_TTL", "3600"))   # 1 hour
CACHE_ASK_TTL = int(_require("CACHE_ASK_TTL", "1800"))   # 30 min

# ─── Evaluation ───────────────────────────────────────────────────────────────
EVAL_DATASET    = "eval/dataset.json"   # checked into repo, never modified at runtime
EVAL_GCS_PREFIX = "eval/"              # gs://{BUCKET_NAME}/eval/
EVAL_TOP_K      = int(_require("EVAL_TOP_K", "10"))

# ─── Serving ──────────────────────────────────────────────────────────────────
API_BASE_URL   = _require("RAG_API_URL", "http://localhost:8000")
STREAMLIT_PORT = 8501

# ─── Derived paths (built from the above — do not override) ──────────────────
COLLECTION_RESOURCE = (
    f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/{COLLECTION_ID}"
)
GCS_DATA_URI = (
    f"gs://{BUCKET_NAME}/{INPUT_PREFIX}{GCS_FILENAME}"
)
