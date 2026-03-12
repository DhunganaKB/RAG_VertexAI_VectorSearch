"""
RAG with Vector Search 2.0 — Project Configuration
====================================================
Single source of truth for all scripts and the app.
"""
from __future__ import annotations

# ─── GCP ─────────────────────────────────────────────────────────────────────
PROJECT_ID = "deve-0000"
LOCATION   = "us-central1"

# ─── Vector Search 2.0 Collection ────────────────────────────────────────────
COLLECTION_ID = "collection-for-demo"

# ─── GCS ─────────────────────────────────────────────────────────────────────
BUCKET_NAME  = "bucket-kamal-0000"
INPUT_PREFIX = "raw/"               # gs://bucket-kamal-987/raw/

# ─── AI Models ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-005"   # 768-dim
GEMINI_MODEL    = "gemini-2.0-flash-001"

# ─── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 200
BATCH_SIZE    = 50    # DataObjects per batch_create call

# ─── RAG ──────────────────────────────────────────────────────────────────────
TOP_K             = 5
MAX_OUTPUT_TOKENS = 1024
TEMPERATURE       = 0.2

# ─── Serving ──────────────────────────────────────────────────────────────────
API_BASE_URL   = "http://localhost:8000"
STREAMLIT_PORT = 8501

# ─── Derived paths ────────────────────────────────────────────────────────────
COLLECTION_RESOURCE = (
    f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/{COLLECTION_ID}"
)
