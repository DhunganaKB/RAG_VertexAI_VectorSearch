from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


# =============================================================================
# ✏️  EDIT THIS SECTION — set your real values before running anything
# =============================================================================

# --- GCP Project ---
PROJECT_ID = "deve-0000"
REGION = "us-central1"             # e.g. us-central1, europe-west1

# --- GCS Bucket (stores raw documents + metadata backup) ---
BUCKET_NAME = "bucket-kamal-000"   # must be globally unique
SERVICE_ACCOUNT_NAME = "rag-vector-search-sa"

# --- AI Models ---
GEMINI_MODEL = "gemini-2.5-flash"       # Generation model
EMBEDDING_MODEL = "text-embedding-005"  # 768-dim embedding model

# --- Vertex AI Vector Search ---
VECTOR_INDEX_DISPLAY_NAME = "rag-index-v"
VECTOR_ENDPOINT_DISPLAY_NAME = "rag-index-endpoint-v"
VECTOR_DEPLOYED_INDEX_ID = "rag_deployed_index_v"
VECTOR_DIMENSIONS = 768            # Must match EMBEDDING_MODEL output dimensions
VECTOR_MACHINE_TYPE = "e2-standard-16"

# --- GCS paths ---
INPUT_PREFIX = "raw/"                       # GCS prefix where raw docs are uploaded
METADATA_LOCAL_PATH = "artifacts/chunks.jsonl"    # Local backup artifact
METADATA_GCS_PATH = "metadata/chunks.jsonl"       # GCS backup artifact

# --- Firestore (scalable metadata store) ---
# One Firestore document per chunk (keyed by chunk_id SHA1).
# The serving layer calls db.get_all(refs) to fetch only top-K docs per query,
# never loading the full corpus into memory.
FIRESTORE_DATABASE = "(default)"   # Firestore database ID — "(default)" for most projects
FIRESTORE_COLLECTION = "rag_chunks"  # Collection name for chunk metadata

# --- RAG parameters ---
TOP_K = 5                # Number of chunks to retrieve from Vector Search
MAX_OUTPUT_TOKENS = 1024
TEMPERATURE = 0.2

# --- Serving / UI ---
# FastAPI backend URL — used by the Streamlit UI to call the RAG endpoint
API_BASE_URL = "http://localhost:8000"
STREAMLIT_PORT = 8501


# =============================================================================
# Runtime — do not edit below this line
# =============================================================================

_RUNTIME_PATH = Path("artifacts/vector_resources.json")


@dataclass(frozen=True)
class ProjectSettings:
    # GCP
    project_id: str
    region: str
    bucket_name: str
    service_account_name: str
    # Models
    gemini_model: str
    embedding_model: str
    # Vector Search
    vector_index_display_name: str
    vector_endpoint_display_name: str
    vector_deployed_index_id: str
    vector_dimensions: int
    vector_machine_type: str
    # GCS paths
    input_prefix: str
    metadata_local_path: str
    metadata_gcs_path: str
    # Firestore
    firestore_database: str
    firestore_collection: str
    # RAG
    top_k: int
    max_output_tokens: int
    temperature: float
    # Serving / UI
    api_base_url: str
    streamlit_port: int
    # Runtime (populated from artifacts/vector_resources.json)
    index_resource_name: str | None
    index_endpoint_resource_name: str | None


def _read_runtime_values() -> tuple[str | None, str | None, str | None]:
    """Read index/endpoint/deployed-id written by create_vector_search.py."""
    if not _RUNTIME_PATH.exists():
        return None, None, None
    payload = json.loads(_RUNTIME_PATH.read_text(encoding="utf-8"))
    return (
        payload.get("index_resource_name"),
        payload.get("index_endpoint_resource_name"),
        payload.get("deployed_index_id"),
    )


def _validate_editable_values() -> None:
    placeholder_values = {PROJECT_ID, BUCKET_NAME}
    if "your-gcp-project-id" in placeholder_values or "your-rag-docs-bucket" in placeholder_values:
        raise RuntimeError(
            "Edit config.py first: set PROJECT_ID and BUCKET_NAME to real values."
        )


def load_project_settings() -> ProjectSettings:
    _validate_editable_values()
    index_resource_name, endpoint_resource_name, deployed_index_id = _read_runtime_values()
    return ProjectSettings(
        project_id=PROJECT_ID,
        region=REGION,
        bucket_name=BUCKET_NAME,
        service_account_name=SERVICE_ACCOUNT_NAME,
        gemini_model=GEMINI_MODEL,
        embedding_model=EMBEDDING_MODEL,
        vector_index_display_name=VECTOR_INDEX_DISPLAY_NAME,
        vector_endpoint_display_name=VECTOR_ENDPOINT_DISPLAY_NAME,
        vector_deployed_index_id=deployed_index_id or VECTOR_DEPLOYED_INDEX_ID,
        vector_dimensions=VECTOR_DIMENSIONS,
        vector_machine_type=VECTOR_MACHINE_TYPE,
        input_prefix=INPUT_PREFIX,
        metadata_local_path=METADATA_LOCAL_PATH,
        metadata_gcs_path=METADATA_GCS_PATH,
        firestore_database=FIRESTORE_DATABASE,
        firestore_collection=FIRESTORE_COLLECTION,
        top_k=TOP_K,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
        api_base_url=API_BASE_URL,
        streamlit_port=STREAMLIT_PORT,
        index_resource_name=index_resource_name,
        index_endpoint_resource_name=endpoint_resource_name,
    )


def save_runtime_resources_with_deployed_id(
    index_resource_name: str,
    index_endpoint_resource_name: str,
    deployed_index_id: str,
) -> None:
    """Persist Vertex AI resource names so the app and ingestion can load them."""
    _RUNTIME_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RUNTIME_PATH.write_text(
        json.dumps(
            {
                "index_resource_name": index_resource_name,
                "index_endpoint_resource_name": index_endpoint_resource_name,
                "deployed_index_id": deployed_index_id,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


# Kept for backwards-compatibility with callers that used the two-argument form.
def save_runtime_resources(index_resource_name: str, index_endpoint_resource_name: str) -> None:
    save_runtime_resources_with_deployed_id(
        index_resource_name=index_resource_name,
        index_endpoint_resource_name=index_endpoint_resource_name,
        deployed_index_id="",
    )
