"""
Vector Search 2.0 - Collection Basics
======================================
Following the official docs exactly:
https://docs.cloud.google.com/vertex-ai/docs/vector-search-2/collections/collections#python

This script shows how to:
  1. Create a Collection  (with data_schema + vector_schema)
  2. Get a Collection
  3. List all Collections
  4. Delete a Collection

Prerequisites:
  pip install google-cloud-vectorsearch

Enable APIs (REQUIRED — run BOTH once in terminal before running this script):
  gcloud services enable vectorsearch.googleapis.com --project "your-project-id"
  gcloud services enable aiplatform.googleapis.com  --project "your-project-id"

  Or enable both together:
  gcloud services enable vectorsearch.googleapis.com aiplatform.googleapis.com \
    --project "your-project-id"

  Verify they are enabled:
  gcloud services list --enabled --project "your-project-id" \
    --filter="name:vectorsearch OR name:aiplatform"

Auth:
  gcloud auth application-default login
"""

from google.cloud import vectorsearch_v1beta
from google.api_core.exceptions import AlreadyExists

# ─── CONFIG ──────────────────────────────────────────────────────────────────
PROJECT_ID    = "deve-000"
LOCATION      = "us-central1"
COLLECTION_ID = "collection-for-demo"
# ─────────────────────────────────────────────────────────────────────────────


def get_client():
    """
    Create the Vector Search 2.0 client.
    NOTE: No client_options / api_endpoint needed — the SDK resolves it automatically.
    """
    return vectorsearch_v1beta.VectorSearchServiceClient()


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA CONCEPTS
# ═══════════════════════════════════════════════════════════════════════════════
#
# A Collection requires TWO schemas:
#
#  1. data_schema  — JSON Schema defining the NON-vector fields (your metadata).
#                    These are the text/number fields you store alongside vectors.
#                    e.g. title, genre, year, director
#                    Follows standard JSON Schema: { "type": "object", "properties": {...} }
#
#  2. vector_schema — A plain dict mapping field_name → vector config.
#                     Each entry defines ONE vector field (dense or sparse).
#                     e.g. { "my_embedding": { "dense_vector": { "dimensions": 768 } } }
#
# IMPORTANT: Both schemas must be plain Python dicts — NOT proto objects.
# ═══════════════════════════════════════════════════════════════════════════════


def create_collection():
    """Create a Vector Search 2.0 Collection with data + vector schemas."""

    client = get_client()

    # ── 1. data_schema ────────────────────────────────────────────────────────
    # Defines the structure of the non-vector (metadata) fields.
    # Uses standard JSON Schema format.
    # These are the fields you can filter/search on alongside your vectors.
    #
    # Supported types: "string", "number", "integer", "boolean"
    # NOTE: "additionalProperties" is NOT supported — all fields must be explicit.
    data_schema = {
        "type": "object",
        "properties": {
            "title":    {"type": "string"},   # e.g. movie or document title
            "genre":    {"type": "string"},   # e.g. "action", "drama"
            "year":     {"type": "number"},   # e.g. 2024
            "director": {"type": "string"},   # e.g. "Christopher Nolan"
        },
    }

    # ── 2. vector_schema ──────────────────────────────────────────────────────
    # Defines which fields hold vector embeddings.
    # Format: { "field_name": { "dense_vector": { "dimensions": N } } }
    #
    # You can have MULTIPLE vector fields (multi-modal use case):
    #   - dense_vector:  standard embedding (e.g. from text-embedding-004)
    #   - sparse_vector: keyword/BM25-style sparse vectors (for hybrid search)
    #
    # The "dimensions" must match your embedding model output:
    #   text-embedding-004              → 768
    #   text-multilingual-embedding-002 → 768
    #   text-embedding-large-exp-03-07  → 3072
    vector_schema = {
        "embedding": {
            "dense_vector": {
                "dimensions": 768    # matches text-embedding-004
            }
        },
    }

    # ── 3. Build the Collection object ────────────────────────────────────────
    collection = vectorsearch_v1beta.Collection(
        data_schema=data_schema,
        vector_schema=vector_schema,
    )

    # ── 4. Build the CreateCollectionRequest ──────────────────────────────────
    request = vectorsearch_v1beta.CreateCollectionRequest(
        parent=f"projects/{PROJECT_ID}/locations/{LOCATION}",
        collection_id=COLLECTION_ID,
        collection=collection,
    )

    print(f"Creating collection '{COLLECTION_ID}' in {LOCATION}...")
    print("This is a long-running operation (can take several minutes)...")

    try:
        operation = client.create_collection(request=request)
        result = operation.result()   # blocks until done

        print(f"\nCollection created successfully!")
        print(f"  Resource Name: {result.name}")
        print(f"  Display Name:  {result.display_name}")
        print(f"  Created at:    {result.create_time}")
        return result

    except AlreadyExists:
        # Collection already exists — fetch and return it instead
        print(f"\nCollection '{COLLECTION_ID}' already exists. Fetching it...")
        return get_collection()


def get_collection():
    """Get a reference to an existing Collection."""

    client = get_client()

    request = vectorsearch_v1beta.GetCollectionRequest(
        name=f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/{COLLECTION_ID}",
    )

    response = client.get_collection(request=request)
    print(f"\nCollection: {response}")
    return response


def list_collections():
    """List all Collections in this project + location."""

    client = get_client()

    request = vectorsearch_v1beta.ListCollectionsRequest(
        parent=f"projects/{PROJECT_ID}/locations/{LOCATION}",
    )

    page_result = client.list_collections(request=request)

    print("\nAll Collections:")
    for col in page_result:
        print(f"  - {col.name}")


def delete_collection():
    """Delete a Collection and all its data (irreversible)."""

    client = get_client()

    request = vectorsearch_v1beta.DeleteCollectionRequest(
        name=f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/{COLLECTION_ID}",
    )

    print(f"Deleting collection '{COLLECTION_ID}'...")
    operation = client.delete_collection(request=request)
    operation.result()
    print("Collection deleted.")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Step 1: Create
    #create_collection()

    # Step 2: List to confirm
    list_collections()

    # Step 3: Inspect
    get_collection()

    # Step 4 (uncomment to clean up):
    # delete_collection()
