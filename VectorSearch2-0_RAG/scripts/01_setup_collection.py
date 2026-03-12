"""
Step 1 — Create the Vector Search 2.0 Collection
=================================================
Safe to run multiple times: if the collection already exists it just prints
the existing resource name and exits.

Collection schema for RAG:
  data_schema   → non-vector metadata: title, source, text (the chunk)
  vector_schema → 'embedding' dense vector (768-dim, text-embedding-005)

Run:
    python scripts/01_setup_collection.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.api_core.exceptions import AlreadyExists
from google.cloud import vectorsearch_v1beta as vs

import config


def get_client() -> vs.VectorSearchServiceClient:
    return vs.VectorSearchServiceClient()


def setup_collection() -> str:
    """Create collection if it doesn't exist. Returns the resource name."""
    client = get_client()
    parent = f"projects/{config.PROJECT_ID}/locations/{config.LOCATION}"

    # ── data_schema: metadata fields stored alongside each vector ────────────
    # These become searchable/returnable fields on every DataObject.
    # 'text' holds the raw chunk text — returned at query time (no Firestore needed).
    data_schema = {
        "type": "object",
        "properties": {
            "title":  {"type": "string"},   # filename
            "source": {"type": "string"},   # gs:// URI
            "text":   {"type": "string"},   # the actual chunk text
        },
    }

    # ── vector_schema: which field holds the embedding ───────────────────────
    # Key   = field name used when inserting DataObjects and querying
    # Value = dense vector with 768 dimensions (matches text-embedding-005)
    vector_schema = {
        "embedding": {
            "dense_vector": {
                "dimensions": 768
            }
        }
    }

    collection = vs.Collection(
        display_name=f"RAG Demo — {config.COLLECTION_ID}",
        description="Vector Search 2.0 collection for RAG demo (PDFs from GCS)",
        data_schema=data_schema,
        vector_schema=vector_schema,
    )

    request = vs.CreateCollectionRequest(
        parent=parent,
        collection_id=config.COLLECTION_ID,
        collection=collection,
    )

    try:
        print(f"Creating collection '{config.COLLECTION_ID}'...")
        operation = client.create_collection(request=request)
        result = operation.result()
        print(f"  Collection created: {result.name}")
        return result.name

    except AlreadyExists:
        resource_name = f"{parent}/collections/{config.COLLECTION_ID}"
        print(f"  Collection already exists: {resource_name}")
        return resource_name


if __name__ == "__main__":
    name = setup_collection()
    print(f"\nReady. Collection resource name:\n  {name}")
