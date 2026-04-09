"""
Step 1 — Create the Vector Search 2.0 Collection
=================================================
Safe to run multiple times: if the collection already exists it prints
the existing resource name and exits.

Collection schema for Airbnb RAG:
  data_schema   → listing metadata: name, neighbourhood, property_type,
                  room_type, accommodates, bedrooms, price, rating,
                  instant_bookable, listing_url, host_name, text
  vector_schema → 'embedding' dense vector (768-dim, text-embedding-005)

Run:
    python scripts/01_setup_collection.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

from google.api_core.exceptions import AlreadyExists
from google.cloud import vectorsearch_v1beta as vs


def get_client() -> vs.VectorSearchServiceClient:
    return vs.VectorSearchServiceClient()


def setup_collection() -> str:
    """Create the VS2.0 collection if it doesn't exist. Returns resource name."""
    client   = get_client()
    parent   = f"projects/{config.PROJECT_ID}/locations/{config.LOCATION}"

    # ── data_schema: metadata fields stored alongside each vector ─────────────
    # These become searchable/returnable fields on every DataObject.
    # Uses JSON Schema types: string, number.
    # 'text' holds the concatenated embedding source text — no Firestore needed.
    data_schema = {
        "type": "object",
        "properties": {
            # Textual fields
            "name":              {"type": "string"},   # listing title
            "neighbourhood":     {"type": "string"},   # neighbourhood / zip
            "property_type":     {"type": "string"},   # e.g. "Entire guesthouse"
            "room_type":         {"type": "string"},   # "Entire home/apt", etc.
            "bathrooms_text":    {"type": "string"},   # e.g. "1 bath"
            "listing_url":       {"type": "string"},   # Airbnb URL
            "host_name":         {"type": "string"},
            "host_is_superhost": {"type": "string"},   # "true" / "false"
            "instant_bookable":  {"type": "string"},   # "true" / "false"
            "text":              {"type": "string"},   # embedding source text

            # Numeric fields (stored as numbers for easy filtering)
            "accommodates":      {"type": "number"},
            "bedrooms":          {"type": "number"},
            "beds":              {"type": "number"},
            "price":             {"type": "number"},   # USD per night (numeric)
            "rating":            {"type": "number"},   # review_scores_rating
            "latitude":          {"type": "number"},
            "longitude":         {"type": "number"},
        },
    }

    # ── vector_schema: which field holds the 768-dim embedding ────────────────
    vector_schema = {
        "embedding": {
            "dense_vector": {
                "dimensions": 768
            }
        }
    }

    collection = vs.Collection(
        display_name=f"Airbnb Listings — {config.COLLECTION_ID}",
        description=(
            "VS2.0 collection for Airbnb Agentic RAG. "
            "Stores Austin, TX Airbnb listings with semantic embeddings."
        ),
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
        result    = operation.result()
        print(f"  ✓  Collection created: {result.name}")
        return result.name

    except AlreadyExists:
        resource_name = f"{parent}/collections/{config.COLLECTION_ID}"
        print(f"  ✓  Collection already exists: {resource_name}")
        return resource_name


def main() -> None:
    print("=" * 60)
    print("Airbnb Agentic RAG — Collection Setup")
    print("=" * 60)
    print(f"  Project    : {config.PROJECT_ID}")
    print(f"  Location   : {config.LOCATION}")
    print(f"  Collection : {config.COLLECTION_ID}")
    print()

    name = setup_collection()

    print(f"\nCollection resource name:")
    print(f"  {name}")
    print(f"\nNext step:")
    print(f"  python scripts/02_ingest.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
