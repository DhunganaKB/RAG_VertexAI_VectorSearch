"""
Step 3 (Optional) — Create a Collection Index for ANN Search
=============================================================
Creates a ScaNN-powered ANN (Approximate Nearest Neighbor) index on the
'embedding' field of the collection.

Without this index  → VS2.0 uses exact kNN (full scan, perfect recall).
With this index     → VS2.0 uses ScaNN ANN (~99% recall, sub-10ms at scale).

The SearchDataObjects API call in rag.py does NOT change — VS2.0 routes
to ANN automatically once the index exists.

When to run:
  - Skip for small collections (< ~100k DataObjects): kNN is fast enough.
  - Run when scaling to hundreds of thousands or millions of DataObjects.

Run:
    python scripts/03_create_index.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

from google.api_core.exceptions import AlreadyExists
from google.cloud import vectorsearch_v1beta as vs

INDEX_ID           = "embedding-ann-index"
INDEX_DISPLAY_NAME = "ScaNN ANN index — embedding field"
INDEX_DESCRIPTION  = (
    "Approximate Nearest Neighbor index using Google ScaNN. "
    "Enables sub-10ms vector search at billion scale. "
    "Distance metric: DOT_PRODUCT (equivalent to cosine for unit-norm vectors). "
    "Feature normalisation: UNIT_L2_NORM."
)


def get_client() -> vs.VectorSearchServiceClient:
    return vs.VectorSearchServiceClient()


def check_existing_index(
    client: vs.VectorSearchServiceClient,
) -> vs.Index | None:
    """Return the index if one already exists on the collection, else None."""
    indexes = list(
        client.list_indexes(
            request=vs.ListIndexesRequest(parent=config.COLLECTION_RESOURCE)
        )
    )
    return indexes[0] if indexes else None


def print_index_info(index: vs.Index) -> None:
    idx_id = index.name.split("/")[-1]
    print(f"  Name          : {index.name}")
    print(f"  ID            : {idx_id}")
    print(f"  Display name  : {index.display_name}")
    print(f"  Vector field  : {index.index_field}")
    print(f"  Distance      : {'DOT_PRODUCT' if index.distance_metric == 1 else index.distance_metric}")
    print(f"  Norm type     : {'UNIT_L2_NORM' if index.dense_scann.feature_norm_type == 2 else index.dense_scann.feature_norm_type}")
    print(f"  Created       : {index.create_time}")


def create_index(client: vs.VectorSearchServiceClient) -> vs.Index:
    """Create the ANN Collection Index and wait for it to become active."""

    # ── Index configuration ───────────────────────────────────────────────────
    #   index_field     → must match the key in vector_schema ("embedding")
    #   distance_metric → DOT_PRODUCT: for unit-normalised embeddings this is
    #                     numerically identical to cosine similarity but faster
    #   dense_scann     → enables the ScaNN engine
    #   feature_norm_type → UNIT_L2_NORM (2): normalises each vector to unit
    #                       length before indexing; required for DOT_PRODUCT to
    #                       behave as cosine similarity
    index = vs.Index(
        display_name=INDEX_DISPLAY_NAME,
        description=INDEX_DESCRIPTION,
        index_field="embedding",
        distance_metric=vs.DistanceMetric.DOT_PRODUCT,
        dense_scann=vs.DenseScannIndex(
            feature_norm_type=2,  # UNIT_L2_NORM
        ),
    )

    request = vs.CreateIndexRequest(
        parent=config.COLLECTION_RESOURCE,
        index_id=INDEX_ID,
        index=index,
    )

    print(f"  Submitting CreateIndex request for '{INDEX_ID}'...")
    print(f"  Vector field  : embedding")
    print(f"  Algorithm     : ScaNN (Dense ANN)")
    print(f"  Distance      : DOT_PRODUCT  (= cosine for unit-norm vectors)")
    print(f"  Norm type     : UNIT_L2_NORM")
    print()

    try:
        operation = client.create_index(request=request)
    except AlreadyExists:
        print("  Index already exists — fetching details...")
        return check_existing_index(client)

    print("  Index build started (Long Running Operation).")
    print(f"  Operation : {operation.operation.name}")
    print()
    print("  Polling every 30s — this can take 5–20 minutes for large")
    print("  collections. Safe to Ctrl+C; the build continues server-side.")
    print()

    start = time.time()
    poll_interval = 30          # seconds between status checks
    max_wait      = 7200        # 2-hour hard ceiling

    while True:
        elapsed = time.time() - start

        if operation.done():
            print(f"\r  ✓  Build complete in {elapsed:.0f}s" + " " * 20)
            return operation.result()

        if elapsed >= max_wait:
            # Operation is still running server-side; just tell the user.
            print()
            print("  ⚠  Client-side wait limit reached.")
            print("  The index build is still running on Google Cloud.")
            print("  Re-run this script to check completion:")
            print("    python scripts/03_create_index.py")
            print()
            # Return whatever list_indexes gives us (may still be pending)
            existing = check_existing_index(client)
            return existing

        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        print(f"\r  ⏳  Waiting... {mins:02d}:{secs:02d} elapsed", end="", flush=True)
        time.sleep(poll_interval)


def main() -> None:
    client = get_client()

    print(f"\n{'═' * 60}")
    print(f"  Collection : {config.COLLECTION_ID}")
    print(f"  Project    : {config.PROJECT_ID}")
    print(f"  Location   : {config.LOCATION}")
    print(f"{'═' * 60}\n")

    # ── Check if index already exists ────────────────────────────────────────
    existing = check_existing_index(client)

    if existing:
        idx_id = existing.name.split("/")[-1]
        print(f"✓  ANN index already exists: '{idx_id}'")
        print()
        print_index_info(existing)
        print()
        print("  Search mode : ANN (ScaNN)  ← ~99% recall, sub-10ms at scale")
        print("  No action needed.\n")
        return

    # ── No index found — create one ───────────────────────────────────────────
    print("  No ANN index found on this collection.")
    print(f"  Current search mode : exact kNN (full scan, 100% recall)")
    print()
    print(f"  Creating ScaNN ANN index '{INDEX_ID}'...")
    print()

    new_index = create_index(client)

    print()
    print("✓  Index created successfully!")
    print()
    print_index_info(new_index)
    print()
    print("  Search mode is now : ANN (ScaNN)")
    print("  No changes needed in rag.py — VS2.0 routes automatically.")
    print()
    print("  Restart the FastAPI server to see the mode reflected in logs:")
    print("    uvicorn app.main:app --reload --port 8000\n")


if __name__ == "__main__":
    main()
