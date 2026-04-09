"""
Step 0 — Upload Airbnb Data to GCS
===================================
Idempotent: checks bucket and file existence before creating or uploading.

Steps performed:
  1. Check if GCS bucket exists → create it in us-central1 if not
  2. Check if listings.csv.gz is already in the bucket → skip upload if present
  3. Upload the local file to gs://{BUCKET_NAME}/{INPUT_PREFIX}{GCS_FILENAME}

The local source file is expected at:
  ../RAGVectoSearch20/inputdata/listings.csv.gz
  (relative to this project's root directory)

Run:
    python scripts/00_upload_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

from google.api_core.exceptions import Conflict
from google.cloud import storage


# ── Resolve local source path ─────────────────────────────────────────────────
# AirbnbAgenticRAG/scripts/ → AirbnbAgenticRAG/ → RAG-VertexAI/
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_FILE   = _PROJECT_ROOT.parent / "RAGVectoSearch20" / "inputdata" / "listings.csv.gz"

GCS_BLOB_NAME = f"{config.INPUT_PREFIX}{config.GCS_FILENAME}"


def get_client() -> storage.Client:
    return storage.Client(project=config.PROJECT_ID)


# ── Step 1: Bucket ────────────────────────────────────────────────────────────

def ensure_bucket(client: storage.Client) -> storage.Bucket:
    """Return the bucket, creating it in us-central1 if it doesn't exist."""
    print(f"\n[1/3] Checking bucket: gs://{config.BUCKET_NAME}")

    bucket = client.bucket(config.BUCKET_NAME)
    if bucket.exists():
        print(f"  ✓  Bucket already exists — skipping creation.")
        return bucket

    # Create bucket in the same region as the VS2.0 collection
    print(f"  ✗  Bucket not found. Creating in {config.LOCATION}...")
    new_bucket = client.bucket(config.BUCKET_NAME)
    new_bucket.storage_class = "STANDARD"
    try:
        created = client.create_bucket(new_bucket, location=config.LOCATION)
        print(f"  ✓  Bucket created: gs://{created.name}")
        return created
    except Conflict:
        # Race condition: another process created it
        print(f"  ✓  Bucket already exists (race condition handled).")
        return client.bucket(config.BUCKET_NAME)


# ── Step 2 + 3: Upload ────────────────────────────────────────────────────────

def ensure_file_uploaded(client: storage.Client, bucket: storage.Bucket) -> None:
    """Upload listings.csv.gz to the bucket if not already present."""
    print(f"\n[2/3] Checking file: gs://{config.BUCKET_NAME}/{GCS_BLOB_NAME}")

    blob = bucket.blob(GCS_BLOB_NAME)
    if blob.exists():
        size_mb = (blob.size or 0) / 1_048_576
        print(f"  ✓  File already present ({size_mb:.1f} MB) — skipping upload.")
        return

    # Verify local source exists
    print(f"\n[3/3] Uploading from local path:")
    print(f"      {_LOCAL_FILE}")
    if not _LOCAL_FILE.exists():
        print(f"\n  ✗  ERROR: Local file not found at:")
        print(f"       {_LOCAL_FILE}")
        print(f"\n  Expected location:")
        print(f"       RAGVectoSearch20/inputdata/listings.csv.gz")
        print(f"\n  Make sure you have the Airbnb listings data downloaded.")
        sys.exit(1)

    size_mb = _LOCAL_FILE.stat().st_size / 1_048_576
    print(f"  Source size : {size_mb:.1f} MB")
    print(f"  Uploading   → gs://{config.BUCKET_NAME}/{GCS_BLOB_NAME}")

    blob.upload_from_filename(str(_LOCAL_FILE), content_type="application/gzip")

    # Verify upload
    blob.reload()
    uploaded_mb = (blob.size or 0) / 1_048_576
    print(f"  ✓  Upload complete ({uploaded_mb:.1f} MB)")


def main() -> None:
    print("=" * 60)
    print("Airbnb Agentic RAG — GCS Data Upload")
    print("=" * 60)
    print(f"  Project  : {config.PROJECT_ID}")
    print(f"  Bucket   : {config.BUCKET_NAME}")
    print(f"  Location : {config.LOCATION}")
    print(f"  GCS file : {GCS_BLOB_NAME}")

    client = get_client()
    bucket = ensure_bucket(client)
    ensure_file_uploaded(client, bucket)

    print(f"\n{'=' * 60}")
    print("Data ready in GCS!")
    print(f"  gs://{config.BUCKET_NAME}/{GCS_BLOB_NAME}")
    print(f"\nNext step:")
    print(f"  python scripts/01_setup_collection.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
