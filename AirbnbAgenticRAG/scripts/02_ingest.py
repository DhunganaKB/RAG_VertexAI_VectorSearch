"""
Step 2 — Ingest Airbnb Listings into the VS2.0 Collection
==========================================================
Reads listings.csv.gz from GCS, builds a rich text description for each
listing, embeds it with text-embedding-005, and stores the result as a
DataObject in the VS2.0 Collection.

Each DataObject contains:
  data    → listing metadata (name, neighbourhood, price, room_type, etc.)
  vectors → { embedding: [768 floats] }

No Firestore needed — VS2.0 stores everything together.
Safe to re-run: existing DataObjects are skipped (AlreadyExists handled).

The embedding text is built from:
  "{name}. {description}. Located in {neighbourhood}. {room_type}.
   Accommodates {accommodates} guests. {neighbourhood_overview}"

Run:
    python scripts/02_ingest.py
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import random
import sys
import time
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

from google.api_core.exceptions import AlreadyExists, ResourceExhausted
from google.cloud import storage
from google.cloud import vectorsearch_v1beta as vs
from tqdm import tqdm
from vertexai import init as vertex_init
from vertexai.language_models import TextEmbeddingModel


# ─── Price parser ─────────────────────────────────────────────────────────────

def parse_price(price_str: str) -> float | None:
    """Convert '$97.00' → 97.0, returns None if unparseable."""
    if not price_str:
        return None
    cleaned = price_str.replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_float(value: str) -> float | None:
    """Parse a float string, return None if empty or invalid."""
    if not value or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_bool_str(value: str) -> str:
    """Convert 't'/'f' → 'true'/'false'."""
    return "true" if value.strip().lower() in {"t", "true", "1", "yes"} else "false"


# ─── Text builder ─────────────────────────────────────────────────────────────

def build_embedding_text(row: dict) -> str:
    """
    Build rich text for embedding from listing fields.
    Combines the most semantically meaningful text fields.
    """
    parts = []

    if row.get("name"):
        parts.append(row["name"].strip())

    if row.get("description"):
        desc = row["description"].strip()
        if desc:
            parts.append(desc[:600])          # cap at 600 chars

    neighbourhood = row.get("neighbourhood_cleansed") or row.get("neighbourhood", "")
    if neighbourhood.strip():
        parts.append(f"Located in {neighbourhood.strip()}.")

    room_type = row.get("room_type", "")
    accommodates = row.get("accommodates", "")
    if room_type or accommodates:
        detail = f"{room_type}" if room_type else ""
        if accommodates:
            detail += f". Accommodates {accommodates} guests"
        parts.append(detail.strip(". ") + ".")

    if row.get("neighborhood_overview"):
        overview = row["neighborhood_overview"].strip()
        if overview:
            parts.append(overview[:400])      # cap at 400 chars

    bedrooms = row.get("bedrooms", "")
    beds     = row.get("beds", "")
    bathrooms = row.get("bathrooms_text", "")
    amenity_str = row.get("amenities", "")
    if bedrooms or beds or bathrooms:
        props = []
        if bedrooms:
            props.append(f"{bedrooms} bedroom(s)")
        if beds:
            props.append(f"{beds} bed(s)")
        if bathrooms:
            props.append(bathrooms)
        parts.append(", ".join(props) + ".")

    return " ".join(parts)


# ─── DataObject ID ─────────────────────────────────────────────────────────────

def make_listing_id(listing_id: str) -> str:
    """Deterministic SHA1 from the Airbnb listing_id."""
    return hashlib.sha1(f"airbnb:{listing_id}".encode()).hexdigest()


# ─── Batching ─────────────────────────────────────────────────────────────────

def batched(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ─── Embedding with retry + exponential backoff ────────────────────────────────

def embed_with_retry(
    model: TextEmbeddingModel,
    texts: list[str],
    max_retries: int = 6,
    base_delay: float = 5.0,
) -> list:
    """
    Call model.get_embeddings() with exponential backoff on 429 / ResourceExhausted.

    Delay schedule (with ±30 % jitter):
        attempt 1 →  ~5 s
        attempt 2 → ~10 s
        attempt 3 → ~20 s
        attempt 4 → ~40 s
        attempt 5 → ~80 s
        attempt 6 → ~160 s  (then raises)
    """
    for attempt in range(max_retries):
        try:
            return model.get_embeddings(texts)
        except ResourceExhausted as exc:
            if attempt == max_retries - 1:
                raise  # give up after final attempt
            wait = base_delay * (2 ** attempt) * (1.0 + random.uniform(-0.3, 0.3))
            tqdm.write(
                f"\n  ⚠  429 Resource Exhausted — waiting {wait:.1f}s "
                f"(attempt {attempt + 1}/{max_retries})..."
            )
            time.sleep(wait)
    # unreachable, but satisfies type checkers
    raise RuntimeError("embed_with_retry: exceeded max retries")


# ─── VS2.0 DataObject helpers ──────────────────────────────────────────────────

def build_data_object(obj_id: str, metadata: dict, embedding: list[float]) -> vs.DataObject:
    """Build a VS2.0 DataObject with listing metadata + vector."""
    return vs.DataObject(
        data_object_id=obj_id,
        data=metadata,
        vectors={
            "embedding": vs.Vector(
                dense=vs.DenseVector(values=embedding)
            )
        },
    )


def upsert_batch(
    client: vs.DataObjectServiceClient,
    parent: str,
    objects: list[vs.DataObject],
) -> int:
    """Batch-upsert DataObjects; handle AlreadyExists gracefully."""
    requests = [
        vs.CreateDataObjectRequest(
            parent=parent,
            data_object_id=obj.data_object_id,
            data_object=obj,
        )
        for obj in objects
    ]
    try:
        client.batch_create_data_objects(
            request=vs.BatchCreateDataObjectsRequest(
                parent=parent,
                requests=requests,
            )
        )
        return len(objects)
    except AlreadyExists:
        # At least one exists — fall back to one-by-one
        created = 0
        for req in requests:
            try:
                client.create_data_object(request=req)
                created += 1
            except AlreadyExists:
                pass
        return created


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Airbnb Agentic RAG — Ingestion Pipeline")
    print("=" * 60)
    vertex_init(project=config.PROJECT_ID, location=config.LOCATION)

    # ── Step 1: Download CSV from GCS ─────────────────────────────────────────
    gcs_path = f"{config.INPUT_PREFIX}{config.GCS_FILENAME}"
    print(f"\n[1/4] Reading gs://{config.BUCKET_NAME}/{gcs_path}")

    storage_client = storage.Client(project=config.PROJECT_ID)
    bucket = storage_client.bucket(config.BUCKET_NAME)
    blob   = bucket.blob(gcs_path)
    if not blob.exists():
        print(f"  ✗  File not found. Run scripts/00_upload_data.py first.")
        sys.exit(1)

    raw_bytes = blob.download_as_bytes()
    print(f"  ✓  Downloaded ({len(raw_bytes) / 1_048_576:.1f} MB)")

    # ── Step 2: Parse CSV and build listing records ───────────────────────────
    limit = config.MAX_LISTINGS if config.MAX_LISTINGS > 0 else None
    print(f"\n[2/4] Parsing CSV (limit: {limit or 'all'} listings)...")

    records: list[dict] = []
    with gzip.open(io.BytesIO(raw_bytes), "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break

            text = build_embedding_text(row)
            if not text.strip():
                continue

            price = parse_price(row.get("price", ""))
            records.append({
                "obj_id": make_listing_id(row["id"]),
                "embedding_text": text,
                "metadata": {
                    # ── Textual ──────────────────────────────────────────────
                    "name":              (row.get("name") or "")[:500],
                    "neighbourhood":     row.get("neighbourhood_cleansed") or "",
                    "property_type":     row.get("property_type") or "",
                    "room_type":         row.get("room_type") or "",
                    "bathrooms_text":    row.get("bathrooms_text") or "",
                    "listing_url":       row.get("listing_url") or "",
                    "host_name":         row.get("host_name") or "",
                    "host_is_superhost": parse_bool_str(row.get("host_is_superhost", "")),
                    "instant_bookable":  parse_bool_str(row.get("instant_bookable", "")),
                    "text":              text[:2000],    # stored for retrieval

                    # ── Numeric ───────────────────────────────────────────────
                    "accommodates": parse_float(row.get("accommodates", "")) or 0.0,
                    "bedrooms":     parse_float(row.get("bedrooms", "")) or 0.0,
                    "beds":         parse_float(row.get("beds", "")) or 0.0,
                    "price":        price if price is not None else 0.0,
                    "rating":       parse_float(row.get("review_scores_rating", "")) or 0.0,
                    "latitude":     parse_float(row.get("latitude", "")) or 0.0,
                    "longitude":    parse_float(row.get("longitude", "")) or 0.0,
                },
            })

    print(f"  ✓  Parsed {len(records)} listings")
    if not records:
        print("  ✗  No listings found — check your CSV file.")
        sys.exit(1)

    # ── Step 3: Embed ─────────────────────────────────────────────────────────
    print(f"\n[3/4] Embedding with {config.EMBEDDING_MODEL} "
          f"(batch_size={config.BATCH_SIZE})...")
    embedding_model = TextEmbeddingModel.from_pretrained(config.EMBEDDING_MODEL)
    data_objects: list[vs.DataObject] = []

    # Small pause between batches keeps us well under the QPM quota.
    INTER_BATCH_SLEEP = 0.5   # seconds

    with tqdm(total=len(records), unit="listing") as pbar:
        for batch in batched(records, config.BATCH_SIZE):
            texts      = [r["embedding_text"] for r in batch]
            embeddings = embed_with_retry(embedding_model, texts)
            for rec, emb in zip(batch, embeddings):
                data_objects.append(
                    build_data_object(
                        obj_id=rec["obj_id"],
                        metadata=rec["metadata"],
                        embedding=list(emb.values),
                    )
                )
            pbar.update(len(batch))
            time.sleep(INTER_BATCH_SLEEP)

    print(f"  ✓  Generated {len(data_objects)} embeddings")

    # ── Step 4: Upsert DataObjects into the Collection ────────────────────────
    print(f"\n[4/4] Upserting {len(data_objects)} DataObjects "
          f"into '{config.COLLECTION_ID}'...")
    do_client  = vs.DataObjectServiceClient()
    parent     = config.COLLECTION_RESOURCE
    total_done = 0

    with tqdm(total=len(data_objects), unit="obj") as pbar:
        for batch in batched(data_objects, config.BATCH_SIZE):
            upsert_batch(do_client, parent, batch)
            total_done += len(batch)
            pbar.update(len(batch))

    print(f"\n{'=' * 60}")
    print("Ingestion complete!")
    print(f"  Collection : {config.COLLECTION_RESOURCE}")
    print(f"  Listings   : {len(records)}")
    print(f"  DataObjects: {len(data_objects)}")
    print(f"\nNext steps:")
    print(f"  (Optional)  python scripts/03_create_index.py  # ScaNN ANN index")
    print(f"  python scripts/inspect_collection.py           # verify data")
    print(f"  uvicorn app.main:app --reload --port 8000      # start backend")
    print(f"  streamlit run ui/streamlit_app.py              # start frontend")
    print("=" * 60)


if __name__ == "__main__":
    main()
