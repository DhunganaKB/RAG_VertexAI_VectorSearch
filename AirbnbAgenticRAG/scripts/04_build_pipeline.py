"""
04_build_pipeline.py — Unified Ingestion Pipeline (single run)
==============================================================
Combines all three offline steps into one sequential script:

  Stage 1: Create VS2.0 Collection  (idempotent — skips if exists)
  Stage 2: Ingest DataObjects        (idempotent — skips duplicates)
  Stage 3: Create ScaNN ANN Index    (idempotent — skips if exists)

Usage:
    # Full pipeline (recommended first run)
    python scripts/04_build_pipeline.py

    # Resume — skip stages already done
    python scripts/04_build_pipeline.py --skip-collection
    python scripts/04_build_pipeline.py --skip-collection --skip-ingest
    python scripts/04_build_pipeline.py --skip-index           # no index (kNN mode)

    # Dry-run — print what would happen, touch nothing
    python scripts/04_build_pipeline.py --dry-run

Options:
    --skip-collection   Skip collection creation (must already exist)
    --skip-ingest       Skip data ingestion
    --skip-index        Do NOT create the ScaNN ANN index (uses exact kNN)
    --dry-run           Print plan only; make no API calls
    --max-listings N    Override config.MAX_LISTINGS for this run
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── sys.path bootstrap — works in all three execution contexts ────────────
#
# Context A  – Local  : `python scripts/04_build_pipeline.py`
#              __file__ is always defined → Path(__file__) resolves correctly.
#
# Context B  – Databricks Python file task (classic / job cluster)
#              Databricks injects __file__ = workspace path → same as A.
#
# Context C  – Databricks *serverless* Python task
#              The runtime uses exec(compile(src, filename, 'exec')) where
#              `filename` = the real workspace path, but __file__ is NOT
#              injected into the executed namespace (NameError on access).
#              However, Python's compile() stores `filename` in the code
#              object's co_filename, so inspect.stack()[0].filename always
#              returns the correct on-disk path regardless of exec context.
#
# Wrong approach (DO NOT USE):  Path.cwd() / "scripts" / "04_build_pipeline.py"
#   In Databricks serverless, cwd() resolves to /databricks/driver — nowhere
#   near the Workspace bundle path — so parents[1] never finds config.py.

import inspect as _inspect

try:
    _ROOT = Path(__file__).resolve().parents[1]          # Context A & B
except NameError:
    # Context C: read co_filename from the innermost stack frame
    _ROOT = Path(_inspect.stack()[0].filename).resolve().parents[1]

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402

from google.api_core.exceptions import AlreadyExists, ResourceExhausted
from google.cloud import storage
from google.cloud import vectorsearch_v1beta as vs
from tqdm import tqdm
from vertexai import init as vertex_init
from vertexai.language_models import TextEmbeddingModel

# ── Index constants (same as 03_create_index.py) ──────────────────────────────
INDEX_ID   = "embedding-ann-index"
INDEX_NAME = "ScaNN ANN index — embedding field"


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Collection
# ═══════════════════════════════════════════════════════════════════════════════

def stage_collection(client: vs.VectorSearchServiceClient, dry_run: bool) -> str:
    """Create the VS2.0 Collection. Returns its resource name."""
    parent = f"projects/{config.PROJECT_ID}/locations/{config.LOCATION}"
    resource_name = f"{parent}/collections/{config.COLLECTION_ID}"

    if dry_run:
        print(f"  [DRY-RUN] Would create collection '{config.COLLECTION_ID}'")
        return resource_name

    data_schema = {
        "type": "object",
        "properties": {
            "name":              {"type": "string"},
            "neighbourhood":     {"type": "string"},
            "property_type":     {"type": "string"},
            "room_type":         {"type": "string"},
            "bathrooms_text":    {"type": "string"},
            "listing_url":       {"type": "string"},
            "host_name":         {"type": "string"},
            "host_is_superhost": {"type": "string"},
            "instant_bookable":  {"type": "string"},
            "text":              {"type": "string"},
            "accommodates":      {"type": "number"},
            "bedrooms":          {"type": "number"},
            "beds":              {"type": "number"},
            "price":             {"type": "number"},
            "rating":            {"type": "number"},
            "latitude":          {"type": "number"},
            "longitude":         {"type": "number"},
        },
    }
    vector_schema = {"embedding": {"dense_vector": {"dimensions": 768}}}

    collection = vs.Collection(
        display_name=f"Airbnb Listings — {config.COLLECTION_ID}",
        description="VS2.0 collection for Airbnb Agentic RAG (Austin, TX listings).",
        data_schema=data_schema,
        vector_schema=vector_schema,
    )
    try:
        op = client.create_collection(
            request=vs.CreateCollectionRequest(
                parent=parent,
                collection_id=config.COLLECTION_ID,
                collection=collection,
            )
        )
        result = op.result()
        print(f"  ✓  Collection created: {result.name}")
        return result.name
    except AlreadyExists:
        print(f"  ✓  Collection already exists — skipping creation.")
        return resource_name


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Ingest
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_price(s: str) -> float | None:
    try:
        return float(s.replace("$", "").replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def _parse_float(s: str) -> float | None:
    try:
        return float(s) if s and s.strip() else None
    except ValueError:
        return None


def _bool_str(s: str) -> str:
    return "true" if s.strip().lower() in {"t", "true", "1", "yes"} else "false"


def _embedding_text(row: dict) -> str:
    parts = []
    if row.get("name"):
        parts.append(row["name"].strip())
    if row.get("description"):
        parts.append(row["description"].strip()[:600])
    nbhd = row.get("neighbourhood_cleansed") or row.get("neighbourhood", "")
    if nbhd.strip():
        parts.append(f"Located in {nbhd.strip()}.")
    rt   = row.get("room_type", "")
    acm  = row.get("accommodates", "")
    if rt or acm:
        parts.append(f"{rt}. Accommodates {acm} guests.".strip())
    if row.get("neighborhood_overview"):
        parts.append(row["neighborhood_overview"].strip()[:400])
    props = []
    if row.get("bedrooms"):
        props.append(f"{row['bedrooms']} bedroom(s)")
    if row.get("beds"):
        props.append(f"{row['beds']} bed(s)")
    if row.get("bathrooms_text"):
        props.append(row["bathrooms_text"])
    if props:
        parts.append(", ".join(props) + ".")
    return " ".join(parts)


def _make_id(listing_id: str) -> str:
    return hashlib.sha1(f"airbnb:{listing_id}".encode()).hexdigest()


def _embed_batch(model: TextEmbeddingModel, texts: list[str], max_retries: int = 6) -> list:
    for attempt in range(max_retries):
        try:
            return model.get_embeddings(texts)
        except ResourceExhausted:
            if attempt == max_retries - 1:
                raise
            wait = 5.0 * (2 ** attempt) * (1.0 + random.uniform(-0.3, 0.3))
            tqdm.write(f"\n  ⚠  429 rate-limit — retrying in {wait:.1f}s...")
            time.sleep(wait)
    raise RuntimeError("Embedding retry limit exceeded")


def _upsert_batch(
    client: vs.DataObjectServiceClient,
    parent: str,
    objects: list[vs.DataObject],
) -> int:
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
            request=vs.BatchCreateDataObjectsRequest(parent=parent, requests=requests)
        )
        return len(objects)
    except AlreadyExists:
        created = 0
        for req in requests:
            try:
                client.create_data_object(request=req)
                created += 1
            except AlreadyExists:
                pass
        return created


def stage_ingest(max_listings: int | None, dry_run: bool) -> int:
    """Download CSV from GCS, embed, and upsert DataObjects. Returns count."""
    gcs_path = f"{config.INPUT_PREFIX}{config.GCS_FILENAME}"
    print(f"  Downloading gs://{config.BUCKET_NAME}/{gcs_path} ...")

    if dry_run:
        print(f"  [DRY-RUN] Would ingest up to {max_listings or 'all'} listings")
        return 0

    storage_client = storage.Client(project=config.PROJECT_ID)
    blob = storage_client.bucket(config.BUCKET_NAME).blob(gcs_path)
    if not blob.exists():
        print(f"  ✗  GCS file not found. Run scripts/00_upload_data.py first.")
        sys.exit(1)

    raw_bytes = blob.download_as_bytes()
    print(f"  ✓  Downloaded ({len(raw_bytes) / 1_048_576:.1f} MB)")

    # Parse
    limit = max_listings if max_listings and max_listings > 0 else None
    records: list[dict] = []
    with gzip.open(io.BytesIO(raw_bytes), "rt", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if limit and i >= limit:
                break
            text = _embedding_text(row)
            if not text.strip():
                continue
            price = _parse_price(row.get("price", ""))
            records.append({
                "obj_id": _make_id(row["id"]),
                "text":   text,
                "meta": {
                    "name":              (row.get("name") or "")[:500],
                    "neighbourhood":     row.get("neighbourhood_cleansed") or "",
                    "property_type":     row.get("property_type") or "",
                    "room_type":         row.get("room_type") or "",
                    "bathrooms_text":    row.get("bathrooms_text") or "",
                    "listing_url":       row.get("listing_url") or "",
                    "host_name":         row.get("host_name") or "",
                    "host_is_superhost": _bool_str(row.get("host_is_superhost", "")),
                    "instant_bookable":  _bool_str(row.get("instant_bookable", "")),
                    "text":              text[:2000],
                    "accommodates":      _parse_float(row.get("accommodates", "")) or 0.0,
                    "bedrooms":          _parse_float(row.get("bedrooms", "")) or 0.0,
                    "beds":              _parse_float(row.get("beds", "")) or 0.0,
                    "price":             price if price is not None else 0.0,
                    "rating":            _parse_float(row.get("review_scores_rating", "")) or 0.0,
                    "latitude":          _parse_float(row.get("latitude", "")) or 0.0,
                    "longitude":         _parse_float(row.get("longitude", "")) or 0.0,
                },
            })

    print(f"  ✓  Parsed {len(records)} listings")

    # Embed + upsert
    embedding_model = TextEmbeddingModel.from_pretrained(config.EMBEDDING_MODEL)
    do_client       = vs.DataObjectServiceClient()
    parent          = config.COLLECTION_RESOURCE
    data_objects: list[vs.DataObject] = []

    print(f"  Embedding (batch_size={config.BATCH_SIZE})...")
    with tqdm(total=len(records), unit="listing") as pbar:
        for i in range(0, len(records), config.BATCH_SIZE):
            batch  = records[i : i + config.BATCH_SIZE]
            embeds = _embed_batch(embedding_model, [r["text"] for r in batch])
            for rec, emb in zip(batch, embeds):
                data_objects.append(vs.DataObject(
                    data_object_id=rec["obj_id"],
                    data=rec["meta"],
                    vectors={"embedding": vs.Vector(dense=vs.DenseVector(values=list(emb.values)))}
                ))
            pbar.update(len(batch))
            time.sleep(0.5)

    print(f"  Upserting {len(data_objects)} DataObjects...")
    with tqdm(total=len(data_objects), unit="obj") as pbar:
        for i in range(0, len(data_objects), config.BATCH_SIZE):
            _upsert_batch(do_client, parent, data_objects[i : i + config.BATCH_SIZE])
            pbar.update(min(config.BATCH_SIZE, len(data_objects) - i))

    return len(data_objects)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Index
# ═══════════════════════════════════════════════════════════════════════════════

def stage_index(client: vs.VectorSearchServiceClient, dry_run: bool) -> None:
    """Create the ScaNN ANN index. Polls until done."""
    # Check if already exists
    existing = list(client.list_indexes(
        request=vs.ListIndexesRequest(parent=config.COLLECTION_RESOURCE)
    ))
    if existing:
        print(f"  ✓  ANN index already exists: '{existing[0].name.split('/')[-1]}' — skipping.")
        return

    if dry_run:
        print(f"  [DRY-RUN] Would create ScaNN index '{INDEX_ID}'")
        return

    index = vs.Index(
        display_name=INDEX_NAME,
        index_field="embedding",
        distance_metric=vs.DistanceMetric.DOT_PRODUCT,
        dense_scann=vs.DenseScannIndex(feature_norm_type=2),  # UNIT_L2_NORM
    )
    try:
        op = client.create_index(
            request=vs.CreateIndexRequest(
                parent=config.COLLECTION_RESOURCE,
                index_id=INDEX_ID,
                index=index,
            )
        )
    except AlreadyExists:
        print(f"  ✓  Index already exists.")
        return

    print(f"  Index build started — polling every 30s (Ctrl+C safe, build continues on GCP)...")
    start, max_wait = time.time(), 7200
    while True:
        elapsed = time.time() - start
        if op.done():
            print(f"\r  ✓  Index built in {elapsed:.0f}s          ")
            return
        if elapsed >= max_wait:
            print("\n  ⚠  2-hour client limit reached — build still running on GCP.")
            print("     Re-run with --skip-collection --skip-ingest to check.")
            return
        m, s = int(elapsed // 60), int(elapsed % 60)
        print(f"\r  ⏳  {m:02d}:{s:02d} elapsed...", end="", flush=True)
        time.sleep(30)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified Airbnb RAG build pipeline (collection → ingest → index)"
    )
    p.add_argument("--skip-collection", action="store_true",
                   help="Skip Stage 1 — collection must already exist")
    p.add_argument("--skip-ingest",     action="store_true",
                   help="Skip Stage 2 — data already ingested")
    p.add_argument("--skip-index",      action="store_true",
                   help="Skip Stage 3 — keep kNN / no ANN index")
    p.add_argument("--dry-run",         action="store_true",
                   help="Print plan only; make no API calls")
    p.add_argument("--max-listings",    type=int, default=None,
                   help=f"Override MAX_LISTINGS (default: {config.MAX_LISTINGS})")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    max_listings = args.max_listings if args.max_listings is not None else config.MAX_LISTINGS

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       Airbnb Agentic RAG — Unified Build Pipeline           ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Project    : {config.PROJECT_ID}")
    print(f"  Location   : {config.LOCATION}")
    print(f"  Collection : {config.COLLECTION_ID}")
    print(f"  GCS source : gs://{config.BUCKET_NAME}/{config.INPUT_PREFIX}{config.GCS_FILENAME}")
    print(f"  Max listing: {max_listings or 'all'}")
    print()
    print("  Stages to run:")
    print(f"    Stage 1 — Collection : {'SKIP' if args.skip_collection else 'RUN'}")
    print(f"    Stage 2 — Ingest     : {'SKIP' if args.skip_ingest     else 'RUN'}")
    print(f"    Stage 3 — ANN Index  : {'SKIP' if args.skip_index      else 'RUN'}")
    if args.dry_run:
        print("  *** DRY-RUN MODE — no API calls will be made ***")
    print()

    t0 = time.time()

    # ── Vertex AI init (skip in dry-run) ─────────────────────────────────────
    if not args.dry_run:
        vertex_init(project=config.PROJECT_ID, location=config.LOCATION)

    vs_client = vs.VectorSearchServiceClient() if not args.dry_run else None

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    if not args.skip_collection:
        print("━━━  Stage 1: Create Collection  ━━━")
        stage_collection(vs_client, args.dry_run)
        print()
    else:
        print("━━━  Stage 1: Collection  [SKIPPED]  ━━━\n")

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    if not args.skip_ingest:
        print("━━━  Stage 2: Ingest DataObjects  ━━━")
        n = stage_ingest(max_listings, args.dry_run)
        print(f"  ✓  {n} DataObjects upserted\n")
    else:
        print("━━━  Stage 2: Ingest  [SKIPPED]  ━━━\n")

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    if not args.skip_index:
        print("━━━  Stage 3: Create ScaNN ANN Index  ━━━")
        stage_index(vs_client, args.dry_run)
        print()
    else:
        print("━━━  Stage 3: ANN Index  [SKIPPED — running in kNN mode]  ━━━\n")

    elapsed = time.time() - t0
    print("╔══════════════════════════════════════════════════════════════╗")
    print(f"║  Pipeline complete in {elapsed:.0f}s{' ' * (38 - len(str(int(elapsed))))  }║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print("  Next:")
    print("    uvicorn app.main:app --reload --port 8000   # start backend")
    print("    streamlit run ui/streamlit_app.py           # start frontend")
    print("    python eval/evaluate.py                     # run evaluation")
    print()


if __name__ == "__main__":
    main()
