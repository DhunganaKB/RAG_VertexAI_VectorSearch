"""
Step 2 — Ingest Documents into the Vector Search 2.0 Collection
================================================================
Reads PDFs from GCS bucket-kamal-987/raw/, chunks them, embeds each
chunk with text-embedding-005, and stores the result as DataObjects
in the VS2.0 Collection.

Each DataObject contains:
  data    → { title, source, text }   ← metadata stored IN the collection
  vectors → { embedding: [768 floats] } ← the embedding

No Firestore needed — VS2.0 stores everything together.

Safe to re-run: existing chunks are skipped (AlreadyExists caught per batch).

Run:
    python scripts/02_ingest.py
"""
from __future__ import annotations

import hashlib
import html
import io
import json
import re
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

from google.api_core.exceptions import AlreadyExists
from google.cloud import storage
from google.cloud import vectorsearch_v1beta as vs
from pypdf import PdfReader
from vertexai import init as vertex_init
from vertexai.language_models import TextEmbeddingModel


# ─── Text Extraction ──────────────────────────────────────────────────────────

def html_to_text(raw: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", raw, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(re.sub(r"\s+", " ", text)).strip()


def extract_text(blob_name: str, content: bytes) -> str:
    suffix = Path(blob_name).suffix.lower()
    if suffix in {".txt", ".md"}:
        return content.decode("utf-8", errors="ignore")
    if suffix in {".html", ".htm"}:
        return html_to_text(content.decode("utf-8", errors="ignore"))
    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    return ""


# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    chunks, start = [], 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def make_chunk_id(uri: str, idx: int, text: str) -> str:
    """Deterministic SHA1 ID — same content always produces same ID."""
    return hashlib.sha1(f"{uri}:{idx}:{text[:120]}".encode()).hexdigest()


# ─── Batching ─────────────────────────────────────────────────────────────────

def batched(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ─── VS2.0 DataObject helpers ─────────────────────────────────────────────────

def build_data_object(chunk_id: str, title: str, source: str, text: str,
                      embedding: list[float]) -> vs.DataObject:
    """Build a VS2.0 DataObject with metadata + vector."""
    return vs.DataObject(
        data_object_id=chunk_id,
        # data_schema fields: stored as metadata in the collection
        data={
            "title":  title,
            "source": source,
            "text":   text,
        },
        # vector_schema field: the embedding
        vectors={
            "embedding": vs.Vector(
                dense=vs.DenseVector(values=embedding)
            )
        },
    )


def upsert_batch(client: vs.DataObjectServiceClient,
                 parent: str,
                 objects: list[vs.DataObject]) -> int:
    """Batch-create DataObjects; skips ones that already exist."""
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
        # At least one in this batch already exists — fall back to one-by-one
        created = 0
        for req in requests:
            try:
                client.create_data_object(request=req)
                created += 1
            except AlreadyExists:
                pass   # skip silently
        return created


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Vector Search 2.0 — RAG Ingestion Pipeline")
    print("=" * 60)

    vertex_init(project=config.PROJECT_ID, location=config.LOCATION)

    # ── Step 1: Read documents from GCS ──────────────────────────────────────
    print(f"\n[1/4] Reading documents from gs://{config.BUCKET_NAME}/{config.INPUT_PREFIX}")
    storage_client = storage.Client(project=config.PROJECT_ID)
    blobs = [
        b for b in storage_client.list_blobs(config.BUCKET_NAME, prefix=config.INPUT_PREFIX)
        if not b.name.endswith("/")
    ]
    if not blobs:
        print("  No files found. Upload PDFs to the bucket first.")
        return
    for b in blobs:
        print(f"  Found: gs://{config.BUCKET_NAME}/{b.name}")

    # ── Step 2: Chunk all documents ───────────────────────────────────────────
    print(f"\n[2/4] Chunking (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")
    chunks: list[dict] = []
    bucket = storage_client.bucket(config.BUCKET_NAME)
    for blob in blobs:
        raw = bucket.blob(blob.name).download_as_bytes()
        text = extract_text(blob.name, raw)
        if not text.strip():
            print(f"  Skipping {blob.name} (no extractable text)")
            continue
        uri   = f"gs://{config.BUCKET_NAME}/{blob.name}"
        title = Path(blob.name).name
        for idx, chunk in enumerate(chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            chunks.append({
                "id":     make_chunk_id(uri, idx, chunk),
                "title":  title,
                "source": uri,
                "text":   chunk,
            })
    print(f"  Total chunks: {len(chunks)}")
    if not chunks:
        print("  No chunks produced. Exiting.")
        return

    # ── Step 3: Embed chunks in batches ───────────────────────────────────────
    print(f"\n[3/4] Embedding with {config.EMBEDDING_MODEL}...")
    embedding_model = TextEmbeddingModel.from_pretrained(config.EMBEDDING_MODEL)
    data_objects: list[vs.DataObject] = []
    for i, batch in enumerate(batched(chunks, config.BATCH_SIZE)):
        embeddings = embedding_model.get_embeddings([c["text"] for c in batch])
        for chunk, emb in zip(batch, embeddings):
            data_objects.append(
                build_data_object(
                    chunk_id=chunk["id"],
                    title=chunk["title"],
                    source=chunk["source"],
                    text=chunk["text"],
                    embedding=list(emb.values),
                )
            )
        done = min((i + 1) * config.BATCH_SIZE, len(chunks))
        print(f"  Embedded {done}/{len(chunks)} chunks", end="\r")
    print(f"  Embedded {len(chunks)}/{len(chunks)} chunks")

    # ── Step 4: Upsert DataObjects into the Collection ────────────────────────
    print(f"\n[4/4] Upserting {len(data_objects)} DataObjects into collection '{config.COLLECTION_ID}'...")
    do_client = vs.DataObjectServiceClient()
    parent = config.COLLECTION_RESOURCE

    total_created = 0
    for i, batch in enumerate(batched(data_objects, config.BATCH_SIZE)):
        created = upsert_batch(do_client, parent, batch)
        total_created += created
        done = min((i + 1) * config.BATCH_SIZE, len(data_objects))
        print(f"  Upserted {done}/{len(data_objects)} DataObjects", end="\r")
    print(f"  Upserted {len(data_objects)}/{len(data_objects)} DataObjects")

    print("\n" + "=" * 60)
    print("Ingestion complete!")
    print(f"  Collection : {config.COLLECTION_RESOURCE}")
    print(f"  Chunks     : {len(chunks)}")
    print(f"  DataObjects: {len(data_objects)}")
    print("=" * 60)
    print("\nNext step: start the chatbot with:")
    print("  uvicorn app.main:app --reload --port 8000")
    print("  streamlit run ui/streamlit_app.py")


if __name__ == "__main__":
    main()
