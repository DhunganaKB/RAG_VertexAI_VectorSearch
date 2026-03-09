from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import re
import sys
from pathlib import Path
from typing import Iterable

from google.cloud import aiplatform_v1
from google.cloud import firestore as firestore_module
from google.cloud import storage
from pypdf import PdfReader
from vertexai import init as vertex_init
from vertexai.language_models import TextEmbeddingModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import load_project_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest files from GCS to Vertex AI Vector Search.")
    parser.add_argument("--project-id")
    parser.add_argument("--region")
    parser.add_argument("--bucket")
    parser.add_argument("--prefix")
    parser.add_argument("--index-resource-name")
    parser.add_argument("--embedding-model")
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--metadata-output")
    parser.add_argument("--metadata-gcs-path")
    parser.add_argument("--firestore-collection")
    return parser.parse_args()


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
        pages: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text:
                pages.append(page_text)
        return "\n".join(pages)
    return ""


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk-overlap must be smaller than chunk-size")

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def batched(items: list, size: int) -> Iterable[list]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def build_chunk_id(uri: str, chunk_index: int, chunk_text_value: str) -> str:
    fingerprint = hashlib.sha1(f"{uri}:{chunk_index}:{chunk_text_value[:120]}".encode("utf-8")).hexdigest()
    return fingerprint


def write_metadata_to_firestore(
    chunks: list[dict],
    project_id: str,
    collection: str,
) -> None:
    """Write chunk metadata to Firestore using batched commits.

    Each document is keyed by chunk_id so the serving layer can retrieve
    exactly the top-K documents it needs via a single db.get_all() RPC,
    rather than loading the entire metadata file into memory.

    Firestore batch limit is 500 operations per commit.
    """
    db = firestore_module.Client(project=project_id)
    col = db.collection(collection)

    total = 0
    for batch_chunk in batched(chunks, 500):
        fs_batch = db.batch()
        for row in batch_chunk:
            doc_ref = col.document(row["id"])
            fs_batch.set(
                doc_ref,
                {
                    "title": row["title"],
                    "uri": row["uri"],
                    "chunk": row["chunk"],
                },
            )
        fs_batch.commit()
        total += len(batch_chunk)
        print(f"  Firestore: committed {total}/{len(chunks)} metadata documents")


def main() -> None:
    args = parse_args()
    settings = load_project_settings()

    project_id = args.project_id or settings.project_id
    region = args.region or settings.region
    bucket_name = args.bucket or settings.bucket_name
    prefix = args.prefix or settings.input_prefix
    index_resource_name = args.index_resource_name or settings.index_resource_name
    embedding_model_name = args.embedding_model or settings.embedding_model
    metadata_output = args.metadata_output or settings.metadata_local_path
    metadata_gcs_path = args.metadata_gcs_path or settings.metadata_gcs_path
    firestore_collection = args.firestore_collection or settings.firestore_collection

    if not index_resource_name:
        raise RuntimeError(
            "Missing index resource name. Run scripts/create_vector_search.py first."
        )

    vertex_init(project=project_id, location=region)

    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blobs = list(storage_client.list_blobs(bucket_name, prefix=prefix))
    if not blobs:
        print(f"No files found in gs://{bucket_name}/{prefix}")
        return

    embedding_model = TextEmbeddingModel.from_pretrained(embedding_model_name)

    chunks: list[dict] = []
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        raw_bytes = bucket.blob(blob.name).download_as_bytes()
        text = extract_text(blob.name, raw_bytes)
        if not text.strip():
            continue
        uri = f"gs://{bucket_name}/{blob.name}"
        title = Path(blob.name).name
        file_chunks = chunk_text(text, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        for idx, chunk in enumerate(file_chunks):
            chunk_id = build_chunk_id(uri=uri, chunk_index=idx, chunk_text_value=chunk)
            chunks.append({"id": chunk_id, "title": title, "uri": uri, "chunk": chunk})

    if not chunks:
        print("No text chunks produced from the provided files.")
        return

    # --- Step 1: Upsert embeddings into Vertex AI Vector Search ---
    index_service = aiplatform_v1.IndexServiceClient(
        client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"}
    )

    datapoints: list[aiplatform_v1.IndexDatapoint] = []
    for batch in batched(chunks, args.batch_size):
        embeddings = embedding_model.get_embeddings([row["chunk"] for row in batch])
        for row, embedding in zip(batch, embeddings, strict=True):
            datapoints.append(
                aiplatform_v1.IndexDatapoint(
                    datapoint_id=row["id"],
                    feature_vector=list(embedding.values),
                )
            )

    for dp_batch in batched(datapoints, 200):
        index_service.upsert_datapoints(
            request=aiplatform_v1.UpsertDatapointsRequest(
                index=index_resource_name,
                datapoints=dp_batch,
            )
        )

    # --- Step 2: Write metadata to Firestore (scalable key-value store) ---
    # The serving layer fetches only top-K documents per query via get_all(),
    # so it never needs to load the full corpus into memory.
    print(f"\nWriting {len(chunks)} metadata records to Firestore collection '{firestore_collection}'...")
    write_metadata_to_firestore(
        chunks=chunks,
        project_id=project_id,
        collection=firestore_collection,
    )

    # --- Step 3: Write local JSONL artifact (backup / offline inspection) ---
    metadata_path = Path(metadata_output)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in chunks:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    bucket.blob(metadata_gcs_path).upload_from_filename(str(metadata_path))

    print(
        json.dumps(
            {
                "chunks_indexed": len(chunks),
                "firestore_collection": firestore_collection,
                "metadata_local_path": str(metadata_path),
                "metadata_gcs_uri": f"gs://{bucket_name}/{metadata_gcs_path}",
                "index_resource_name": index_resource_name,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
