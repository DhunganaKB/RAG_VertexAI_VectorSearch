# RAG with Vertex AI Vector Search + Firestore + Gemini

End-to-end Retrieval-Augmented Generation (RAG) pipeline on Google Cloud Platform.

```
Documents (GCS)
    │
    ▼ ingest_from_bucket.py
┌──────────────┐       ┌─────────────────────────────┐
│  Text Chunks │──────▶│  Vertex AI Vector Search     │  (ANN index)
│  + Metadata  │──────▶│  Firestore  (rag_chunks)     │  (metadata store)
└──────────────┘       └─────────────────────────────┘
                                   │
                          query at serve time
                                   ▼
                 embed(query) → Vector Search → top-K IDs
                                   │
                         Firestore get_all(IDs)   ← only K docs fetched
                                   │
                            Gemini (answer)
                                   │
                    FastAPI /rag  ◀──▶  Streamlit UI
```

**Why Firestore instead of a JSONL file?**
The naive approach loads `chunks.jsonl` fully into RAM at startup — O(total_corpus) memory.
Firestore's `db.get_all(refs)` fetches exactly the top-K documents returned by Vector Search —
O(top_k) per query, regardless of corpus size. Google's recommended pattern for scalable RAG.

---

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | 3.11+ | [python.org](https://www.python.org/downloads/) |
| gcloud CLI | latest | [cloud.google.com/sdk](https://cloud.google.com/sdk/docs/install) |
| GCP project | — | must have billing enabled |

---

## Project structure

```
RAGVertexAISimple/
├── config.py                     ← ✏️  Single place to edit all settings
├── requirements.txt
├── Dockerfile
│
├── app/                          ← FastAPI backend
│   ├── config.py                 ←   Settings dataclass (reads root config.py)
│   ├── main.py                   ←   /rag  /health  /  endpoints
│   ├── rag.py                    ←   VectorRetriever + FirestoreMetadataStore
│   └── llm.py                    ←   Gemini generation wrapper
│
├── ui/                           ← Streamlit frontend
│   └── streamlit_app.py
│
├── scripts/
│   ├── bootstrap_resources.sh    ← Step 4: one-time GCP setup (idempotent)
│   ├── create_vector_search.py   ← Step 6: create/reuse Vector Search index
│   ├── ingest_from_bucket.py     ← Step 7: chunk → embed → index + Firestore
│   ├── trigger_ingestion.sh      ← convenience wrapper for ingest
│   └── clearup.sh                ← convenience wrapper for create_vector_search
│
├── artifacts/                    ← runtime files (gitignored)
│   ├── vector_resources.json     ←   index/endpoint names written by step 6
│   └── chunks.jsonl              ←   local backup of chunk metadata
│
└── input_docs/                   ← put your source documents here
```

---

## Step 1 — Clone the repo and create a Python virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python --version                 # should show 3.11+
pip install -r requirements.txt
```

Keep the same shell active for all subsequent steps so `.venv` is in scope.

---

## Step 2 — Authenticate with Google Cloud

```bash
# Login with your user account
gcloud auth login

# Set Application Default Credentials (ADC) — used by all Python GCP clients
gcloud auth application-default login
```

---

## Step 3 — Edit `config.py`

Open `config.py` and fill in the block under `✏️ EDIT THIS SECTION`:

```python
# config.py

PROJECT_ID   = "your-gcp-project-id"    # ← change this
REGION       = "us-central1"            # ← nearest Vertex AI region
BUCKET_NAME  = "your-unique-bucket"     # ← globally unique GCS bucket name
SERVICE_ACCOUNT_NAME = "rag-vector-search-sa"   # can keep as-is

GEMINI_MODEL    = "gemini-2.5-flash"
EMBEDDING_MODEL = "text-embedding-005"

VECTOR_INDEX_DISPLAY_NAME    = "rag-index-v"
VECTOR_ENDPOINT_DISPLAY_NAME = "rag-index-endpoint-v"
VECTOR_DEPLOYED_INDEX_ID     = "rag_deployed_index_v"
VECTOR_DIMENSIONS            = 768
VECTOR_MACHINE_TYPE          = "e2-standard-16"

FIRESTORE_DATABASE   = "(default)"   # keep as-is for most projects
FIRESTORE_COLLECTION = "rag_chunks"  # can keep as-is

TOP_K            = 5
MAX_OUTPUT_TOKENS = 1024
TEMPERATURE      = 0.2

API_BASE_URL  = "http://localhost:8000"  # FastAPI URL used by Streamlit
STREAMLIT_PORT = 8501
```

> **All scripts and the app read from this single file. You should not need
> to change any other file.**

---

## Step 4 — Bootstrap GCP resources (one-time, idempotent)

```bash
./scripts/bootstrap_resources.sh
```

This script is **safe to re-run** — it skips anything that already exists.

What it does:

| # | Resource | Action |
|---|---|---|
| 1 | GCP APIs | Enables `aiplatform`, `storage`, `iam`, `firestore` |
| 2 | Service account | Creates `<SA_NAME>@<PROJECT>.iam.gserviceaccount.com` |
| 3 | IAM roles | Grants `aiplatform.user`, `storage.objectAdmin`, `datastore.user` |
| 4 | GCS bucket | Creates `gs://<BUCKET_NAME>` with uniform bucket access |
| 5 | Firestore database | Creates Native-mode database `(default)` in your region |
| 6 | Local folders | Creates `artifacts/`, `input_docs/`, `ui/` if missing |

> **Firestore Native mode** is required. Datastore-mode databases are
> incompatible with the Python Firestore client library's `get_all()` API.

Expected output (abridged):
```
[1/6] Enabling GCP APIs...       Done
[2/6] Service account...         Created: rag-vector-search-sa@...
[3/6] Granting IAM roles...      Granted roles/aiplatform.user ...
[4/6] GCS bucket...              Created: gs://your-bucket
[5/6] Firestore database...      Created Firestore database '(default)' in us-central1
[6/6] Creating local folders...  artifacts/ input_docs/ ui/
Bootstrap complete!
```

---

## Step 5 — Upload your documents to GCS

Place source files in `input_docs/`. Supported formats: `.txt`, `.md`, `.html`, `.htm`, `.pdf`

```bash
# Copy all files to the raw/ prefix in your bucket
gcloud storage cp ./input_docs/* gs://<BUCKET_NAME>/raw/

# Verify
gcloud storage ls gs://<BUCKET_NAME>/raw/
```

Replace `<BUCKET_NAME>` with the value you set in `config.py`.

---

## Step 6 — Create the Vector Search index and endpoint

```bash
python scripts/create_vector_search.py
```

This script is **idempotent** — if the index or endpoint already exists (matched
by display name), it reuses them without recreating.

What it does:
1. Looks for an existing index matching `VECTOR_INDEX_DISPLAY_NAME` — creates if missing
2. Looks for an existing endpoint matching `VECTOR_ENDPOINT_DISPLAY_NAME` — creates if missing
3. Deploys the index to the endpoint (uses `VECTOR_DEPLOYED_INDEX_ID`)
4. Writes resource names to `artifacts/vector_resources.json`

Expected output:
```json
{
  "index_resource_name": "projects/.../indexes/...",
  "index_endpoint_resource_name": "projects/.../indexEndpoints/...",
  "deployed_index_id": "rag_deployed_index_v",
  "dimensions": 768,
  "saved_to": "artifacts/vector_resources.json"
}
```

> ⚠️ **This step can take 20–40 minutes** the first time (index creation + deployment).
> Subsequent runs complete in seconds if resources already exist.

Optional: force full recreation (destructive — deletes existing index/endpoint):
```bash
python scripts/create_vector_search.py --recreate
```

---

## Step 7 — Ingest documents

```bash
python scripts/ingest_from_bucket.py
```

What it does:

| Phase | What happens |
|---|---|
| Download | Reads each file from `gs://<BUCKET>/raw/` |
| Extract | Parses PDF / HTML / TXT / MD to plain text |
| Chunk | Splits text into overlapping windows (1200 chars, 200 overlap) |
| Embed | Calls `text-embedding-005` in batches of 32 |
| Vector Search | Upserts embeddings to the Vertex AI index (batches of 200) |
| Firestore | Writes `{title, uri, chunk}` per chunk to `rag_chunks` collection (batches of 500) |
| Backup | Writes `artifacts/chunks.jsonl` + uploads to `gs://<BUCKET>/metadata/chunks.jsonl` |

Expected output:
```json
{
  "chunks_indexed": 142,
  "firestore_collection": "rag_chunks",
  "metadata_local_path": "artifacts/chunks.jsonl",
  "metadata_gcs_uri": "gs://your-bucket/metadata/chunks.jsonl",
  "index_resource_name": "projects/.../indexes/..."
}
```

> **Re-running ingestion is safe.** Chunk IDs are deterministic SHA1 hashes of
> `(uri, chunk_index, chunk_prefix)`, so re-ingesting the same files is a no-op
> for existing chunks and automatically adds new ones.

---

## Step 8 — Start the FastAPI backend

Open **terminal 1**:

```bash
uvicorn app.main:app --reload --port 8000
```

Verify it's running:
```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

Browse the interactive API docs:
- `http://localhost:8000/docs` (Swagger UI)
- `http://localhost:8000/redoc`

---

## Step 9 — Start the Streamlit UI

Open **terminal 2**:

```bash
streamlit run ui/streamlit_app.py --server.port 8501
```

Open your browser at **`http://localhost:8501`**

The UI provides:
- Text area to type your question
- Top-K slider (1–20 sources)
- Backend health check button
- Answer displayed with source citations
- Expandable source cards (title, GCS URI, excerpt, relevance score)
- Full query history within the session

---

## Step 10 — Test via curl (optional)

```bash
curl -s http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the main topics", "top_k": 5}' | python -m json.tool
```

---

## Config reference

All settings live in **`config.py`** (root of the project):

| Constant | Default | Description |
|---|---|---|
| `PROJECT_ID` | — | GCP project ID |
| `REGION` | `us-central1` | GCP region for Vertex AI + Firestore |
| `BUCKET_NAME` | — | GCS bucket for documents and metadata backup |
| `SERVICE_ACCOUNT_NAME` | `rag-vector-search-sa` | SA used for IAM grants |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Generation LLM |
| `EMBEDDING_MODEL` | `text-embedding-005` | Must produce 768-dim vectors |
| `VECTOR_INDEX_DISPLAY_NAME` | `rag-index-v` | Vector Search index display name |
| `VECTOR_ENDPOINT_DISPLAY_NAME` | `rag-index-endpoint-v` | Endpoint display name |
| `VECTOR_DEPLOYED_INDEX_ID` | `rag_deployed_index_v` | Deployed index identifier |
| `VECTOR_DIMENSIONS` | `768` | Must match embedding model output |
| `VECTOR_MACHINE_TYPE` | `e2-standard-16` | VM type for the index |
| `INPUT_PREFIX` | `raw/` | GCS prefix for source documents |
| `METADATA_LOCAL_PATH` | `artifacts/chunks.jsonl` | Local JSONL backup |
| `METADATA_GCS_PATH` | `metadata/chunks.jsonl` | GCS JSONL backup |
| `FIRESTORE_DATABASE` | `(default)` | Firestore database ID |
| `FIRESTORE_COLLECTION` | `rag_chunks` | Firestore collection for chunk metadata |
| `TOP_K` | `5` | Default number of chunks retrieved per query |
| `MAX_OUTPUT_TOKENS` | `1024` | Gemini max output tokens |
| `TEMPERATURE` | `0.2` | Gemini temperature (0 = deterministic) |
| `API_BASE_URL` | `http://localhost:8000` | FastAPI URL used by Streamlit |
| `STREAMLIT_PORT` | `8501` | Streamlit server port (for documentation) |

---

## Architecture deep-dive

### Why Firestore for metadata?

| Approach | Memory | Latency | Scales to millions? |
|---|---|---|---|
| Load entire JSONL into RAM | O(corpus) | High startup | ❌ |
| **Firestore `get_all`** | **O(top_k)** | **~10–30 ms** | **✅** |
| BigQuery | O(top_k) | ~500 ms | ✅ (analytics) |
| Bigtable | O(top_k) | <5 ms | ✅ (ultra-high QPS) |

Firestore is Google's recommended choice for Vertex AI Vector Search RAG because:
- Serverless — no infrastructure to manage
- `db.get_all(refs)` is a single RPC fetching only the exact K documents
- Native GCP integration — same project, same IAM, same billing
- Free tier: 1 GiB storage, 50K reads/day

### Request flow

```
User (Streamlit)
    │  POST /rag  {"query": "..."}
    ▼
FastAPI (app/main.py)
    │
    ▼  embed(query)
TextEmbeddingModel (text-embedding-005)
    │  768-dim vector
    ▼
Vertex AI Vector Search (find_neighbors, top_k=5)
    │  returns [id_1, id_2, id_3, id_4, id_5]
    ▼
Firestore db.get_all([doc_ref_1 ... doc_ref_5])   ← single RPC, 5 docs only
    │  {title, uri, chunk} per doc
    ▼
build_context(chunks) → context string
    │
    ▼
Gemini generate_content(prompt + context)
    │  grounded answer
    ▼
RagResponse {answer, sources, model, endpoint}
    │
    ▼
Streamlit renders answer + expandable source cards
```

### Firestore data model

```
Collection: rag_chunks
  Document ID: <sha1(uri:chunk_index:chunk_prefix)>   ← same as Vector Search datapoint ID
    Fields:
      title : string   — filename
      uri   : string   — gs://bucket/raw/filename.pdf
      chunk : string   — raw text excerpt (up to ~1200 chars)
```

---

## Script quick-reference

| Script | Purpose | Idempotent? |
|---|---|---|
| `./scripts/bootstrap_resources.sh` | Enable APIs, create SA/bucket/Firestore/IAM | ✅ Yes |
| `python scripts/create_vector_search.py` | Create/reuse index + endpoint | ✅ Yes |
| `./scripts/clearup.sh` | Alias for `create_vector_search.py` | ✅ Yes |
| `python scripts/ingest_from_bucket.py` | Chunk → embed → upsert + Firestore | ✅ Yes (upsert) |
| `./scripts/trigger_ingestion.sh` | Alias for `ingest_from_bucket.py` | ✅ Yes |

---

## Troubleshooting

### `403 PermissionDenied` on Vertex AI or Firestore
```bash
# Verify your authenticated identity
gcloud auth list

# Re-run bootstrap to re-apply IAM grants
./scripts/bootstrap_resources.sh
```

### `RuntimeError: Vector endpoint not found`
`artifacts/vector_resources.json` is missing or empty. Run:
```bash
python scripts/create_vector_search.py
```

### `No files found in gs://...`
Files are not uploaded to the `raw/` prefix:
```bash
gcloud storage ls gs://<BUCKET_NAME>/raw/
gcloud storage cp ./input_docs/* gs://<BUCKET_NAME>/raw/
```

### Streamlit shows "Cannot reach backend"
FastAPI is not running. In terminal 1:
```bash
uvicorn app.main:app --reload --port 8000
```

### Sample questions: 
Currently, two sample PDF documents are included in the repository. To try out the RAG pipeline, you can use questions such as:

What is Self-Organizing Maps?

What is Distributed Signal Amplification?


### `/rag` returns an empty sources list
Ingestion has not been run, or ran before the index was deployed. Check:
```bash
cat artifacts/chunks.jsonl | wc -l    # should be > 0
cat artifacts/vector_resources.json   # should have endpoint + deployed_index_id
python scripts/ingest_from_bucket.py  # re-run if needed
```

### Firestore `google.api_core.exceptions.NotFound: ... database not found`
The Firestore database was not created. Re-run bootstrap:
```bash
./scripts/bootstrap_resources.sh
```

### `ModuleNotFoundError: No module named 'google.cloud.firestore'`
Firestore package is missing. Re-install dependencies:
```bash
pip install -r requirements.txt
```

---

## Docker (optional)

Build and run the FastAPI backend in a container:

```bash
docker build -t rag-api .
docker run -p 8080:8080 \
  -e GOOGLE_APPLICATION_CREDENTIALS=/key.json \
  -v /path/to/key.json:/key.json \
  -v $(pwd)/artifacts:/app/artifacts \
  rag-api
```

The Streamlit app can still run locally and point to the container:
```bash
RAG_API_URL=http://localhost:8080 streamlit run ui/streamlit_app.py
```
