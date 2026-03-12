# RAG with Vertex AI Vector Search 2.0

A production-ready RAG pipeline using **Vector Search 2.0 Collections**.

## Why VS2.0 over VS1.0?

| Feature | VS1.0 (old) | VS2.0 (this project) |
|---|---|---|
| Resources needed | Index + Endpoint + Firestore | **Collection only** |
| Metadata storage | Firestore (separate RPC) | **Inside the Collection** |
| Retrieval | find_neighbors → Firestore lookup | **Single `SearchDataObjects` call** |
| Setup complexity | High | **Low** |

---

## Project Structure

```
RAGVectoSearch20/
├── config.py                    # ← Single config file (edit this)
├── requirements.txt
├── README.md
│
├── scripts/
│   ├── 00_delete_collections.py # Optional: Delete collections & purge DataObjects
│   ├── 01_setup_collection.py   # Step 1: Create the VS2.0 Collection
│   ├── 02_ingest.py             # Step 2: Chunk → Embed → Upsert DataObjects
│   └── inspect_collection.py   # Optional: Inspect stored records (metadata + vectors)
│
├── app/
│   ├── rag.py                   # VS2.0 retriever (DataObjectSearchServiceClient)
│   ├── llm.py                   # Gemini answer generation
│   └── main.py                  # FastAPI backend  →  POST /rag
│
└── ui/
    └── streamlit_app.py         # Streamlit chatbot frontend
```

---

## Prerequisites

### 1. GCP APIs (run once)
```bash
gcloud services enable vectorsearch.googleapis.com aiplatform.googleapis.com \
  --project deve-0000

# Verify
gcloud services list --enabled --project deve-0000 \
  --filter="name:vectorsearch OR name:aiplatform"
```

### 2. Authentication
```bash
gcloud auth application-default login
```

### 3. Create a dedicated Python environment

It is recommended to create a **fresh** conda environment for this project
rather than reusing an existing one, to avoid package conflicts.

```bash
# Create a new environment (Python 3.11 recommended)
conda create -n vs2-rag python=3.11 -y

# Activate it
conda activate vs2-rag

# Install all required packages
pip install -r requirements.txt
```

> **Why a new environment?**
> `google-cloud-vectorsearch` pins specific versions of `protobuf` and
> `grpcio` that can conflict with other Google Cloud packages already present
> in a shared environment. Starting fresh avoids those issues.

Once created, always activate this environment before running any script:
```bash
conda activate vs2-rag
```

---

## Run the Pipeline

### Step 1 — Create the Collection
```bash
conda activate vs2-rag
python scripts/01_setup_collection.py
```
> Safe to run multiple times — skips creation if the collection already exists.

### Step 2 — Ingest Documents
```bash
python scripts/02_ingest.py
```
Reads `gs://bucket-kamal-000/raw/*.pdf`, chunks each document, embeds
with `text-embedding-005`, and stores everything as **DataObjects** in the
Collection. No Firestore needed.

### Step 3 — Start the FastAPI backend
```bash
uvicorn app.main:app --reload --port 8000
```
Test it:
```bash
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic of the documents?", "top_k": 5}'
```

### Step 4 — Start the Streamlit chatbot
```bash
streamlit run ui/streamlit_app.py
```
Open http://localhost:8501 in your browser.

---

## How It Works

```
User Question
     │
     ▼
text-embedding-005  (embed the question)
     │
     ▼
VS2.0 Collection  SearchDataObjects
     │  ← returns top-K DataObjects with both text AND metadata
     ▼
Gemini 2.0 Flash  (grounded answer generation)
     │
     ▼
Answer + Sources
```

### DataObject Structure
Each document chunk is stored as a **DataObject** containing:
```
DataObject
├── data_object_id  → SHA1(uri + chunk_index + text)
├── data            → { title, source, text }   ← metadata
└── vectors         → { embedding: [768 floats] }
```

The `data_schema` (metadata fields) and `vector_schema` (embedding field)
are defined on the Collection when it is created.

---

## Optional — Start Fresh (Delete & Rebuild)

If you want to wipe all existing collections and start from scratch, use the
delete script included in the project.

### Delete a specific collection
```bash
python scripts/00_delete_collections.py --id collection-for-rag-demo
```

### Delete all collections at once (no prompt)
```bash
python scripts/00_delete_collections.py --all
```

### Interactive mode (pick which ones to delete)
```bash
python scripts/00_delete_collections.py
```
Lists all collections with a numbered menu — enter the number(s) to delete,
`all`, or `q` to quit.

> **Note:** The script automatically purges all DataObjects inside a collection
> before deleting it. VS2.0 requires the collection to be empty before it can
> be removed — this is handled for you.

### Full reset workflow
```bash
# 1. Delete everything
python scripts/00_delete_collections.py --all

# 2. Recreate the collection
python scripts/01_setup_collection.py

# 3. Re-ingest documents
python scripts/02_ingest.py

# 4. Start the backend - one terminal
uvicorn app.main:app --reload --port 8000

# 5. Start the chatbot - second terminal
streamlit run ui/streamlit_app.py
```

---

## Optional — Inspect Collection Records

After ingestion, you can peek inside the collection to verify that metadata and
embeddings were stored correctly.

```bash
python scripts/inspect_collection.py
```

### What it shows

Each record prints three sections:

| Section | Fields | Description |
|---|---|---|
| **System** | `ID`, `Created`, `Updated` | SHA1 chunk ID and API timestamps |
| **Metadata** | `title`, `source`, `text` | Filename, GCS URI, chunk text (truncated) |
| **Embedding** | `dims`, `values`, `range` | 768-dim vector — first 8 + last 4 values shown, plus min/max/mean stats |

### Command options

```bash
# Show first 3 records (default)
python scripts/inspect_collection.py

# Show first 5 records
python scripts/inspect_collection.py --n 5

# Show 1 record with full text and all 768 vector values
python scripts/inspect_collection.py --n 1 --full
```

### Example output

```
══ VS2.0 Collection: collection-for-rag-demo ══
Location : us-central1   Project : deve-487713
Records  : 3 shown (more exist)

┌─ Record 1 ──────────────────────────────────────────────────────────────
│ ID          : be26c7c133c645e4ca8205fdf18f4ebcc838eaae
│ Created     : 2026-03-12 12:23:46+00:00
│ Updated     : 2026-03-12 12:23:46+00:00
│
│ ── METADATA ──
│ title       : document2.pdf
│ source      : gs://bucket-kamal-987/raw/document2.pdf
│ text        : Benchmark Dataset for Multi-Cultural Value Awareness ...  …
│
│ ── EMBEDDING ──
│ field       : embedding
│ dims        : 768
│ values      : [-0.018605, -0.052140, 0.041653, …  -0.006900, 0.059695]  (768 dims)
│ range       : min=-0.118341  max=0.140543  mean=-0.002525
└─────────────────────────────────────────────────────────────────────────
```

> **Tip:** The `ID` is a SHA1 hash of `source + chunk_index + text`, so
> re-ingesting the same document produces the same IDs — safe to run
> `02_ingest.py` multiple times without creating duplicates.

---

## Configuration

Edit `config.py` to change any settings:

| Variable | Default | Description |
|---|---|---|
| `PROJECT_ID` | `deve-487713` | GCP project |
| `LOCATION` | `us-central1` | Region |
| `COLLECTION_ID` | `collection-for-rag-demo` | VS2.0 Collection name |
| `BUCKET_NAME` | `bucket-kamal-987` | GCS bucket |
| `INPUT_PREFIX` | `raw/` | GCS folder with documents |
| `EMBEDDING_MODEL` | `text-embedding-005` | 768-dim embedding |
| `GEMINI_MODEL` | `gemini-2.0-flash-001` | Generation model |
| `CHUNK_SIZE` | `1200` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `5` | Chunks retrieved per query |
