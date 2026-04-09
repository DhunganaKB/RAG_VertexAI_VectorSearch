# Airbnb Agentic RAG — Vertex AI Vector Search 2.0

A production-ready Retrieval-Augmented Generation system for searching Austin, TX Airbnb listings, deployed on Google Cloud Run.

**Stack:** Vertex AI Vector Search 2.0 · Gemini 2.5 Flash · text-embedding-005 · Cloud Memorystore Redis · FastAPI · Streamlit

> **Full step-by-step deployment guide** (environment setup, GCP infrastructure, pipeline, deploy, verify, eval, load testing): see [`ragrun.html`](ragrun.html)

---

## What It Does

Two search modes over 3,000 real Airbnb listings in Austin, TX:

| Endpoint | Strategy | Best for |
|---|---|---|
| `POST /rag` | Simple RAG — embed query → ANN search → Gemini | Open-ended, preference-based queries |
| `POST /ask` | Agentic RAG — Gemini extracts filters → `find_rentals()` tool → filtered results → Gemini | Structured queries with price, bedrooms, room type, neighbourhood constraints |

Both endpoints share a **Redis cache** — a cache HIT returns in under 2ms, bypassing Vector Search and Gemini entirely.

---

## Architecture

```
OFFLINE  (run once)
──────────────────────────────────────────────────────────
GCS Bucket ──► 02_ingest.py ──► VS2.0 Collection
               (text-emb-005)   (3,000 DataObjects)
                                │
               03_create_index.py ──► ScaNN ANN Index

ONLINE  (Cloud Run — scales to zero)
──────────────────────────────────────────────────────────
Browser ──► airbnb-rag-ui (Streamlit)
                │  HTTPS
                ▼
        airbnb-rag-api (FastAPI)
                │
                ├─► Redis HIT  ──► return immediately (~1ms)
                │
                └─► MISS: embed → VS2.0 ScaNN ANN search
                              │
                         GetDataObject ×N  (parallel)
                              │
                    /rag: context → Gemini → answer
                    /ask: Gemini ReAct loop → answer
                              │
                         Redis SET (TTL)
```

---

## Project Structure

```
AirbnbAgenticRAG/
├── config.py                    # Single source of truth — reads all values from env vars
├── .env.example                 # Template: copy to .env and fill in your GCP values
├── .gitignore
├── Dockerfile.api               # FastAPI backend container
├── Dockerfile.ui                # Streamlit frontend container
├── requirements.txt             # Full dev dependencies
├── requirements-api.txt         # API container dependencies
├── requirements-ui.txt          # UI container dependencies
│
├── app/
│   ├── main.py                  # FastAPI: /rag, /ask, /health, /cache/stats
│   ├── rag.py                   # VS2Retriever — ANN search + context builder
│   └── cache.py                 # Memorystore Redis helpers (get/set/flush)
│
├── agent/
│   ├── agent.py                 # Gemini ReAct loop — function calling, max 5 turns
│   └── tools.py                 # find_rentals() — over-fetch + Python-side filter
│
├── scripts/                     # Offline data pipeline (run once)
│   ├── 00_upload_data.py        # Upload local CSV → GCS
│   ├── 01_setup_collection.py   # Create VS2.0 Collection + schema
│   ├── 02_ingest.py             # Embed + ingest listings as DataObjects
│   ├── 03_create_index.py       # Build ScaNN ANN index
│   ├── 04_build_pipeline.py     # Convenience wrapper — runs stages 1–3
│   ├── inspect_collection.py    # Show collection count, index status, samples
│   └── 00_delete_collections.py # Delete collection + index (reset)
│
├── eval/
│   ├── dataset.json             # 25 labeled evaluation queries
│   ├── evaluate.py              # Runner: calls API → scores → uploads to GCS
│   ├── metrics.py               # answer_relevance, filter_accuracy, keyword_hit_rate
│   └── report.py                # HTML report builder
│
├── load_test/
│   ├── locustfile.py            # Locust user scenarios (Health, RAG, Agent, Mixed)
│   └── locust.conf              # Default settings (20 users, 5 min)
│
├── ui/
│   └── streamlit_app.py         # Streamlit chatbot frontend
│
├── notebooks/                   # Jupyter walkthroughs
│   ├── 01_ingestion_walkthrough.ipynb
│   ├── 02_rag_vs_ask_walkthrough.ipynb
│   ├── 03_collection_and_index_explorer.ipynb
│   ├── 04_knn_vs_scann_benchmark.ipynb
│   └── 05_cache_latency_benchmark.ipynb
│
├── run/                         # Operation scripts — run in order
│   ├── 01_setup_gcp.sh          # One-time GCP infrastructure setup
│   ├── 02_run_pipeline.sh       # Run data ingestion pipeline
│   ├── 03_run_local.sh          # Start API + UI locally
│   ├── 04_deploy.sh             # Build images + deploy to Cloud Run
│   ├── 05_verify.sh             # Health check + endpoint tests
│   ├── 06_run_eval.sh           # RAGAS evaluation
│   ├── 07_run_loadtest.sh       # Locust load tests
│   └── 08_ops.sh                # Cache, Redis, re-ingestion, logs
│
└── docs/                        # Architecture diagrams (HTML)
```

---

## Quick Start

### 1. Prerequisites

- `gcloud` CLI ≥ 470.0.0
- Python ≥ 3.10 with pip
- A GCP project with billing enabled

### 2. Configuration

```bash
# Copy the template and fill in your values
cp .env.example .env
```

Minimum required fields in `.env`:

```bash
GCP_PROJECT_ID=your-gcp-project-id
GCS_BUCKET_NAME=your-gcs-bucket-name
```

### 3. Follow the Run Guide

All steps — GCP setup, data pipeline, local dev, Cloud Run deployment, verification, evaluation, and load testing — are documented end-to-end in **[`ragrun.html`](DetailRunSteps.html)**.

Open it in a browser and follow Steps 1 through 8 in order.

---

## GCP Services

| Service | Role |
|---|---|
| Vertex AI Vector Search 2.0 | Stores DataObjects (metadata + 768-dim vectors), ScaNN ANN search |
| Vertex AI text-embedding-005 | Generates 768-dim vectors at ingestion and query time |
| Vertex AI Gemini 2.5 Flash | Answer generation + agentic function calling (ReAct loop) |
| Cloud Storage | Raw CSV staging + evaluation report storage |
| Cloud Memorystore Redis | Semantic result cache — 0.5–2ms reads, TTL-based expiry |
| Serverless VPC Access | Bridge between Cloud Run and Memorystore private VPC |
| Cloud Run | Hosts FastAPI API + Streamlit UI (scales to zero) |
| Artifact Registry | Stores Docker images built by Cloud Build |
| Cloud Build | Builds AMD64 images on GCP — no local Docker required |

---

## Configuration Reference

All settings are read from environment variables. See `.env.example` for the full list.

| Variable | Default | Description |
|---|---|---|
| `GCP_PROJECT_ID` | *(required)* | Your GCP project ID |
| `GCS_BUCKET_NAME` | *(required)* | GCS bucket for data + eval reports |
| `GCP_REGION` | `us-central1` | GCP region |
| `COLLECTION_ID` | `airbnb-listings-collection` | VS2.0 Collection name |
| `EMBEDDING_MODEL` | `text-embedding-005` | 768-dim dense embedding model |
| `GEMINI_MODEL` | `gemini-2.5-flash` | LLM for generation + function calling |
| `MAX_LISTINGS` | `3000` | Listings to ingest (0 = all) |
| `BATCH_SIZE` | `50` | DataObjects per `batch_create` call |
| `TOP_K` | `10` | Default ANN results per query |
| `REDIS_HOST` | *(empty)* | Memorystore private IP — auto-set by `04_deploy.sh`; leave empty locally |
| `CACHE_RAG_TTL` | `3600` | `/rag` result TTL in seconds |
| `CACHE_ASK_TTL` | `1800` | `/ask` result TTL in seconds |
| `RAG_API_URL` | `http://localhost:8000` | API base URL used by the UI and eval scripts |

---

## VS2.0 DataObject Structure

```
DataObject
├── data_object_id  →  SHA1("airbnb:{listing_id}")
├── data            →  {
│     name, neighbourhood, property_type, room_type,
│     accommodates, bedrooms, price, rating,
│     instant_bookable, listing_url, host_name, text
│   }
└── vectors         →  { embedding: [768 floats] }
```

> **Note — two-step fetch pattern (VS2.0 beta):** `SearchDataObjects` returns IDs + distances only. Metadata is fetched in a separate parallel `GetDataObject` call per result. Both `app/rag.py` and `agent/tools.py` use `ThreadPoolExecutor` for parallel fetching.

---

## IAM Note

`roles/aiplatform.user` grants **zero** `vectorsearch.*` permissions — these are a separate IAM namespace. The setup script (`run/01_setup_gcp.sh`) creates a custom `vectorSearchUser` project role with the required permissions and grants it to the service account automatically.

---

## License

MIT
