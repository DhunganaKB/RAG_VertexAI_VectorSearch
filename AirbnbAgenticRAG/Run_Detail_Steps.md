# AirbnbAgenticRAG

**Run Guide — From blank GCP project to running application**

## Badges
- Vertex AI VS2.0
- Gemini 2.5 Flash
- Cloud Memorystore Redis
- Cloud Run
- FastAPI
- Streamlit

---

## Project Overview

AirbnbAgenticRAG is a Retrieval-Augmented Generation system for searching 3,000 Austin TX Airbnb listings. It exposes two API endpoints with different retrieval strategies:

| Endpoint | Strategy | Best for |
|----------|----------|----------|
| `POST /rag` | Simple RAG — embed → ANN search → Gemini | Open-ended, preference-based queries |
| `POST /ask` | Agentic RAG — extract filters → tool call → filtered results → Gemini | Structured queries (price, bedrooms, neighbourhood) |

Both endpoints share a **Redis cache** — a cache HIT returns the full response in <2ms, bypassing embeddings, Vector Search, and Gemini entirely.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  OFFLINE  (run once)                     │
│                                                          │
│  GCS Bucket  →  02_ingest.py  →  VS2.0 Collection        │
│                 (embed + batch_create_data_objects)       │
│                                                          │
│  03_create_index.py  →  ScaNN ANN Index                  │
└──────────────────────────────────────────────────────────┘

┌──────────────┐   ┌──────────────────────────────────────┐
│  Streamlit   │   │          ONLINE                      │
│     UI       │──▶│  FastAPI  /rag  or  /ask             │
│  (port 8501) │◀──│    │                                 │
└──────────────┘   │    ├─▶ Redis HIT  ──▶ return         │
                   │    │                                 │
                   │    └─▶ MISS: embed (text-emb-005)    │
                   │              │                       │
                   │         VS2.0 ScaNN ANN search       │
                   │              │                       │
                   │         GetDataObject ×N (parallel)  │
                   │              │                       │
                   │    /rag: build context → Gemini      │
                   │    /ask: filter → ReAct loop         │
                   │              │                       │
                   │         Redis SET (SETEX + TTL)      │
                   └──────────────────────────────────────┘
```

---

## GCP Services Used

| Service | Role | Cost |
|---------|------|------|
| Cloud Storage | Raw CSV staging + eval report storage | ~$0.02/GB/mo |
| VS2.0 Collection | Stores DataObjects (metadata + 768-dim vectors) | Pay per vector |
| VS2.0 ScaNN Index | ANN search — sub-10ms at scale | Included |
| Vertex AI Embeddings | text-embedding-005 — 768-dim vectors | $0.000025/1k chars |
| Gemini 2.5 Flash | LLM for answer generation + agentic function calling | Pay per token |
| Cloud Memorystore | Redis cache — 0.5–2ms latency via VPC connector | ~$40–80/mo (BASIC 1GB) |
| Cloud Run | Serverless containers — FastAPI + Streamlit | Free tier: 2M req/mo |
| Artifact Registry | Docker image storage for Cloud Run | ~$0.10/GB/mo |

---

## Prerequisites

Install these tools on your local machine before running any scripts.

| Tool | Version | Install |
|------|---------|---------|
| `gcloud` | ≥ 470.0.0 | `brew install google-cloud-sdk` |
| `python` | ≥ 3.10 | Pre-installed, or use conda / `brew install python` |
| `pip` | latest | `pip install -r requirements.txt` |
| `locust` | ≥ 2.28 | `pip install locust` (load testing only) |

> **ℹ️ No local Docker required**
> `run/04_deploy.sh` uses Cloud Build to build images on GCP. You do not need Docker Desktop installed locally. This also avoids Apple Silicon ARM64/AMD64 issues.

---

## Authentication

Authenticate once. All scripts and app code use Application Default Credentials (ADC).

```bash
# Log in with ADC — used by all scripts and the app locally
gcloud auth application-default login

# Set your default project
gcloud config set project YOUR_PROJECT_ID

# Verify auth is working
gcloud auth application-default print-access-token | head -c 40
```

> **⚠️ Redis is not reachable from your local machine**
> Memorystore runs on a private VPC IP (e.g. `10.x.x.x`). It is only reachable from Cloud Run via the VPC connector. Locally, the app detects an empty `REDIS_HOST` and disables cache silently — the app works normally, just without caching.

---

## Project Structure

Each directory has one job. The `run/` directory is where you spend most of your time.

```
AirbnbAgenticRAG/
│
├── config.py                  ← Single source of truth — reads all values from env vars
├── .env.example               ← Template: copy to .env and fill in your GCP values
├── .gitignore                 ← Excludes .env, __pycache__, local data, secrets
│
├── app/                       ← FastAPI backend (online query layer)
│   ├── main.py                   /rag and /ask endpoints + cache check/write
│   ├── rag.py                    VS2.0 retriever + context builder
│   └── cache.py                  Redis helpers (get_cached, set_cached, keys)
│
├── agent/                     ← Agentic RAG logic
│   ├── agent.py                  ReAct loop — Gemini + tool calling (max 5 turns)
│   └── tools.py                  find_rentals() — over-fetch + Python-side filter
│
├── scripts/                   ← Offline data pipeline (run once)
│   ├── 00_upload_data.py         Upload local CSV to GCS
│   ├── 01_setup_collection.py    Create VS2.0 Collection + schema
│   ├── 02_ingest.py              Embed + ingest 3,000 listings as DataObjects
│   ├── 03_create_index.py        Build ScaNN ANN index (server-side LRO)
│   ├── 04_build_pipeline.py      Runs stages 1–3 in sequence (convenience wrapper)
│   └── inspect_collection.py     Show collection count, index status, sample objects
│
├── eval/                      ← Evaluation (RAGAS-style scoring)
│   ├── dataset.json              25 labeled queries across 5 categories
│   ├── evaluate.py               Runner: calls API → scores → uploads to GCS
│   ├── metrics.py                Pure scoring functions (no I/O)
│   └── report.py                 HTML report builder
│
├── load_test/                 ← Locust load testing
│   ├── locustfile.py             User scenarios (HealthUser, RAGUser, AgentUser)
│   └── locust.conf               Default settings (users=20, 5 min)
│
├── ui/                        ← Streamlit frontend
├── notebooks/                 ← Jupyter walkthroughs (ingestion, RAG, eval)
├── docs/                      ← Architecture diagrams (HTML)
│
├── run/                       ← Operation scripts — run these in order
│   ├── 01_setup_gcp.sh           One-time GCP setup
│   ├── 02_run_pipeline.sh        Data ingestion pipeline
│   ├── 03_run_local.sh           Start API + UI locally
│   ├── 04_deploy.sh              Build + deploy to Cloud Run
│   ├── 05_verify.sh              Health check + endpoint tests
│   ├── 06_run_eval.sh            RAGAS evaluation
│   ├── 07_run_loadtest.sh        Locust load tests
│   └── 08_ops.sh                 Cache, Redis, re-ingestion operations
│
├── Dockerfile.api             ← FastAPI container image
├── Dockerfile.ui              ← Streamlit container image
├── requirements.txt           ← All dependencies (dev + runtime)
├── requirements-api.txt       ← API-only dependencies (used in Dockerfile.api)
└── requirements-ui.txt        ← UI-only dependencies (used in Dockerfile.ui)
```

---

## Configuration — .env Setup

All project-specific values live in a `.env` file. Create it once from the template before running any step.

```bash
# Copy the template
cp .env.example .env

# Edit with your values — minimum required fields:
GCP_PROJECT_ID=your-gcp-project-id
GCS_BUCKET_NAME=your-gcs-bucket-name
```

### Full list of available settings:

| Variable | Required | Description |
|----------|----------|-------------|
| `GCP_PROJECT_ID` | ✅ Yes | Your GCP project ID |
| `GCS_BUCKET_NAME` | ✅ Yes | GCS bucket for data + eval reports |
| `GCP_REGION` | Optional | Defaults to `us-central1` |
| `COLLECTION_ID` | Optional | Defaults to `airbnb-listings-collection` |
| `EMBEDDING_MODEL` | Optional | Defaults to `text-embedding-005` |
| `GEMINI_MODEL` | Optional | Defaults to `gemini-2.5-flash` |
| `REDIS_HOST` | Optional | Leave empty locally — cache is auto-disabled |
| `RAG_API_URL` | Optional | Defaults to `http://localhost:8000` |

> **⚠️ Never commit .env to git**
> `.env` is listed in `.gitignore`. Only commit `.env.example` (placeholder values, no credentials). `config.py` reads all sensitive values from environment variables at runtime.

> **💡 One config, multiple environments**
> The same code runs locally (reads from `.env`), on Cloud Run (reads from service env vars), and in any other environment — without any code changes.

---

## Step 1 — GCP Setup

**Script:** `run/01_setup_gcp.sh` · **Run once** · **~10 min**

Provisions all GCP infrastructure needed for the project. This is a one-time step per project — safe to re-run (all resource creation is idempotent).

```bash
./run/01_setup_gcp.sh
```

### What this script does:

- Enables all required GCP APIs (including `vectorsearch.googleapis.com` separately from `aiplatform.googleapis.com`)
- Creates Artifact Registry Docker repository for container images
- Creates service account `airbnb-rag-sa` with all required IAM roles
- Creates custom `vectorSearchUser` IAM role (required — see IAM Note)
- Provisions Memorystore Redis instance (BASIC 1GB, Redis 7.x) and waits for READY
- Creates Serverless VPC Access connector and waits for READY

> **✓ Idempotent — safe to re-run**
> Every resource is guarded by an existence check. Already-existing resources are skipped without error.

---

## Step 2 — Run Data Pipeline

**Script:** `run/02_run_pipeline.sh` · **Run once** · **~15–45 min**

Uploads the dataset to GCS, creates the Vector Search 2.0 collection, embeds and ingests all 3,000 listings, then builds the ScaNN ANN index. The app cannot serve queries until this step completes.

```bash
# Full pipeline: upload → collection → ingest 3,000 listings → ScaNN index
./run/02_run_pipeline.sh

# Quick test — ingest only 100 listings (~3 min, no ScaNN index)
./run/02_run_pipeline.sh --quick

# Skip index build — uses exact kNN instead of ANN (fine for dev/testing)
./run/02_run_pipeline.sh --skip-index

# Run full pipeline then inspect the collection
./run/02_run_pipeline.sh --verify
```

### Pipeline stages:

| Stage | Script | What happens | Time |
|-------|--------|--------------|------|
| 0 | `00_upload_data.py` | Upload local CSV → GCS (skipped if already there) | ~1 min |
| 1 | `01_setup_collection.py` | Create VS2.0 Collection with data schema + 768-dim vector schema | ~10 sec |
| 2 | `02_ingest.py` | Download CSV → parse → embed (text-embedding-005) → batch upload DataObjects | ~8–12 min |
| 3 | `03_create_index.py` | Build ScaNN ANN index on the embedding field (server-side long-running operation) | ~5–20 min |

> **⚠️ Embedding rate limits**
> `text-embedding-005` has a QPM quota. The pipeline uses `BATCH_SIZE=50` with exponential backoff on 429 errors. If you hit limits, set a lower `MAX_LISTINGS` value in your `.env`.

```bash
# After the pipeline completes — inspect the collection
python scripts/inspect_collection.py

# Start fresh (delete collection and re-run)
python scripts/00_delete_collections.py
```

---

## Step 3 — Run Locally

**Script:** `run/03_run_local.sh` · **Optional development step**

Starts the FastAPI backend and Streamlit UI on your local machine for development and testing before deploying to Cloud Run.

```bash
# Install dependencies first (one time)
pip install -r requirements.txt

# Start both API (port 8000) and UI (port 8501)
./run/03_run_local.sh

# Start API only
./run/03_run_local.sh --api

# Start UI only (API must already be running)
./run/03_run_local.sh --ui
```

### Local URLs:

| Service | URL | Notes |
|---------|-----|-------|
| FastAPI backend | `http://localhost:8000` | Auto-reloads on code changes |
| Swagger docs | `http://localhost:8000/docs` | Try endpoints interactively |
| Streamlit UI | `http://localhost:8501` | Reads `RAG_API_URL` from `.env` |

> **ℹ️ Redis cache is disabled locally — this is expected**
> `REDIS_HOST` is empty in your local `.env`, so the app prints *"cache disabled"* and continues normally. No errors, no crashes — queries hit Vector Search directly every time.

---

## Step 4 — Deploy to Cloud Run

**Script:** `run/04_deploy.sh` · **~5–10 min**

Builds Docker images using Cloud Build (no local Docker required), then deploys both the API and UI as serverless Cloud Run services. All required environment variables including `GCP_PROJECT_ID` and `REDIS_HOST` are automatically passed to Cloud Run.

```bash
# Build and deploy both API + UI
./run/04_deploy.sh

# Re-deploy API only after code changes
./run/04_deploy.sh --api

# Re-deploy UI only
./run/04_deploy.sh --ui
```

### Cloud Run services:

| Service | Image | Memory | VPC connector |
|---------|-------|--------|---------------|
| `airbnb-rag-api` | Dockerfile.api | 2Gi | Yes — for Redis access |
| `airbnb-rag-ui` | Dockerfile.ui | 1Gi | No — calls API over HTTPS |

> **⚠️ Apple Silicon Mac — use Cloud Build, not local Docker**
> Building locally on Apple Silicon produces ARM64 images that crash on Cloud Run with `exec format error`. `04_deploy.sh` always uses Cloud Build which runs on linux/amd64 — no `--platform` flag needed.

---

## Step 5 — Verify Deployment

**Script:** `run/05_verify.sh`

Runs a sequence of checks against the deployed Cloud Run API: health check → `/rag` test → `/ask` test → cache HIT/MISS timing comparison → cache stats summary.

```bash
./run/05_verify.sh
```

### Expected health check output:

```json
{
  "status": "ok",
  "collection": "projects/YOUR_PROJECT_ID/.../airbnb-listings-collection",
  "cache": {
    "status": "connected",
    "host": "10.x.x.x",
    "hits": 0,
    "misses": 0
  }
}
```

---

## Step 6 — Evaluation

**Script:** `run/06_run_eval.sh` · **~3–8 min**

Runs 25 labeled queries against the deployed Cloud Run API and scores each response using LLM-as-judge. Results are uploaded to GCS as an HTML report and opened locally.

```bash
# Full evaluation — 25 queries, both endpoints, Gemini-as-judge
./run/06_run_eval.sh

# Run /rag endpoint only
./run/06_run_eval.sh --rag

# Run /ask endpoint only
./run/06_run_eval.sh --ask

# Quick smoke — 5 queries, no LLM judge (fast, no Gemini cost)
./run/06_run_eval.sh --smoke

# Download and open the latest saved report
./run/06_run_eval.sh --view
```

### Evaluation metrics:

| Metric | Range | Description |
|--------|-------|-------------|
| `answer_relevance` | 0–4 | Gemini-as-judge rates the answer against the query |
| `filter_accuracy` | 0–1 | % of expected filters correctly extracted (`/ask` only) |
| `keyword_hit_rate` | 0–1 | % of expected keywords found in the answer |
| `has_results` | bool | Did the response return at least one listing? |
| `latency_ms` | float | Wall-clock time from request to response |

> **ℹ️ Second eval run is much faster**
> The first run populates Redis with all 25 query results. A second run returns near-instant cache HITs. Run `./run/08_ops.sh cache-flush` before re-evaluating to get fresh responses.

---

## Step 7 — Load Testing

**Script:** `run/07_run_loadtest.sh` · **~20 min full sequence**

Uses Locust to simulate concurrent users and measure throughput, latency percentiles, and error rates across different load levels. Also validates cache speedup by running the same queries twice.

```bash
# Full sequence: smoke → load → stress
./run/07_run_loadtest.sh

# Individual stages
./run/07_run_loadtest.sh --smoke    # 5 users, 60s — sanity check
./run/07_run_loadtest.sh --load     # 20 users, 5 min — standard load
./run/07_run_loadtest.sh --stress   # 50 users, 10 min — finding limits

# Run same queries twice and compare MISS vs HIT latency
./run/07_run_loadtest.sh --cache

# Open interactive Locust web dashboard at http://localhost:8089
./run/07_run_loadtest.sh --ui
```

### Performance benchmarks:

| Stage | /rag p50 latency | /ask p50 latency | Error rate |
|-------|------------------|------------------|------------|
| Smoke — cold (5 users) | 800–3000 ms | 2000–6000 ms | < 1% |
| Smoke — cached (5 users) | 20–80 ms | 20–80 ms | < 1% |
| Load (20 users) | 1000–4000 ms | 3000–8000 ms | < 2% |
| Stress (50 users) | 2000–8000 ms | 5000–15000 ms | < 5% |

---

## Step 8 — Operations

**Script:** `run/08_ops.sh` · **Ongoing**

Day-to-day maintenance commands for cache management, Redis lifecycle, re-ingestion, and log monitoring.

```bash
./run/08_ops.sh cache-stats    # view hit/miss counts and total cached keys
./run/08_ops.sh cache-flush    # clear all cached results (run after re-ingestion)
./run/08_ops.sh redis-stop     # delete Redis + VPC connector (saves ~$46/month)
./run/08_ops.sh redis-start    # re-create Redis + VPC connector (~10 min)
./run/08_ops.sh reindex        # delete collection + re-run full pipeline
./run/08_ops.sh logs           # tail Cloud Run API logs live
```

---

## Cache Management

Both `/rag` and `/ask` follow the same cache pattern in `app/main.py`:

1. Build a deterministic key: `rag_key(query, top_k)` or `ask_key(query)`
2. Check cache: `get_cached(key)` → return immediately on HIT (<2ms)
3. On MISS: run full pipeline (embed → VS search → Gemini)
4. Store result: `set_cached(key, result, ttl=CACHE_RAG_TTL)`

### Cache TTL settings:

| Endpoint | TTL | Key format |
|----------|-----|------------|
| `/rag` | 1 hour | `rag:<sha256(query+top_k)[:32]>` |
| `/ask` | 30 min | `ask:<sha256(query)[:32]>` |

> **⚠️ Always flush cache after re-ingestion**
> Cached responses reference old listing data. Run `./run/08_ops.sh cache-flush` immediately after `./run/08_ops.sh reindex` completes.

---

## Redis Cost Management

Memorystore Redis costs ~$40–55/month even when idle. When pausing the project, stop it to avoid charges — the app continues to work without cache (queries just take longer).

```bash
# Stop Redis — saves ~$46/month, app continues without cache
./run/08_ops.sh redis-stop

# Re-enable Redis when needed (~10 min to provision)
./run/08_ops.sh redis-start
```

> **✓ Monthly cost with Redis stopped**
> Cloud Run: **$0** (scales to zero) · VS2.0 (3k vectors): **~$2–5** · GCS: **<$0.01** · Total: **~$2–5/month**

---

## Cost Estimates

| Scenario | Monthly cost |
|----------|--------------|
| Idle (no traffic) — Redis running | ~$45–60 |
| Idle (no traffic) — Redis stopped | ~$2–5 |
| Active development (moderate traffic) | ~$60–100 |
| Production with real users | ~$100–200+ |

---

## IAM Note — Vector Search 2.0

> **⚠️ roles/aiplatform.user grants ZERO vectorsearch.* permissions**
> `vectorsearch.googleapis.com` is a separate API namespace from `aiplatform.googleapis.com`. The predefined `roles/aiplatform.user` has no `vectorsearch.*` permissions. Without the custom role you will get:
> `403 Permission 'vectorsearch.dataObjects.search' denied`
>
> The `01_setup_gcp.sh` script creates the custom `vectorSearchUser` project role and grants it to the service account automatically.

### Required IAM roles:

| Role | Why needed |
|------|------------|
| `roles/aiplatform.user` | Embeddings (text-embedding-005), Gemini LLM calls |
| `projects/…/roles/vectorSearchUser` | VS2.0 search, get & create DataObjects — custom role, no predefined equivalent |
| `roles/storage.objectViewer` | Read CSV from GCS during ingestion |
| `roles/storage.objectCreator` | Write eval reports to GCS |
| `roles/artifactregistry.reader` | Cloud Run pulls Docker images from Artifact Registry |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `403 vectorsearch.dataObjects.search denied` | Custom IAM role missing | Re-run `./run/01_setup_gcp.sh` |
| Cloud Run 500 on all endpoints | `GCP_PROJECT_ID` not set in Cloud Run env vars | Re-run `./run/04_deploy.sh --api` — it sets all env vars automatically |
| Cloud Run container crashes with `exec format error` | ARM64 image built locally on Apple Silicon | Use `./run/04_deploy.sh` — it uses Cloud Build (always AMD64) |
| Cache shows `"status": "unavailable"` | REDIS_HOST not set or VPC connector not attached | Re-run `./run/04_deploy.sh --api` — auto-reads Redis IP and passes it to Cloud Run |
| Redis unreachable from local machine | Memorystore uses private VPC IP only | Expected — cache is disabled locally. Use Cloud Shell for direct Redis CLI access. |
| `/ask` returns 500 with "finish reason: 2" | Old vertexai SDK incorrectly flags function-call responses as safety blocks | Ensure `agent.py` uses `model.start_chat(response_validation=False)` |
| Eval returns `results=0` for all queries | Running against wrong server or collection not yet ingested | Health check must show `collection` + `cache` keys. Run Step 2 pipeline first. |
| 429 errors during ingestion or eval | Gemini or embedding API rate limit | Reduce `MAX_LISTINGS` in `.env` or increase quota in Cloud Console |
