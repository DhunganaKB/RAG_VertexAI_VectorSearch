"""
Airbnb Agentic RAG — Locust Load Test
======================================
Tests the deployed Cloud Run API with three realistic user types.

Usage:
    # 1. Export your Cloud Run API URL
    export RAG_API_URL=$(gcloud run services describe airbnb-rag-api \
        --region=YOUR_REGION --project=YOUR_PROJECT_ID \
        --format="value(status.url)")

    # 2. Run with web dashboard (interactive)
    locust -f load_test/locustfile.py --host=$RAG_API_URL

    # 3. Run headless (CI/CD or scripted)
    locust -f load_test/locustfile.py \
        --host=$RAG_API_URL \
        --headless \
        --users=20 \
        --spawn-rate=2 \
        --run-time=5m \
        --html=load_test/results/report.html

    # 4. Quick smoke test (5 users, 60s)
    locust -f load_test/locustfile.py \
        --host=$RAG_API_URL \
        --headless --users=5 --spawn-rate=1 --run-time=60s

Architecture notes:
    - /rag   → embed (Vertex AI) + ANN search (VS2.0) + Gemini answer  ≈ 800–3000ms cold
    - /ask   → Gemini function calling (1–2 rounds) + VS2.0 search     ≈ 2000–6000ms cold
    - Redis  → cache HIT for repeat queries                             ≈ 20–80ms
    - Cloud Run min-instances=0 → first request has a cold start (~5s)

Test users:
    HealthUser   — lightweight poller (health + cache stats)
    RAGUser      — calls POST /rag with semantic queries
    AgentUser    — calls POST /ask with structured filter queries
    MixedUser    — realistic mix: 60% /rag, 30% /ask, 10% health
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Allow running from project root OR from load_test/ directory
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from locust import HttpUser, between, constant_pacing, events, task
from locust.runners import MasterRunner, WorkerRunner

# ---------------------------------------------------------------------------
# Query pools — drawn from the evaluation dataset
# Balanced across all 5 categories so every test run covers the full spectrum
# ---------------------------------------------------------------------------

# Semantic queries → /rag  (no hard filters, purely vector similarity)
SEMANTIC_QUERIES = [
    "cozy place to relax with a great view",
    "stylish modern apartment close to restaurants and nightlife",
    "quiet retreat away from the city noise, good for working remotely",
    "romantic getaway with a pool",
    "pet-friendly home with a yard",
    "loft with exposed brick and great natural light",
    "cozy studio walking distance to coffee shops",
    "hip neighbourhood with great food scene",
    "peaceful place with a nice outdoor area",
    "bright spacious home with a home office setup",
]

# Price-bounded queries → /ask  (agent must extract max_price / min_price)
PRICE_QUERIES = [
    "find me a private room under $60 a night",
    "entire apartment between $80 and $150 per night",
    "cheap place to stay, max $50 per night",
    "luxury stay over $300 a night with great ratings",
    "affordable entire home for a week, budget around $100 max",
    "something under $75 per night for two people",
    "budget stay, anything under $45",
    "mid-range place around $120 a night",
]

# Room type queries → /ask  (agent must extract room_type, instant_bookable)
ROOM_TYPE_QUERIES = [
    "I need a private room in someone's home, not the whole place",
    "I want the whole apartment to myself",
    "shared room for a solo traveller on a tight budget",
    "private room that can instantly book without waiting for host approval",
    "entire home that sleeps at least 6 people",
    "instant bookable entire apartment",
    "private room with its own bathroom",
]

# Multi-filter queries → /ask  (hardest — 3–4 simultaneous filters)
MULTI_FILTER_QUERIES = [
    "2-bedroom entire home under $120 near downtown Austin",
    "instant bookable private room under $75 with at least 3 bedrooms",
    "family house in East Austin, at least 3 bedrooms, under $200",
    "superhost entire apartment for 4 guests under $150 instantly bookable",
    "cheap studio or private room in South Austin for one person",
    "entire home with 2+ bedrooms under $180, instant book",
]

# Neighbourhood queries → /ask (agent must extract neighbourhood)
NEIGHBOURHOOD_QUERIES = [
    "anything available in East Austin",
    "I want to stay near the University of Texas campus",
    "listings in the Rainey Street or 6th Street area",
    "South Congress neighbourhood — cozy place to stay",
    "North Loop or Hyde Park area, entire home under $130",
    "something in South Lamar",
    "East Austin neighbourhood, modern place",
]

# Warm-up queries — small set used intentionally to test Redis cache HITs
# Run these repeatedly to demonstrate that the second call is much faster
WARMUP_QUERIES = [
    "cozy place to relax with a great view",
    "find me a private room under $60 a night",
    "2-bedroom entire home under $120 near downtown Austin",
]


# ---------------------------------------------------------------------------
# Shared counters (accessible in test reports)
# ---------------------------------------------------------------------------
class Stats:
    cache_hits   = 0
    cache_misses = 0
    rag_calls    = 0
    ask_calls    = 0
    errors       = 0


# ---------------------------------------------------------------------------
# Helper: extract cache hit/miss from response header (if set by API)
# ---------------------------------------------------------------------------
def _track_cache(response) -> None:
    """Check X-Cache-Hit header if the API sets it; otherwise skip."""
    hit = response.headers.get("X-Cache-Hit", "").lower()
    if hit == "true":
        Stats.cache_hits += 1
    elif hit == "false":
        Stats.cache_misses += 1


# ============================================================================
# User Type 1: HealthUser
# Lightweight — just polls /health and /cache/stats.
# Use this to verify the API stays alive under load without adding LLM cost.
# ============================================================================
class HealthUser(HttpUser):
    """
    Simulates a monitoring probe that checks API health.
    Very lightweight — no Gemini or VS2.0 calls.
    Useful as a baseline to measure infrastructure overhead.
    """
    weight    = 1                   # 1 out of every N users is a HealthUser
    wait_time = between(5, 15)      # wait 5–15s between requests (monitoring cadence)

    @task(3)
    def check_health(self):
        """GET /health — verifies the API is up and cache is connected."""
        with self.client.get("/health", catch_response=True) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") != "ok":
                    resp.failure(f"Unexpected status: {data.get('status')}")
            else:
                resp.failure(f"Health check failed: {resp.status_code}")

    @task(1)
    def check_cache_stats(self):
        """GET /cache/stats — retrieves Redis hit/miss counters."""
        with self.client.get("/cache/stats", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"Cache stats failed: {resp.status_code}")

    @task(1)
    def check_root(self):
        """GET / — verifies the root endpoint (service info)."""
        with self.client.get("/", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"Root endpoint failed: {resp.status_code}")


# ============================================================================
# User Type 2: RAGUser
# Calls POST /rag — semantic search only, no agent loop.
# Moderate cost — 1 embedding call + 1 ANN search + 1 Gemini call per request.
# ============================================================================
class RAGUser(HttpUser):
    """
    Simulates users doing pure semantic search (no structured filters).
    Maps to the /rag endpoint.

    Wait strategy: between(10, 30) seconds — realistic think time between searches.
    First queries will be MISS (cold cache), repeat queries will be HIT.
    """
    weight    = 3                   # 3x more RAGUsers than HealthUsers
    wait_time = between(10, 30)     # realistic think time — don't hammer Gemini

    def on_start(self):
        """Called once when user starts — warm up with a health check."""
        self.client.get("/health")

    @task(7)
    def semantic_search_unique(self):
        """
        POST /rag with a randomly chosen semantic query.
        First call = cache MISS (calls Vertex AI + Gemini).
        Subsequent calls with same query = cache HIT (Redis, ~20–80ms).
        """
        query = random.choice(SEMANTIC_QUERIES)
        Stats.rag_calls += 1

        with self.client.post(
            "/rag",
            json={"query": query, "top_k": 10},
            name="/rag [semantic]",      # name groups the metric in Locust UI
            catch_response=True,
            timeout=60,
        ) as resp:
            _track_cache(resp)
            if resp.status_code == 200:
                data = resp.json()
                if not data.get("answer"):
                    resp.failure("Empty answer in /rag response")
            else:
                Stats.errors += 1
                resp.failure(f"/rag returned {resp.status_code}: {resp.text[:200]}")

    @task(3)
    def semantic_search_warmup(self):
        """
        POST /rag using a small fixed pool of queries (WARMUP_QUERIES).
        After the first user runs these, all subsequent calls will be cache HITs.
        This validates Redis caching is working correctly under concurrent load.
        """
        query = random.choice(WARMUP_QUERIES)
        Stats.rag_calls += 1

        with self.client.post(
            "/rag",
            json={"query": query, "top_k": 10},
            name="/rag [cache-warmup]",  # separate metric name for cache-HIT requests
            catch_response=True,
            timeout=60,
        ) as resp:
            _track_cache(resp)
            if resp.status_code != 200:
                Stats.errors += 1
                resp.failure(f"/rag warmup returned {resp.status_code}")


# ============================================================================
# User Type 3: AgentUser
# Calls POST /ask — full agentic flow (Gemini function calling).
# High cost — 2 Gemini calls per request + VS2.0 search.
# Use lower concurrency than RAGUser to avoid Gemini rate limits.
# ============================================================================
class AgentUser(HttpUser):
    """
    Simulates users making complex structured queries.
    Maps to the /ask endpoint (agentic, function-calling loop).

    Wait strategy: between(20, 60) seconds — agentic queries take longer,
    and users typically think more before asking complex questions.
    """
    weight    = 2                   # fewer AgentUsers to respect Gemini quotas
    wait_time = between(20, 60)     # longer wait — /ask is expensive

    def on_start(self):
        """Verify the API is healthy before starting agent queries."""
        self.client.get("/health")

    @task(3)
    def price_query(self):
        """POST /ask with a price-bounded query — agent must extract max/min_price."""
        query = random.choice(PRICE_QUERIES)
        Stats.ask_calls += 1

        with self.client.post(
            "/ask",
            json={"query": query},
            name="/ask [price]",
            catch_response=True,
            timeout=120,            # /ask can take up to 2 min on cache miss
        ) as resp:
            _track_cache(resp)
            if resp.status_code == 200:
                data = resp.json()
                if not data.get("answer"):
                    resp.failure("Empty answer in /ask response")
                elif not data.get("tool_calls"):
                    resp.failure("No tool_calls in /ask response — agent did not run")
            else:
                Stats.errors += 1
                resp.failure(f"/ask price returned {resp.status_code}: {resp.text[:200]}")

    @task(2)
    def room_type_query(self):
        """POST /ask with a room_type query — agent must extract room_type."""
        query = random.choice(ROOM_TYPE_QUERIES)
        Stats.ask_calls += 1

        with self.client.post(
            "/ask",
            json={"query": query},
            name="/ask [room_type]",
            catch_response=True,
            timeout=120,
        ) as resp:
            _track_cache(resp)
            if resp.status_code == 200:
                data = resp.json()
                if not data.get("answer"):
                    resp.failure("Empty answer in /ask response")
            else:
                Stats.errors += 1
                resp.failure(f"/ask room_type returned {resp.status_code}")

    @task(3)
    def neighbourhood_query(self):
        """POST /ask with a neighbourhood query."""
        query = random.choice(NEIGHBOURHOOD_QUERIES)
        Stats.ask_calls += 1

        with self.client.post(
            "/ask",
            json={"query": query},
            name="/ask [neighbourhood]",
            catch_response=True,
            timeout=120,
        ) as resp:
            _track_cache(resp)
            if resp.status_code != 200:
                Stats.errors += 1
                resp.failure(f"/ask neighbourhood returned {resp.status_code}")

    @task(2)
    def multi_filter_query(self):
        """
        POST /ask with a multi-filter query — hardest for the agent (3–4 filters).
        Expect higher latency and occasional failures if Gemini misses a filter.
        """
        query = random.choice(MULTI_FILTER_QUERIES)
        Stats.ask_calls += 1

        with self.client.post(
            "/ask",
            json={"query": query},
            name="/ask [multi_filter]",
            catch_response=True,
            timeout=120,
        ) as resp:
            _track_cache(resp)
            if resp.status_code == 200:
                data = resp.json()
                tc = data.get("tool_calls", [])
                if not tc:
                    resp.failure("No tool_calls — agent skipped filter extraction on multi-filter query")
            else:
                Stats.errors += 1
                resp.failure(f"/ask multi_filter returned {resp.status_code}")


# ============================================================================
# User Type 4: MixedUser  ← USE THIS FOR REALISTIC PRODUCTION SIMULATION
# Realistic mix: 60% /rag (browse), 30% /ask (complex search), 10% health
# ============================================================================
class MixedUser(HttpUser):
    """
    Realistic production traffic mix.
    Most users do simple semantic searches (/rag), fewer do complex agent queries (/ask).
    Use this user type for the primary load test scenario.

    Weight: set to 0 by default — enable by commenting out HealthUser/RAGUser/AgentUser
            and adjusting weight here, or use --tags mixed when running locust.
    """
    weight    = 0                   # disabled by default; enable for realistic simulation
    wait_time = between(8, 25)

    def on_start(self):
        self.client.get("/health")

    @task(6)
    def rag_browse(self):
        """60% of requests: semantic browse → /rag."""
        all_rag = SEMANTIC_QUERIES + WARMUP_QUERIES
        query   = random.choice(all_rag)

        with self.client.post(
            "/rag",
            json={"query": query, "top_k": 10},
            name="/rag [mixed-browse]",
            catch_response=True,
            timeout=60,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"/rag browse: {resp.status_code}")

    @task(3)
    def ask_structured(self):
        """30% of requests: structured agent query → /ask."""
        all_ask = PRICE_QUERIES + ROOM_TYPE_QUERIES + NEIGHBOURHOOD_QUERIES
        query   = random.choice(all_ask)

        with self.client.post(
            "/ask",
            json={"query": query},
            name="/ask [mixed-structured]",
            catch_response=True,
            timeout=120,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"/ask structured: {resp.status_code}")

    @task(1)
    def poll_health(self):
        """10% of requests: lightweight health check."""
        self.client.get("/health", name="/health [mixed]")


# ============================================================================
# Hooks: print a summary line after the test completes
# ============================================================================
@events.quitting.add_listener
def on_quit(environment, **kwargs):
    total = Stats.rag_calls + Stats.ask_calls
    if total == 0:
        return
    print("\n" + "=" * 60)
    print("  Load Test Summary")
    print("=" * 60)
    print(f"  /rag calls   : {Stats.rag_calls}")
    print(f"  /ask calls   : {Stats.ask_calls}")
    print(f"  Total API    : {total}")
    print(f"  Errors       : {Stats.errors}  ({Stats.errors/total*100:.1f}%)" if total else "")
    print(f"  Cache HITs   : {Stats.cache_hits}  (requires X-Cache-Hit header)")
    print(f"  Cache MISSes : {Stats.cache_misses}")
    print("=" * 60)
