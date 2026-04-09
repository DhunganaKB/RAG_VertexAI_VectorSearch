"""
Cloud Memorystore (Redis) Cache — Airbnb Agentic RAG
=====================================================
GCP-native semantic cache using Cloud Memorystore for Redis.

Latency comparison:
  Memorystore Redis  →  0.5 – 2 ms   (in-VPC, same region)
  Cloud Firestore    →  20 – 50 ms   (HTTP-based document store)
  No cache           →  300 – 1500 ms (embed + ANN search + Gemini)

Architecture on GCP:
  Cloud Run (service)
      │  Serverless VPC Access connector
      ▼
  VPC Network (us-central1)
      │  private IP  10.x.x.x:6379
      ▼
  Cloud Memorystore for Redis (BASIC or STANDARD tier)

Authentication:
  None needed — Redis is a private-IP service inside the VPC.
  Cloud Run reaches it through the Serverless VPC Access connector.
  No keys, no credentials, no ADC required for the cache itself.

Connection:
  REDIS_HOST  →  set as Cloud Run env var (private IP from Memorystore console)
  REDIS_PORT  →  6379  (Memorystore default, rarely changes)
  REDIS_DB    →  0

Serialisation:
  Values are JSON-encoded strings stored as Redis STRINGs with SETEX (TTL).
  Keys are SHA-256 prefixed with endpoint tag for namespace isolation:
    rag:<32-char-hex>   →  /rag results  (TTL: CACHE_RAG_TTL)
    ask:<32-char-hex>   →  /ask results  (TTL: CACHE_ASK_TTL)

Failure behaviour:
  Any Redis error (connection refused, timeout) is caught and logged.
  The application continues without cache — never returns 5xx due to cache.

Pool sizing for Cloud Run:
  Cloud Run scales containers horizontally; each instance gets its own
  ConnectionPool. max_connections=10 per instance is safe for Memorystore
  BASIC tier (max 65,000 connections total).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)

# ── Lazy Redis connection pool ────────────────────────────────────────────────
_pool   = None
_failed = False   # once Redis is confirmed unavailable, stop retrying


def _get_pool():
    """
    Return a shared Redis ConnectionPool, or None if unavailable / disabled.
    Connection is attempted once; on failure _failed is set to True so we
    don't retry on every request (avoids per-request 2-second timeout stalls).
    """
    global _pool, _failed

    if not getattr(config, "CACHE_ENABLED", True):
        return None
    if _failed:
        return None
    if _pool is not None:
        return _pool

    host = getattr(config, "REDIS_HOST", "")
    if not host:
        logger.info("REDIS_HOST not set — cache disabled.")
        print("[Cache] REDIS_HOST not configured — cache disabled.")
        _failed = True
        return None

    try:
        import redis  # type: ignore
        pool = redis.ConnectionPool(
            host=host,
            port=getattr(config, "REDIS_PORT", 6379),
            db=getattr(config, "REDIS_DB", 0),
            decode_responses=True,          # return str, not bytes
            socket_connect_timeout=1.5,     # fail fast on cold-start
            socket_timeout=1.5,             # per-command timeout
            max_connections=10,             # per Cloud Run instance
            retry_on_timeout=False,         # don't block; skip cache on timeout
        )
        # Eager PING to validate the connection before the first request
        client = redis.Redis(connection_pool=pool)
        client.ping()
        _pool = pool
        logger.info("Memorystore Redis connected — %s:%s db=%s",
                    host, getattr(config, "REDIS_PORT", 6379),
                    getattr(config, "REDIS_DB", 0))
        print(f"[Cache] Memorystore Redis connected — {host}:{getattr(config, 'REDIS_PORT', 6379)}")
        return _pool
    except Exception as exc:
        _failed = True
        logger.warning("Memorystore Redis unavailable — running without cache: %s", exc)
        print(f"[Cache] Redis unavailable ({exc}) — cache disabled.")
        return None


def _client():
    """Return a Redis client from the shared pool, or None."""
    pool = _get_pool()
    if pool is None:
        return None
    try:
        import redis  # type: ignore
        return redis.Redis(connection_pool=pool)
    except Exception:
        return None


# ── Key builder ───────────────────────────────────────────────────────────────

def _make_key(prefix: str, payload: str) -> str:
    """
    Build a short, deterministic cache key.
    Format:  <prefix>:<sha256[:32]>
    Example: rag:3f1c8a2b9d0e4f7a1b2c3d4e5f6a7b8c
    """
    digest = hashlib.sha256(payload.encode()).hexdigest()[:32]
    return f"{prefix}:{digest}"


def rag_key(query: str, top_k: int) -> str:
    return _make_key("rag", f"{query.strip().lower()}|{top_k}")


def ask_key(payload: str) -> str:
    return _make_key("ask", payload.strip().lower())


# ── Public API ────────────────────────────────────────────────────────────────

def get_cached(key: str) -> Any | None:
    """
    Retrieve a cached value.
    Returns the deserialised Python object on HIT, None on MISS or error.
    Redis TTL handles expiry automatically — no in-code expiry check needed.
    """
    r = _client()
    if r is None:
        return None
    try:
        raw = r.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as exc:
        logger.warning("Cache GET error (key=%s): %s", key, exc)
        return None


def set_cached(key: str, value: Any, ttl: int,
               endpoint: str = "", query_preview: str = "") -> None:
    """
    Store *value* (must be JSON-serialisable) with *ttl* seconds expiry.
    Uses SETEX — atomic SET + EXPIRE in one command.
    Silently swallows all errors — cache failures must never break the app.
    The *endpoint* and *query_preview* params are accepted for API compatibility
    but not stored (Redis STRINGS don't need metadata fields).
    """
    r = _client()
    if r is None:
        return
    try:
        r.setex(key, ttl, json.dumps(value, ensure_ascii=False))
    except Exception as exc:
        logger.warning("Cache SET error (key=%s): %s", key, exc)


def invalidate(key: str) -> bool:
    """Delete a specific cache key. Returns True if it existed."""
    r = _client()
    if r is None:
        return False
    try:
        return bool(r.delete(key))
    except Exception:
        return False


def flush_all() -> int:
    """
    Flush ALL keys in the Redis DB (FLUSHDB).
    WARNING: this clears everything in db=REDIS_DB.
    Returns the number of keys that were present before flush.
    """
    r = _client()
    if r is None:
        return 0
    try:
        count = r.dbsize()
        r.flushdb(asynchronous=True)   # async flush — non-blocking
        logger.info("Cache flushed: ~%d keys removed.", count)
        return count
    except Exception as exc:
        logger.warning("Cache flush error: %s", exc)
        return 0


def cache_stats() -> dict:
    """
    Return cache statistics for the /health and /cache/stats endpoints.
    """
    enabled = getattr(config, "CACHE_ENABLED", True)
    host    = getattr(config, "REDIS_HOST", "")

    if not enabled:
        return {"enabled": False, "backend": "memorystore_redis", "status": "disabled"}
    if not host:
        return {"enabled": False, "backend": "memorystore_redis",
                "status": "REDIS_HOST not set"}

    r = _client()
    if r is None:
        return {"enabled": True, "backend": "memorystore_redis",
                "host": host, "status": "unavailable"}
    try:
        info  = r.info("server")
        stats = r.info("stats")
        mem   = r.info("memory")
        return {
            "enabled":         True,
            "backend":         "memorystore_redis",
            "status":          "connected",
            "host":            host,
            "port":            getattr(config, "REDIS_PORT", 6379),
            "db":              getattr(config, "REDIS_DB", 0),
            "redis_version":   info.get("redis_version"),
            "total_keys":      r.dbsize(),
            "used_memory_mb":  round(mem.get("used_memory", 0) / 1024 / 1024, 2),
            "hits":            stats.get("keyspace_hits", 0),
            "misses":          stats.get("keyspace_misses", 0),
            "rag_ttl_s":       getattr(config, "CACHE_RAG_TTL", 3600),
            "ask_ttl_s":       getattr(config, "CACHE_ASK_TTL", 1800),
        }
    except Exception as exc:
        return {"enabled": True, "backend": "memorystore_redis",
                "host": host, "status": f"error: {exc}"}
