"""
FastAPI Backend — Airbnb Agentic RAG
=====================================
Exposes two endpoints:

  POST /ask   { "query": "..." }
    → Agentic: Gemini + find_rentals tool (multi-turn function calling)
    → Returns natural-language answer + tool_calls log

  POST /rag   { "query": "...", "top_k": 10 }
    → Direct RAG: embed → VS2.0 SearchDataObjects → Gemini answer
    → Returns structured answer + sources list

Start:
    uvicorn app.main:app --reload --port 8000

Test:
    curl -s -X POST http://localhost:8000/ask \
      -H 'Content-Type: application/json' \
      -d '{"query": "cozy private room under $80 near downtown"}'
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import vertexai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from vertexai.generative_models import GenerationConfig, GenerativeModel

from agent.agent import run_agent
from app.cache import ask_key, cache_stats, flush_all, get_cached, rag_key, set_cached
from app.rag import VS2Retriever, build_context

app = FastAPI(
    title="Airbnb Agentic RAG — VS2.0 + Gemini",
    version="1.0.0",
    description=(
        "Agentic RAG for Austin Airbnb listings powered by "
        "Vertex AI Vector Search 2.0 and Gemini function calling."
    ),
)

# ── Lazy singletons ───────────────────────────────────────────────────────────
_retriever: VS2Retriever | None = None
_model:     GenerativeModel | None = None
_vertex_ok  = False


def _ensure_vertex() -> None:
    global _vertex_ok
    if not _vertex_ok:
        vertexai.init(project=config.PROJECT_ID, location=config.LOCATION)
        _vertex_ok = True


def get_retriever() -> VS2Retriever:
    global _retriever
    if _retriever is None:
        _ensure_vertex()
        _retriever = VS2Retriever(
            collection_resource=config.COLLECTION_RESOURCE,
            embedding_model_name=config.EMBEDDING_MODEL,
        )
    return _retriever


def get_model() -> GenerativeModel:
    global _model
    if _model is None:
        _ensure_vertex()
        _model = GenerativeModel(
            model_name=config.GEMINI_MODEL,
            generation_config=GenerationConfig(
                temperature=config.TEMPERATURE,
                max_output_tokens=config.MAX_OUTPUT_TOKENS,
            ),
        )
    return _model


# ── Request / Response schemas ────────────────────────────────────────────────

class AgentRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language rental query")

class AgentResponse(BaseModel):
    answer:     str
    tool_calls: list[dict]
    model:      str
    collection: str

class RagRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(None, ge=1, le=50)

class RagResponse(BaseModel):
    answer:     str
    sources:    list[dict]
    model:      str
    collection: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {
        "status":     "ok",
        "collection": config.COLLECTION_RESOURCE,
        "cache":      cache_stats(),
    }


@app.get("/cache/stats")
def get_cache_stats() -> dict:
    """Return Redis cache hit/miss statistics."""
    return cache_stats()


@app.post("/cache/flush")
def flush_cache() -> dict:
    """Flush the entire Redis cache (use after re-ingestion)."""
    ok = flush_all()
    return {"flushed": ok}


@app.get("/")
def root() -> dict:
    return {
        "service":  "Airbnb Agentic RAG — VS2.0 + Gemini",
        "endpoints": {
            "POST /ask": "Agentic query (Gemini + find_rentals tool)",
            "POST /rag": "Direct RAG query (embed → search → Gemini)",
        },
        "docs":       "/docs",
        "collection": config.COLLECTION_RESOURCE,
        "model":      config.GEMINI_MODEL,
    }


@app.post("/ask", response_model=AgentResponse)
def ask(req: AgentRequest) -> AgentResponse:
    """
    Agentic endpoint — Gemini decides which filters to apply and calls
    find_rentals() one or more times before producing the final answer.
    Results are cached in Memorystore Redis (TTL = CACHE_ASK_TTL).
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(400, "query must be non-empty")

    # ── Cache check ───────────────────────────────────────────────────────────
    key    = ask_key(query)
    cached = get_cached(key)
    if cached is not None:
        return AgentResponse(**cached)

    try:
        answer, tool_calls = run_agent(query)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

    result = AgentResponse(
        answer=answer,
        tool_calls=tool_calls,
        model=config.GEMINI_MODEL,
        collection=config.COLLECTION_RESOURCE,
    )
    set_cached(key, result.model_dump(), ttl=config.CACHE_ASK_TTL)
    return result


@app.post("/rag", response_model=RagResponse)
def rag(req: RagRequest) -> RagResponse:
    """
    Direct RAG endpoint — embed query → VS2.0 SearchDataObjects → Gemini answer.
    No agent loop; suitable for simple retrieval without filter reasoning.
    Results are cached in Memorystore Redis (TTL = CACHE_RAG_TTL).
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(400, "query must be non-empty")

    top_k = req.top_k or config.TOP_K

    # ── Cache check ───────────────────────────────────────────────────────────
    key    = rag_key(query, top_k)
    cached = get_cached(key)
    if cached is not None:
        return RagResponse(**cached)

    try:
        listings = get_retriever().search(query=query, top_k=top_k)
        context, sources = build_context(listings)

        prompt = f"""You are a helpful Airbnb rental assistant for Austin, Texas.
Answer the QUESTION using ONLY the listings in CONTEXT below.
If none of the listings match what the user is asking for, say so and suggest
what they could try instead.
Present matching listings clearly: name, location, price, size, rating, URL.

CONTEXT:
{context}

QUESTION:
{query}
"""
        resp = get_model().generate_content(prompt)
        answer = (resp.text or "").strip()

    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

    result = RagResponse(
        answer=answer,
        sources=sources,
        model=config.GEMINI_MODEL,
        collection=config.COLLECTION_RESOURCE,
    )
    set_cached(key, result.model_dump(), ttl=config.CACHE_RAG_TTL)
    return result
