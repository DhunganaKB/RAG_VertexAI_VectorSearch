"""
FastAPI Backend — RAG with Vector Search 2.0
=============================================
Exposes a single /rag endpoint:
  POST /rag  { "query": "...", "top_k": 5 }
           → { "answer": "...", "sources": [...] }

Start:
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

import vertexai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from vertexai.generative_models import GenerativeModel

from app.llm import generate_answer, init_vertex
from app.rag import VS2Retriever, build_context

app = FastAPI(
    title="RAG — Vector Search 2.0 + Gemini",
    version="1.0.0",
    description="RAG using Vertex AI Vector Search 2.0 Collections (no Firestore needed)",
)

# ── Lazy singletons ───────────────────────────────────────────────────────────
_retriever: VS2Retriever | None = None
_model:     GenerativeModel  | None = None


def get_retriever() -> VS2Retriever:
    global _retriever
    if _retriever is None:
        init_vertex(config.PROJECT_ID, config.LOCATION)
        _retriever = VS2Retriever(
            collection_resource=config.COLLECTION_RESOURCE,
            embedding_model_name=config.EMBEDDING_MODEL,
        )
    return _retriever


def get_model() -> GenerativeModel:
    global _model
    if _model is None:
        init_vertex(config.PROJECT_ID, config.LOCATION)
        _model = GenerativeModel(config.GEMINI_MODEL)
    return _model


# ── Schemas ───────────────────────────────────────────────────────────────────

class RagRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(None, ge=1, le=20)


class RagResponse(BaseModel):
    answer:  str
    sources: list[dict]
    model:   str
    collection: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def root() -> dict:
    return {
        "message": "POST /rag with {query: string, top_k?: int}",
        "docs": "/docs",
        "collection": config.COLLECTION_RESOURCE,
        "model": config.GEMINI_MODEL,
    }


@app.post("/rag", response_model=RagResponse)
def rag(req: RagRequest) -> RagResponse:
    """Embed → Vector Search 2.0 → Gemini."""
    query = req.query.strip()
    if not query:
        raise HTTPException(400, "query must be non-empty")

    top_k = req.top_k or config.TOP_K

    try:
        chunks  = get_retriever().search(query=query, top_k=top_k)
        context, sources = build_context(chunks)
        answer  = generate_answer(
            model=get_model(),
            question=query,
            context=context,
            max_output_tokens=config.MAX_OUTPUT_TOKENS,
            temperature=config.TEMPERATURE,
        )
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

    return RagResponse(
        answer=answer,
        sources=sources,
        model=config.GEMINI_MODEL,
        collection=config.COLLECTION_RESOURCE,
    )
