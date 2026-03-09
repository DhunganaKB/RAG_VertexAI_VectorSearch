from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from vertexai.generative_models import GenerativeModel

from app.config import Settings, load_settings
from app.llm import generate_grounded_answer, init_vertex
from app.rag import VectorRetriever, build_context


app = FastAPI(
    title="Vertex AI Vector Search RAG",
    version="1.0.0",
    description=(
        "Retrieval-Augmented Generation powered by Vertex AI Vector Search, "
        "Firestore (scalable metadata), and Gemini."
    ),
)

_settings: Settings | None = None
_retriever: VectorRetriever | None = None
_model: GenerativeModel | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def get_retriever() -> VectorRetriever:
    global _retriever
    if _retriever is None:
        settings = get_settings()
        init_vertex(project_id=settings.project_id, location=settings.vertex_location)
        _retriever = VectorRetriever(
            project_id=settings.project_id,
            location=settings.vertex_location,
            endpoint_resource_name=settings.vector_index_endpoint,
            deployed_index_id=settings.vector_deployed_index_id,
            embedding_model_name=settings.embedding_model,
            firestore_collection=settings.firestore_collection,
            firestore_database=settings.firestore_database,
        )
    return _retriever


def get_model() -> GenerativeModel:
    global _model
    if _model is None:
        settings = get_settings()
        init_vertex(project_id=settings.project_id, location=settings.vertex_location)
        _model = GenerativeModel(settings.gemini_model)
    return _model


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class RagRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    top_k: int | None = Field(None, ge=1, le=20, description="Number of chunks to retrieve")


class RagResponse(BaseModel):
    answer: str
    sources: list[dict]
    vector_index_endpoint: str
    model: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/", tags=["ops"])
def root() -> dict:
    settings = get_settings()
    return {
        "message": "POST /rag with JSON body {query: string, top_k?: int}",
        "docs": "/docs",
        "vector_index_endpoint": settings.vector_index_endpoint,
        "model": settings.gemini_model,
        "firestore_collection": settings.firestore_collection,
    }


@app.post("/rag", response_model=RagResponse, tags=["rag"])
def rag(req: RagRequest) -> RagResponse:
    """Run a RAG query: embed → Vector Search → Firestore → Gemini."""
    settings = get_settings()

    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must be non-empty")

    top_k = req.top_k or settings.top_k

    try:
        chunks = get_retriever().search(query=query, top_k=top_k)
        context, sources = build_context(chunks)
        answer = generate_grounded_answer(
            model=get_model(),
            question=query,
            context=context,
            max_output_tokens=settings.max_output_tokens,
            temperature=settings.temperature,
        )
    except Exception as exc:
        # Surface error in dev; tighten this for production deployments.
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RagResponse(
        answer=answer,
        sources=sources,
        vector_index_endpoint=settings.vector_index_endpoint,
        model=settings.gemini_model,
    )
