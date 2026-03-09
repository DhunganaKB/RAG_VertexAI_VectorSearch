from __future__ import annotations

from dataclasses import dataclass
from config import load_project_settings


@dataclass(frozen=True)
class Settings:
    project_id: str
    vertex_location: str
    gemini_model: str
    embedding_model: str
    vector_index_endpoint: str
    vector_deployed_index_id: str
    firestore_database: str
    firestore_collection: str
    top_k: int
    max_output_tokens: int
    temperature: float


def load_settings() -> Settings:
    project_settings = load_project_settings()
    if not project_settings.index_endpoint_resource_name:
        raise RuntimeError(
            "Vector endpoint not found in artifacts/vector_resources.json. "
            "Run scripts/create_vector_search.py first."
        )

    return Settings(
        project_id=project_settings.project_id,
        vertex_location=project_settings.region,
        gemini_model=project_settings.gemini_model,
        embedding_model=project_settings.embedding_model,
        vector_index_endpoint=project_settings.index_endpoint_resource_name,
        vector_deployed_index_id=project_settings.vector_deployed_index_id,
        firestore_database=project_settings.firestore_database,
        firestore_collection=project_settings.firestore_collection,
        top_k=project_settings.top_k,
        max_output_tokens=project_settings.max_output_tokens,
        temperature=project_settings.temperature,
    )
