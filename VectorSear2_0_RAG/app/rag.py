"""
RAG Retriever — Vector Search 2.0
===================================
Uses DataObjectSearchServiceClient to find the top-K most similar
chunks for a given query.

VS2.0 advantage over VS1.0:
  - Metadata (title, source, text) is stored IN the collection alongside vectors.
  - A single search call returns both the vector match AND the chunk text.
  - No Firestore, no secondary store, no extra RPC needed.

Flow:
  query → embed → SearchDataObjects → SearchResult[].data_object.data → LLM
"""
from __future__ import annotations

from dataclasses import dataclass

from google.cloud import vectorsearch_v1beta as vs
from vertexai.language_models import TextEmbeddingModel


@dataclass(frozen=True)
class RetrievedChunk:
    id:       str
    title:    str | None
    source:   str | None
    text:     str | None
    score:    float


class VS2Retriever:
    """Vector Search 2.0 retriever — no Firestore dependency."""

    def __init__(
        self,
        collection_resource: str,
        embedding_model_name: str,
    ) -> None:
        self._collection = collection_resource
        self._embedding_model = TextEmbeddingModel.from_pretrained(embedding_model_name)
        self._search_client = vs.DataObjectSearchServiceClient()

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        # 1. Embed the query
        query_vector = list(self._embedding_model.get_embeddings([query])[0].values)

        # 2. Vector search in VS2.0 Collection
        # output_fields.data_fields tells VS2.0 which metadata fields to return
        # alongside the vector match — this avoids a secondary store lookup.
        request = vs.SearchDataObjectsRequest(
            parent=self._collection,
            vector_search=vs.VectorSearch(
                vector=vs.DenseVector(values=query_vector),
                search_field="embedding",   # matches vector_schema key
                top_k=top_k,
                output_fields=vs.OutputFields(
                    data_fields=["title", "source", "text"],
                ),
            ),
        )
        response = self._search_client.search_data_objects(request=request)

        # 3. Unpack results — metadata lives in data_object.data (no extra RPC)
        chunks: list[RetrievedChunk] = []
        for result in response.results:
            obj  = result.data_object
            data = dict(obj.data)           # Struct → plain Python dict
            chunks.append(RetrievedChunk(
                id=obj.data_object_id,
                title=data.get("title"),
                source=data.get("source"),
                text=data.get("text"),
                score=float(result.distance),
            ))
        return chunks


def build_context(chunks: list[RetrievedChunk]) -> tuple[str, list[dict]]:
    """Format retrieved chunks as (context_string, sources_list) for the LLM."""
    lines:   list[str]  = []
    sources: list[dict] = []

    for i, c in enumerate(chunks, start=1):
        sources.append({
            "rank":   i,
            "id":     c.id,
            "title":  c.title,
            "source": c.source,
            "text":   c.text,
            "score":  c.score,
        })
        lines.append(f"[{i}] {c.title or c.id}")
        if c.source:
            lines.append(f"Source: {c.source}")
        if c.text:
            lines.append(f"Excerpt: {c.text}")
        lines.append("")

    return "\n".join(lines).strip(), sources
