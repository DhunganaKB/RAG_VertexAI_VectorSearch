from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from google.cloud import aiplatform
from google.cloud import firestore
from vertexai.language_models import TextEmbeddingModel


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    title: str | None
    uri: str | None
    snippet: str | None
    score: float | None


class FirestoreMetadataStore:
    """Fetches chunk metadata from Firestore using a single batched RPC.

    Scalability rationale
    ---------------------
    The naive approach (loading the entire chunks.jsonl into RAM) breaks at
    scale — millions of chunks means GBs of memory and slow startup.

    Google's recommended pattern for Vertex AI Vector Search RAG:
        query → embed → Vector Search (top-K IDs) → Firestore get_all(IDs) → LLM

    db.get_all(refs) is a single Firestore RPC that returns exactly the K
    documents you ask for.  Cost / latency = O(top_k), not O(total_corpus).
    """

    def __init__(self, project_id: str, collection: str, database: str = "(default)") -> None:
        self._db = firestore.Client(project=project_id, database=database)
        self._col = self._db.collection(collection)

    def batch_get(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        """Single Firestore RPC that returns metadata for the given chunk IDs."""
        if not ids:
            return {}
        refs = [self._col.document(chunk_id) for chunk_id in ids]
        result: dict[str, dict[str, Any]] = {}
        for doc in self._db.get_all(refs):
            if doc.exists:
                result[doc.id] = doc.to_dict()
        return result


def build_context(chunks: list[RetrievedChunk]) -> tuple[str, list[dict[str, Any]]]:
    """Convert retrieved chunks into (context_string, sources_list) for the LLM."""
    sources: list[dict[str, Any]] = []
    lines: list[str] = []

    for idx, chunk in enumerate(chunks, start=1):
        sources.append(
            {
                "rank": idx,
                "id": chunk.id,
                "title": chunk.title,
                "uri": chunk.uri,
                "snippet": chunk.snippet,
                "score": chunk.score,
            }
        )

        title = (chunk.title or "").strip()
        uri = (chunk.uri or "").strip()
        snippet = (chunk.snippet or "").strip()

        lines.append(f"[{idx}] {title or chunk.id}")
        if uri:
            lines.append(f"URL: {uri}")
        if snippet:
            lines.append(f"Snippet: {snippet}")
        lines.append("")

    return "\n".join(lines).strip(), sources


class VectorRetriever:
    def __init__(
        self,
        project_id: str,
        location: str,
        endpoint_resource_name: str,
        deployed_index_id: str,
        embedding_model_name: str,
        firestore_collection: str,
        firestore_database: str = "(default)",
    ) -> None:
        self.project_id = project_id
        self.location = location
        self.endpoint_resource_name = endpoint_resource_name
        self.deployed_index_id = deployed_index_id
        self.embedding_model = TextEmbeddingModel.from_pretrained(embedding_model_name)
        self.index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=endpoint_resource_name,
            project=project_id,
            location=location,
        )
        self._metadata_store = FirestoreMetadataStore(
            project_id=project_id,
            collection=firestore_collection,
            database=firestore_database,
        )

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        # 1. Embed the query
        embedding = self.embedding_model.get_embeddings([query])[0].values

        # 2. ANN search in Vertex AI Vector Search
        neighbor_lists = self.index_endpoint.find_neighbors(
            deployed_index_id=self.deployed_index_id,
            queries=[list(embedding)],
            num_neighbors=top_k,
            return_full_datapoint=False,
            #return_full_datapoint=True,
        )

        if not neighbor_lists:
            return []

        neighbors = neighbor_lists[0]
        print("------>")
        print(neighbors)
        neighbor_ids = [str(getattr(n, "id", "")) for n in neighbors]
        print(neighbor_ids)
        print('----->')

        # 3. Fetch metadata for only these top-K chunk IDs from Firestore.
        #    This is one RPC regardless of how large the total corpus is.
        metadata_map = self._metadata_store.batch_get(neighbor_ids)
        print(metadata_map)

        chunks: list[RetrievedChunk] = []
        for neighbor_id, neighbor in zip(neighbor_ids, neighbors):
            meta = metadata_map.get(neighbor_id, {})
            chunks.append(
                RetrievedChunk(
                    id=neighbor_id,
                    title=meta.get("title"),
                    uri=meta.get("uri"),
                    snippet=meta.get("chunk"),
                    score=float(getattr(neighbor, "distance", 0.0)),
                )
            )
        return chunks
