"""
RAG Retriever — Vector Search 2.0 (Airbnb)
==========================================
Uses DataObjectSearchServiceClient to find the top-K most similar
Airbnb listings for a given query.

VS2.0 advantage:
  - Listing metadata (name, price, location, etc.) stored IN the collection.
  - Single SearchDataObjects call returns both vector match AND all metadata.
  - No Firestore, no secondary store, no extra RPC.

Search mode (auto-detected at startup):
  - No Collection Index → exact kNN  (100% recall)
  - Collection Index present → ANN   (ScaNN, ~99% recall, sub-10ms)

Flow:
  query → embed → SearchDataObjects → results[].data_object.data → LLM/Agent
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from google.cloud import vectorsearch_v1beta as vs
from vertexai.language_models import TextEmbeddingModel

from app.cache import get_cached, rag_key, set_cached

logger = logging.getLogger(__name__)

_RETURN_FIELDS = [
    "name", "neighbourhood", "property_type", "room_type",
    "accommodates", "bedrooms", "price", "rating",
    "instant_bookable", "listing_url", "host_name", "text",
]


def _detect_search_mode(collection_resource: str) -> str:
    """
    Detect whether a Collection Index (ScaNN ANN) exists.

    Returns:
        "ANN (ScaNN) — index: '<id>'"   if index exists
        "kNN (exact) — no index found"  otherwise
    """
    try:
        client  = vs.VectorSearchServiceClient()
        indexes = list(
            client.list_indexes(
                request=vs.ListIndexesRequest(parent=collection_resource)
            )
        )
        if indexes:
            idx_id = indexes[0].name.split("/")[-1]
            return f"ANN (ScaNN) — index: '{idx_id}'"
        return "kNN (exact) — no Collection Index found"
    except Exception as exc:
        logger.warning("Could not detect search mode: %s", exc)
        return "unknown"


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ListingResult:
    id:              str
    name:            str | None
    neighbourhood:   str | None
    property_type:   str | None
    room_type:       str | None
    accommodates:    float | None
    bedrooms:        float | None
    price:           float | None
    rating:          float | None
    instant_bookable: str | None
    listing_url:     str | None
    host_name:       str | None
    text:            str | None
    score:           float


# ── Retriever ─────────────────────────────────────────────────────────────────

class VS2Retriever:
    """
    Vector Search 2.0 retriever for Airbnb listings.

    Detects search mode (kNN vs ANN) at startup and logs it.
    The search() method is identical regardless of mode — VS2.0 routes
    to ScaNN automatically when an index exists.
    """

    def __init__(
        self,
        collection_resource: str,
        embedding_model_name: str,
    ) -> None:
        self._collection      = collection_resource
        self._embedding_model = TextEmbeddingModel.from_pretrained(embedding_model_name)
        self._search_client   = vs.DataObjectSearchServiceClient()
        self._do_client       = vs.DataObjectServiceClient()

        mode = _detect_search_mode(collection_resource)
        logger.info("VS2Retriever initialised | search mode: %s", mode)
        print(f"[VS2Retriever] Search mode: {mode}")

    def _fetch_data(self, obj_id: str) -> dict:
        """Fetch a single DataObject's metadata by ID."""
        name = f"{self._collection}/dataObjects/{obj_id}"
        try:
            obj = self._do_client.get_data_object(
                request=vs.GetDataObjectRequest(name=name)
            )
            return dict(obj.data) if obj.data is not None else {}
        except Exception as exc:
            logger.warning("Could not fetch DataObject %s: %s", obj_id, exc)
            return {}

    def search(self, query: str, top_k: int) -> list[ListingResult]:
        """
        Embed query, search the VS2.0 Collection, then fetch metadata.

        Cache strategy (Redis):
          Key   → rag_key(query, top_k)
          TTL   → config.REDIS_RAG_TTL  (default 3600s)
          Store → list of ListingResult dicts (JSON-serialisable)
        """
        import config as _cfg

        # ── Cache read ────────────────────────────────────────────────────────
        ckey   = rag_key(query, top_k)
        cached = get_cached(ckey)
        if cached is not None:
            logger.debug("Cache HIT for RAG query: %s", query[:60])
            return [ListingResult(**r) for r in cached]

        # ── Cache miss: compute normally ──────────────────────────────────────
        query_vector = list(self._embedding_model.get_embeddings([query])[0].values)

        request = vs.SearchDataObjectsRequest(
            parent=self._collection,
            vector_search=vs.VectorSearch(
                vector=vs.DenseVector(values=query_vector),
                search_field="embedding",
                top_k=top_k,
            ),
        )
        response = self._search_client.search_data_objects(request=request)

        # search_data_objects only returns IDs and distance — fetch metadata in parallel
        hits = [(res.data_object.data_object_id, float(res.distance))
                for res in (response.results or [])]

        if not hits:
            return []

        with ThreadPoolExecutor(max_workers=min(len(hits), 10)) as pool:
            futures = {pool.submit(self._fetch_data, obj_id): (obj_id, dist)
                       for obj_id, dist in hits}
            data_map: dict[str, tuple[dict, float]] = {}
            for future in as_completed(futures):
                obj_id, dist = futures[future]
                data_map[obj_id] = (future.result(), dist)

        results: list[ListingResult] = []
        for obj_id, dist in hits:  # preserve ranking order
            data, score = data_map[obj_id]
            results.append(ListingResult(
                id=obj_id,
                name=data.get("name"),
                neighbourhood=data.get("neighbourhood"),
                property_type=data.get("property_type"),
                room_type=data.get("room_type"),
                accommodates=data.get("accommodates"),
                bedrooms=data.get("bedrooms"),
                price=data.get("price"),
                rating=data.get("rating"),
                instant_bookable=data.get("instant_bookable"),
                listing_url=data.get("listing_url"),
                host_name=data.get("host_name"),
                text=data.get("text"),
                score=score,
            ))

        # ── Cache write (Firestore) ───────────────────────────────────────────
        ttl = getattr(_cfg, "CACHE_RAG_TTL", 3600)
        set_cached(ckey, [vars(r) for r in results], ttl=ttl,
                   endpoint="rag", query_preview=query)

        return results


# ── Context builder for LLM ───────────────────────────────────────────────────

def build_context(results: list[ListingResult]) -> tuple[str, list[dict]]:
    """Format retrieved listings as (context_string, sources_list) for LLM."""
    lines:   list[str]  = []
    sources: list[dict] = []

    for i, r in enumerate(results, start=1):
        sources.append({
            "rank":           i,
            "id":             r.id,
            "name":           r.name,
            "neighbourhood":  r.neighbourhood,
            "room_type":      r.room_type,
            "price":          r.price,
            "rating":         r.rating,
            "accommodates":   r.accommodates,
            "instant_bookable": r.instant_bookable,
            "listing_url":    r.listing_url,
            "host_name":      r.host_name,
            "text":           r.text,
            "score":          r.score,
        })

        lines.append(f"[{i}] {r.name or r.id}")
        if r.neighbourhood:
            lines.append(f"Location: {r.neighbourhood}")
        if r.room_type:
            lines.append(f"Type: {r.room_type}")
        if r.price:
            lines.append(f"Price: ${r.price:.0f}/night")
        if r.rating:
            lines.append(f"Rating: {r.rating:.2f} ★")
        if r.accommodates:
            lines.append(f"Accommodates: {int(r.accommodates)} guests")
        if r.bedrooms:
            lines.append(f"Bedrooms: {int(r.bedrooms)}")
        if r.instant_bookable:
            lines.append(f"Instant bookable: {r.instant_bookable}")
        if r.listing_url:
            lines.append(f"URL: {r.listing_url}")
        if r.text:
            lines.append(f"Description: {r.text[:300]}")
        lines.append("")

    return "\n".join(lines).strip(), sources
