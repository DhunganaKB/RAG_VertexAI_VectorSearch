"""
Airbnb Search Tool — VS2.0 Retriever
=====================================
Provides the find_rentals() function used by the Gemini agent.

The tool:
  1. Embeds the query with text-embedding-005
  2. Searches the VS2.0 Collection via SearchDataObjects (kNN or ScaNN ANN)
  3. Applies Python-side metadata filtering (price, room_type, etc.)
  4. Returns a formatted string of top-K listing results

No secondary store needed — all metadata lives inside the Collection.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

from google.cloud import vectorsearch_v1beta as vs
from vertexai import init as vertex_init
from vertexai.language_models import TextEmbeddingModel

from app.cache import ask_key, get_cached, set_cached

# ── Singletons (initialised on first call) ────────────────────────────────────
_embedding_model: TextEmbeddingModel | None = None
_search_client:   vs.DataObjectSearchServiceClient | None = None
_do_client:       vs.DataObjectServiceClient | None = None
_vertex_ready = False

# Fields to request from VS2.0 for every search result
_RETURN_FIELDS = [
    "name", "neighbourhood", "property_type", "room_type",
    "accommodates", "bedrooms", "price", "rating",
    "instant_bookable", "listing_url", "host_name", "text",
]


def _init() -> None:
    global _embedding_model, _search_client, _do_client, _vertex_ready
    if _vertex_ready:
        return
    vertex_init(project=config.PROJECT_ID, location=config.LOCATION)
    _embedding_model = TextEmbeddingModel.from_pretrained(config.EMBEDDING_MODEL)
    _search_client   = vs.DataObjectSearchServiceClient()
    _do_client       = vs.DataObjectServiceClient()
    _vertex_ready    = True


# ── Filter helpers ────────────────────────────────────────────────────────────

def _passes_filters(
    data: dict,
    max_price: float | None,
    min_price: float | None,
    room_type: str | None,
    min_bedrooms: int | None,
    neighbourhood: str | None,
    instant_bookable: bool | None,
    accommodates: int | None,
) -> bool:
    """Return True if the listing matches all specified filters."""
    price = data.get("price", 0.0)
    if max_price is not None and price > max_price:
        return False
    if min_price is not None and price < min_price:
        return False

    if room_type is not None:
        if data.get("room_type", "").lower() != room_type.lower():
            return False

    if min_bedrooms is not None:
        if data.get("bedrooms", 0) < min_bedrooms:
            return False

    if neighbourhood is not None:
        listing_nbhd = data.get("neighbourhood", "").lower()
        if neighbourhood.lower() not in listing_nbhd:
            return False

    if instant_bookable is not None:
        ib_val = data.get("instant_bookable", "false").lower() == "true"
        if ib_val != instant_bookable:
            return False

    if accommodates is not None:
        if data.get("accommodates", 0) < accommodates:
            return False

    return True


# ── Formatter ─────────────────────────────────────────────────────────────────

def _format_listing(rank: int, data: dict, score: float) -> str:
    """Format a single listing as a readable string for the agent."""
    name    = data.get("name", "Unknown")
    nbhd    = data.get("neighbourhood", "")
    r_type  = data.get("room_type", "")
    price   = data.get("price", 0.0)
    rating  = data.get("rating", 0.0)
    beds    = int(data.get("bedrooms", 0))
    accom   = int(data.get("accommodates", 0))
    ib      = data.get("instant_bookable", "false")
    host    = data.get("host_name", "")
    url     = data.get("listing_url", "")
    text    = data.get("text", "")[:200]

    ib_str  = "✓ Instant Book" if ib == "true" else "Request required"
    beds_str = f"{beds} BR" if beds else "Studio"

    lines = [
        f"[{rank}] {name}",
        f"    📍 {nbhd}  |  🏠 {r_type}  |  {beds_str}  |  👥 {accom} guests",
        f"    💰 ${price:.0f}/night  |  ⭐ {rating:.2f}  |  {ib_str}",
        f"    Host: {host}",
        f"    {url}",
        f"    \"{text}\"",
    ]
    return "\n".join(lines)


# ── Public tool function ───────────────────────────────────────────────────────

def find_rentals(
    query: str,
    max_price: float | None = None,
    min_price: float | None = None,
    room_type: str | None = None,
    min_bedrooms: int | None = None,
    neighbourhood: str | None = None,
    instant_bookable: bool | None = None,
    accommodates: int | None = None,
    top_k: int = 8,
) -> str:
    """
    Search for Airbnb rentals in Austin, TX.

    Performs semantic vector search against the VS2.0 Collection and applies
    metadata filters on the results. Returns a formatted list of matching listings.

    Args:
        query:            Natural language description (e.g. "quiet studio near downtown")
        max_price:        Maximum price per night in USD
        min_price:        Minimum price per night in USD
        room_type:        "Entire home/apt", "Private room", "Shared room", "Hotel room"
        min_bedrooms:     Minimum number of bedrooms
        neighbourhood:    Neighborhood name or zip code to filter by
        instant_bookable: True to show only instantly bookable listings
        accommodates:     Minimum number of guests the listing must accommodate
        top_k:            Number of results to return (default 8, max 20)

    Returns:
        Formatted string of matching listings, or a "no results" message.
    """
    _init()
    top_k = min(max(top_k, 1), 20)

    # ── Cache check ───────────────────────────────────────────────────────────
    # Key includes all filter params so different filter combos don't collide.
    _cache_key = ask_key(
        f"{query}|{max_price}|{min_price}|{room_type}|{min_bedrooms}"
        f"|{neighbourhood}|{instant_bookable}|{accommodates}|{top_k}"
    )
    _cached = get_cached(_cache_key)
    if _cached is not None:
        return _cached

    # Retrieve more candidates than needed so filters have enough to work with
    fetch_k = min(top_k * 4, 80)

    # ── 1. Embed the query ────────────────────────────────────────────────────
    query_vector = list(_embedding_model.get_embeddings([query])[0].values)

    # ── 2. Vector search in VS2.0 ─────────────────────────────────────────────
    request = vs.SearchDataObjectsRequest(
        parent=config.COLLECTION_RESOURCE,
        vector_search=vs.VectorSearch(
            vector=vs.DenseVector(values=query_vector),
            search_field="embedding",
            top_k=fetch_k,
        ),
    )
    response = _search_client.search_data_objects(request=request)

    # search_data_objects only returns IDs — fetch metadata in parallel
    hits = [(r.data_object.data_object_id, float(r.distance))
            for r in (response.results or [])]

    def _fetch(obj_id: str) -> dict:
        name = f"{config.COLLECTION_RESOURCE}/dataObjects/{obj_id}"
        try:
            obj = _do_client.get_data_object(request=vs.GetDataObjectRequest(name=name))
            return dict(obj.data) if obj.data is not None else {}
        except Exception:
            return {}

    with ThreadPoolExecutor(max_workers=min(len(hits), 10)) as pool:
        futures = {pool.submit(_fetch, obj_id): (obj_id, dist) for obj_id, dist in hits}
        data_by_id: dict[str, tuple[dict, float]] = {}
        for future in as_completed(futures):
            obj_id, dist = futures[future]
            data_by_id[obj_id] = (future.result(), dist)

    # ── 3. Apply metadata filters ─────────────────────────────────────────────
    results = []
    for obj_id, dist in hits:
        data, distance = data_by_id[obj_id]
        if _passes_filters(
            data,
            max_price=max_price,
            min_price=min_price,
            room_type=room_type,
            min_bedrooms=min_bedrooms,
            neighbourhood=neighbourhood,
            instant_bookable=instant_bookable,
            accommodates=accommodates,
        ):
            results.append((data, distance))

    results = results[:top_k]

    # ── 4. Format and return ──────────────────────────────────────────────────
    if not results:
        filter_summary = []
        if max_price:
            filter_summary.append(f"max ${max_price:.0f}/night")
        if min_price:
            filter_summary.append(f"min ${min_price:.0f}/night")
        if room_type:
            filter_summary.append(f"room_type={room_type}")
        if min_bedrooms:
            filter_summary.append(f"min {min_bedrooms} bedrooms")
        if neighbourhood:
            filter_summary.append(f"neighbourhood={neighbourhood}")
        if instant_bookable:
            filter_summary.append("instant bookable")
        if accommodates:
            filter_summary.append(f"accommodates>={accommodates}")
        filters_str = ", ".join(filter_summary) if filter_summary else "none"
        return (
            f"No listings found matching your criteria (filters: {filters_str}). "
            f"Try broadening the search — e.g. increase max_price, relax room_type, "
            f"or remove the neighbourhood filter."
        )

    lines = [
        f"Found {len(results)} listing(s) in Austin, TX matching '{query}':",
        "",
    ]
    for rank, (data, score) in enumerate(results, 1):
        lines.append(_format_listing(rank, data, score))
        lines.append("")

    result_str = "\n".join(lines).strip()

    # ── Cache write (Firestore) ───────────────────────────────────────────────
    ttl = getattr(config, "CACHE_ASK_TTL", 1800)
    set_cached(_cache_key, result_str, ttl=ttl,
               endpoint="ask", query_preview=query)

    return result_str
