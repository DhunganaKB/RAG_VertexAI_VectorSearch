"""
Inspect Vector Search 2.0 Collection Records
=============================================
Displays the first N DataObjects stored in the collection, showing:
  - System fields  (ID, create/update time)
  - Metadata       (title, source, text snippet)
  - Embedding      (dimensions + first 8 / last 4 values)

Usage:
    python scripts/inspect_collection.py              # show first 3 records
    python scripts/inspect_collection.py --n 5        # show first 5 records
    python scripts/inspect_collection.py --n 1 --full # show full text + full vector
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

from google.cloud import vectorsearch_v1beta as vs

# ── ANSI colours (degrade gracefully if terminal doesn't support them) ────────
BOLD  = "\033[1m"
CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
DIM   = "\033[2m"
RESET = "\033[0m"


def fmt_vector(values: list[float], full: bool = False) -> str:
    """Return a compact string representation of a dense vector."""
    n = len(values)
    if full or n <= 12:
        inner = ", ".join(f"{v:.6f}" for v in values)
    else:
        head = ", ".join(f"{v:.6f}" for v in values[:8])
        tail = ", ".join(f"{v:.6f}" for v in values[-4:])
        inner = f"{head},  …  {tail}"
    return f"[{inner}]  ({n} dims)"


def fmt_text(text: str, full: bool = False, width: int = 80) -> str:
    """Return a wrapped, optionally truncated text block."""
    if not full and len(text) > 400:
        text = text[:400].rstrip() + "  …"
    return textwrap.fill(text, width=width, subsequent_indent="              ")


def inspect(n: int = 3, full: bool = False) -> None:
    sc = vs.DataObjectSearchServiceClient()

    resp = sc.query_data_objects(
        request=vs.QueryDataObjectsRequest(
            parent=config.COLLECTION_RESOURCE,
            page_size=n,
            output_fields=vs.OutputFields(
                data_fields=["title", "source", "text"],
                vector_fields=["embedding"],
                metadata_fields=["create_time", "update_time"],
            ),
        )
    )

    objects = resp.data_objects
    total_hint = f"{n} shown" + (" (more exist)" if resp.next_page_token else "")

    print(f"\n{BOLD}{CYAN}══ VS2.0 Collection: {config.COLLECTION_ID} ══{RESET}")
    print(f"{DIM}Location : {config.LOCATION}   Project : {config.PROJECT_ID}{RESET}")
    print(f"{DIM}Records  : {total_hint}{RESET}\n")

    for i, obj in enumerate(objects, 1):
        data    = dict(obj.data)
        vectors = obj.vectors          # map<string, Vector>

        # ── embedding values ────────────────────────────────────────────────
        emb_values: list[float] = []
        if "embedding" in vectors:
            emb_values = list(vectors["embedding"].dense.values)

        print(f"{BOLD}{GREEN}┌─ Record {i} {'─' * 60}{RESET}")

        # ── System fields ────────────────────────────────────────────────────
        print(f"{BOLD}│ ID          {RESET}: {obj.data_object_id}")
        print(f"{BOLD}│ Created     {RESET}: {obj.create_time}")
        print(f"{BOLD}│ Updated     {RESET}: {obj.update_time}")

        # ── Metadata ─────────────────────────────────────────────────────────
        print(f"{BOLD}│{RESET}")
        print(f"{BOLD}│ {YELLOW}── METADATA ──{RESET}")
        print(f"{BOLD}│ title       {RESET}: {data.get('title', '—')}")
        print(f"{BOLD}│ source      {RESET}: {data.get('source', '—')}")
        text_block = fmt_text(data.get("text", ""), full=full)
        print(f"{BOLD}│ text        {RESET}: {text_block}")

        # ── Vector ───────────────────────────────────────────────────────────
        print(f"{BOLD}│{RESET}")
        print(f"{BOLD}│ {YELLOW}── EMBEDDING ──{RESET}")
        if emb_values:
            print(f"{BOLD}│ field       {RESET}: embedding")
            print(f"{BOLD}│ dims        {RESET}: {len(emb_values)}")
            print(f"{BOLD}│ values      {RESET}: {fmt_vector(emb_values, full=full)}")
            # Stats
            mn = min(emb_values)
            mx = max(emb_values)
            mean = sum(emb_values) / len(emb_values)
            print(f"{BOLD}│ range       {RESET}: min={mn:.6f}  max={mx:.6f}  mean={mean:.6f}")
        else:
            print(f"{BOLD}│{RESET} (no vector returned)")

        print(f"{BOLD}{GREEN}└{'─' * 73}{RESET}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect VS2.0 DataObjects")
    parser.add_argument("--n",    type=int, default=3,
                        help="Number of records to display (default: 3)")
    parser.add_argument("--full", action="store_true",
                        help="Show full text and all vector values")
    args = parser.parse_args()
    inspect(n=args.n, full=args.full)


if __name__ == "__main__":
    main()
