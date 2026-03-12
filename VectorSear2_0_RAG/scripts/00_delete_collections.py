"""
Delete Vector Search 2.0 Collections
=====================================
Lists all collections in the project and lets you delete them
interactively or all at once.

Run:
    python scripts/00_delete_collections.py             # interactive (asks per collection)
    python scripts/00_delete_collections.py --all       # delete all without prompting
    python scripts/00_delete_collections.py --id collection-for-rag-demo  # delete one by ID
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

from google.api_core.exceptions import NotFound
from google.cloud import vectorsearch_v1beta as vs

_BATCH_SIZE = 100  # max DataObjects per batch_delete call


def get_client() -> vs.VectorSearchServiceClient:
    return vs.VectorSearchServiceClient()


def list_collections(client: vs.VectorSearchServiceClient) -> list[vs.Collection]:
    parent = f"projects/{config.PROJECT_ID}/locations/{config.LOCATION}"
    return list(client.list_collections(
        request=vs.ListCollectionsRequest(parent=parent)
    ))


def purge_data_objects(collection_name: str) -> int:
    """Delete every DataObject in *collection_name*. Returns count removed."""
    search_client = vs.DataObjectSearchServiceClient()
    do_client     = vs.DataObjectServiceClient()

    names: list[str] = []
    page_token = ""

    # Page through ALL DataObjects (empty filter = match everything)
    while True:
        resp = search_client.query_data_objects(
            request=vs.QueryDataObjectsRequest(
                parent=collection_name,
                page_size=500,
                page_token=page_token,
            )
        )
        for obj in resp.data_objects:
            names.append(obj.name)
        page_token = resp.next_page_token
        if not page_token:
            break

    if not names:
        return 0

    # Batch-delete in chunks of _BATCH_SIZE
    for i in range(0, len(names), _BATCH_SIZE):
        chunk = names[i : i + _BATCH_SIZE]
        do_client.batch_delete_data_objects(
            request=vs.BatchDeleteDataObjectsRequest(
                parent=collection_name,
                requests=[vs.DeleteDataObjectRequest(name=n) for n in chunk],
            )
        )

    return len(names)


def delete_one(client: vs.VectorSearchServiceClient, name: str) -> None:
    col_id = name.split("/")[-1]
    print(f"  Purging DataObjects from '{col_id}'...", end=" ", flush=True)
    removed = purge_data_objects(name)
    print(f"{removed} removed")

    print(f"  Deleting collection '{col_id}'...", end=" ", flush=True)
    try:
        operation = client.delete_collection(
            request=vs.DeleteCollectionRequest(name=name)
        )
        operation.result()
        print("DONE")
    except NotFound:
        print("NOT FOUND (already deleted)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="Delete ALL collections without prompting")
    parser.add_argument("--id", metavar="COLLECTION_ID",
                        help="Delete a single collection by ID")
    args = parser.parse_args()

    client = get_client()

    # ── Single collection by ID ───────────────────────────────────────────────
    if args.id:
        name = f"projects/{config.PROJECT_ID}/locations/{config.LOCATION}/collections/{args.id}"
        print(f"\nDeleting collection: {args.id}")
        delete_one(client, name)
        print("\nDone.")
        return

    # ── List what exists ──────────────────────────────────────────────────────
    collections = list_collections(client)

    if not collections:
        print("\nNo collections found in "
              f"projects/{config.PROJECT_ID}/locations/{config.LOCATION}")
        return

    print(f"\nFound {len(collections)} collection(s):\n")
    for i, col in enumerate(collections, 1):
        col_id = col.name.split("/")[-1]
        print(f"  [{i}] {col_id}")
        print(f"       Created : {col.create_time}")
        vec_fields = list(col.vector_schema.keys())
        print(f"       Vectors : {vec_fields}")
        print()

    # ── Delete all without prompting ──────────────────────────────────────────
    if args.all:
        print(f"Deleting all {len(collections)} collection(s)...\n")
        for col in collections:
            delete_one(client, col.name)
        print("\nAll collections deleted.")
        print("You can now start fresh with:")
        print("  python scripts/01_setup_collection.py")
        return

    # ── Interactive mode ──────────────────────────────────────────────────────
    print("Options:")
    print("  Enter collection number(s) to delete  (e.g. 1  or  1,2,3)")
    print("  Enter 'all' to delete everything")
    print("  Enter 'q' to quit\n")

    choice = input("Your choice: ").strip().lower()

    if choice == "q" or choice == "":
        print("Aborted — nothing deleted.")
        return

    to_delete: list[vs.Collection] = []

    if choice == "all":
        to_delete = collections
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(",")]
            for idx in indices:
                if 0 <= idx < len(collections):
                    to_delete.append(collections[idx])
                else:
                    print(f"  Invalid index: {idx + 1} — skipping")
        except ValueError:
            print("Invalid input. Aborted.")
            return

    if not to_delete:
        print("Nothing to delete.")
        return

    print(f"\nAbout to delete {len(to_delete)} collection(s):")
    for col in to_delete:
        print(f"  - {col.name.split('/')[-1]}")

    confirm = input("\nConfirm? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Aborted — nothing deleted.")
        return

    print()
    for col in to_delete:
        delete_one(client, col.name)

    print("\nDone.")
    print("\nTo start fresh:")
    print("  python scripts/01_setup_collection.py")
    print("  python scripts/02_ingest.py")


if __name__ == "__main__":
    main()
