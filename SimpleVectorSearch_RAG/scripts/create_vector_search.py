from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from google.api_core import exceptions
from google.cloud import aiplatform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import load_project_settings, save_runtime_resources_with_deployed_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensure Vertex AI Vector Search index/endpoint exists and is deployed."
    )
    parser.add_argument("--project-id")
    parser.add_argument("--region")
    parser.add_argument("--bucket", help="Optional staging bucket name without gs:// prefix.")
    parser.add_argument("--index-display-name")
    parser.add_argument("--endpoint-display-name")
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--deployed-index-id")
    parser.add_argument("--machine-type")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete matching resources first, then recreate.",
    )
    return parser.parse_args()


def _run_with_endpoint_lock_retry(action_label: str, action) -> None:
    max_attempts = 30
    sleep_seconds = 20
    for attempt in range(1, max_attempts + 1):
        try:
            action()
            return
        except exceptions.FailedPrecondition as exc:
            message = str(exc)
            retryable = (
                "other operations running on the IndexEndpoint" in message
                or "has deployed or being-deployed DeployedIndex(s)" in message
            )
            if not retryable:
                raise
            if attempt == max_attempts:
                raise RuntimeError(f"{action_label} failed after retries. Last error: {message}") from exc
            print(f"{action_label} blocked by endpoint precondition. Retrying in {sleep_seconds}s...")
            time.sleep(sleep_seconds)
        except exceptions.NotFound:
            return


def _find_index_by_display_name(display_name: str):
    matches = [index for index in aiplatform.MatchingEngineIndex.list() if index.display_name == display_name]
    if not matches:
        return None
    return aiplatform.MatchingEngineIndex(matches[0].resource_name)


def _find_endpoint_by_display_name(display_name: str):
    matches = [
        endpoint
        for endpoint in aiplatform.MatchingEngineIndexEndpoint.list()
        if endpoint.display_name == display_name
    ]
    if not matches:
        return None
    return aiplatform.MatchingEngineIndexEndpoint(matches[0].resource_name)


def _endpoint_deployed_ids(endpoint_resource_name: str) -> list[str]:
    endpoint = aiplatform.MatchingEngineIndexEndpoint(endpoint_resource_name)
    deployed = list(getattr(endpoint, "deployed_indexes", []) or [])
    values: list[str] = []
    for item in deployed:
        existing_id = getattr(item, "id", None)
        if existing_id:
            values.append(existing_id)
    return values


def _find_endpoint_by_deployed_index_id(deployed_index_id: str):
    for endpoint in aiplatform.MatchingEngineIndexEndpoint.list():
        deployed = list(getattr(endpoint, "deployed_indexes", []) or [])
        for item in deployed:
            existing_id = getattr(item, "id", None)
            if existing_id == deployed_index_id:
                return aiplatform.MatchingEngineIndexEndpoint(endpoint.resource_name)
    return None


def _undeploy_all_from_endpoint(endpoint_resource_name: str) -> None:
    max_rounds = 30
    for _ in range(max_rounds):
        deployed_ids = _endpoint_deployed_ids(endpoint_resource_name)
        if not deployed_ids:
            return
        for existing_id in deployed_ids:
            print(f"Undeploying deployed index '{existing_id}' from {endpoint_resource_name}...")
            _run_with_endpoint_lock_retry(
                action_label=f"Undeploy {existing_id} from {endpoint_resource_name}",
                action=lambda existing_id=existing_id: aiplatform.MatchingEngineIndexEndpoint(
                    endpoint_resource_name
                ).undeploy_index(deployed_index_id=existing_id),
            )
        time.sleep(10)
    raise RuntimeError(f"Endpoint {endpoint_resource_name} still has deployed indexes after retries.")


def _cleanup_existing_resources(deployed_index_id: str, index_display_name: str, endpoint_display_name: str) -> None:
    endpoints = aiplatform.MatchingEngineIndexEndpoint.list()
    for endpoint in endpoints:
        should_delete = endpoint.display_name == endpoint_display_name
        deployed = list(getattr(endpoint, "deployed_indexes", []) or [])
        for item in deployed:
            existing_id = getattr(item, "id", None)
            if existing_id == deployed_index_id:
                should_delete = True
        if should_delete:
            _undeploy_all_from_endpoint(endpoint.resource_name)
            print(f"Deleting endpoint {endpoint.resource_name}...")
            _run_with_endpoint_lock_retry(
                action_label=f"Delete endpoint {endpoint.resource_name}",
                action=lambda: aiplatform.MatchingEngineIndexEndpoint(endpoint.resource_name).delete(sync=True),
            )

    for index in aiplatform.MatchingEngineIndex.list():
        if index.display_name == index_display_name:
            print(f"Deleting existing index {index.resource_name}...")
            aiplatform.MatchingEngineIndex(index.resource_name).delete(sync=True)


def _deploy_with_retry(
    endpoint: aiplatform.MatchingEngineIndexEndpoint,
    index: aiplatform.MatchingEngineIndex,
    deployed_index_id: str,
    machine_type: str,
) -> str:
    max_attempts = 20
    sleep_seconds = 20
    conflict_markers = (
        "There exists a DeployedIndex with same ID",
        "There already exists a DeployedIndex with same ID",
    )
    lock_markers = (
        "other operations running on the IndexEndpoint",
        "has deployed or being-deployed DeployedIndex(s)",
    )

    for attempt in range(1, max_attempts + 1):
        try:
            endpoint.deploy_index(
                index=index,
                deployed_index_id=deployed_index_id,
                machine_type=machine_type,
                min_replica_count=1,
                max_replica_count=1,
                sync=True,
            )
            return deployed_index_id
        except (exceptions.FailedPrecondition, exceptions.AlreadyExists) as exc:
            message = str(exc)
            if any(marker in message for marker in conflict_markers + lock_markers):
                if attempt < max_attempts:
                    print(
                        f"Deploy id '{deployed_index_id}' is not ready yet. Retrying in "
                        f"{sleep_seconds}s ({attempt}/{max_attempts})..."
                    )
                    time.sleep(sleep_seconds)
                    continue
            raise

    sanitized_base = re.sub(r"[^A-Za-z0-9_]", "_", deployed_index_id)
    if not sanitized_base or not sanitized_base[0].isalpha():
        sanitized_base = f"d_{sanitized_base}"
    fallback_id = f"{sanitized_base}_{int(time.time())}"
    print(f"Deploy id '{deployed_index_id}' unavailable. Falling back to '{fallback_id}'.")
    endpoint.deploy_index(
        index=index,
        deployed_index_id=fallback_id,
        machine_type=machine_type,
        min_replica_count=1,
        max_replica_count=1,
        sync=True,
    )
    return fallback_id


def ensure_resources(
    index_display_name: str,
    endpoint_display_name: str,
    dimensions: int,
    deployed_index_id: str,
    machine_type: str,
    recreate: bool,
) -> dict[str, str | int]:
    if recreate:
        print("Recreate mode enabled: deleting matching resources first...")
        _cleanup_existing_resources(
            deployed_index_id=deployed_index_id,
            index_display_name=index_display_name,
            endpoint_display_name=endpoint_display_name,
        )

    index = _find_index_by_display_name(index_display_name)
    endpoint = _find_endpoint_by_display_name(endpoint_display_name)

    if index is None:
        index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=index_display_name,
            dimensions=dimensions,
            approximate_neighbors_count=150,
            distance_measure_type="COSINE_DISTANCE",
            leaf_node_embedding_count=500,
            leaf_nodes_to_search_percent=7,
            index_update_method="STREAM_UPDATE",
            description="RAG index created by scripts/create_vector_search.py",
            sync=True,
        )
    else:
        print(f"Reusing existing index: {index.resource_name}")

    if endpoint is None:
        endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=endpoint_display_name,
            public_endpoint_enabled=True,
            description="RAG endpoint created by scripts/create_vector_search.py",
            sync=True,
        )
    else:
        print(f"Reusing existing endpoint: {endpoint.resource_name}")

    existing_deployed_ids = _endpoint_deployed_ids(endpoint.resource_name)
    if deployed_index_id in existing_deployed_ids:
        used_deployed_id = deployed_index_id
        print(f"Reusing existing deployed index id: {used_deployed_id}")
    else:
        in_other_endpoint = _find_endpoint_by_deployed_index_id(deployed_index_id)
        if in_other_endpoint and in_other_endpoint.resource_name != endpoint.resource_name:
            print(
                f"Deploy id '{deployed_index_id}' currently exists on another endpoint "
                f"{in_other_endpoint.resource_name}. Waiting/fallback logic will be applied."
            )
        used_deployed_id = _deploy_with_retry(
            endpoint=endpoint,
            index=index,
            deployed_index_id=deployed_index_id,
            machine_type=machine_type,
        )

    save_runtime_resources_with_deployed_id(
        index_resource_name=index.resource_name,
        index_endpoint_resource_name=endpoint.resource_name,
        deployed_index_id=used_deployed_id,
    )
    return {
        "index_resource_name": index.resource_name,
        "index_endpoint_resource_name": endpoint.resource_name,
        "deployed_index_id": used_deployed_id,
        "dimensions": dimensions,
        "saved_to": "artifacts/vector_resources.json",
    }


def main() -> None:
    args = parse_args()
    settings = load_project_settings()

    project_id = args.project_id or settings.project_id
    region = args.region or settings.region
    bucket = args.bucket or settings.bucket_name
    index_display_name = args.index_display_name or settings.vector_index_display_name
    endpoint_display_name = args.endpoint_display_name or settings.vector_endpoint_display_name
    dimensions = args.dimensions or settings.vector_dimensions
    deployed_index_id = args.deployed_index_id or settings.vector_deployed_index_id
    machine_type = args.machine_type or settings.vector_machine_type
    staging_bucket = f"gs://{bucket}" if bucket else None

    aiplatform.init(project=project_id, location=region, staging_bucket=staging_bucket)

    output = ensure_resources(
        index_display_name=index_display_name,
        endpoint_display_name=endpoint_display_name,
        dimensions=dimensions,
        deployed_index_id=deployed_index_id,
        machine_type=machine_type,
        recreate=args.recreate,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
