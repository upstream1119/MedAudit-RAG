"""Build and audit a deterministic runtime graph over exact KB candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

from experiments.phase7_formal_experiments.graph_candidate_expansion import (
    GRAPH_INDEX_VERSION,
    build_candidate_graph_index,
)


COLLECTIONS = ("detail_128", "concept_512", "context_1024")
INDEX_FILENAME = "graph_candidate_index_v0_1.json"
AUDIT_FILENAME = "graph_candidate_index_audit_v0_1.json"
MANIFEST_FILENAME = "manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(
                f"JSONL row must be an object at line {line_number}: {path}"
            )
        rows.append(row)
    return rows


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", delete=False, dir=path.parent, prefix=f".{path.name}."
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def _load_exact_candidates(
    *, exact_asset_dir: Path, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = exact_asset_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    expected_manifest_sha256 = str(
        config.get("expected_asset_manifest_sha256", "")
    ).strip().lower()
    observed_manifest_sha256 = sha256_file(manifest_path)
    if not expected_manifest_sha256 or observed_manifest_sha256 != expected_manifest_sha256:
        raise ValueError("exact asset manifest SHA-256 mismatch")
    if manifest.get("asset_version") != "bge-m3-exact-dense-assets-v0.1":
        raise ValueError("unsupported exact asset version")
    if not manifest.get("ready"):
        raise RuntimeError("exact dense assets are not ready")

    configured_hashes = config.get("expected_collection_rows_sha256")
    if not isinstance(configured_hashes, dict):
        raise ValueError("expected_collection_rows_sha256 must be an object")

    rows: list[dict[str, Any]] = []
    collection_counts: dict[str, int] = {}
    observed_hashes: dict[str, str] = {}
    for collection_name in COLLECTIONS:
        collection_meta = manifest.get("collections", {}).get(collection_name)
        if not isinstance(collection_meta, dict):
            raise ValueError(f"missing exact asset collection: {collection_name}")
        rows_path = exact_asset_dir / str(collection_meta.get("rows_file", ""))
        observed_hash = sha256_file(rows_path)
        manifest_hash = str(collection_meta.get("rows_sha256", "")).lower()
        configured_hash = str(configured_hashes.get(collection_name, "")).lower()
        if not manifest_hash or observed_hash != manifest_hash:
            raise ValueError(f"manifest row SHA-256 mismatch: {collection_name}")
        if not configured_hash or observed_hash != configured_hash:
            raise ValueError(f"configured row SHA-256 mismatch: {collection_name}")
        collection_rows = _read_jsonl(rows_path)
        expected_collection_count = int(collection_meta.get("count", -1))
        if len(collection_rows) != expected_collection_count:
            raise ValueError(f"exact asset row count mismatch: {collection_name}")
        if any(row.get("collection") != collection_name for row in collection_rows):
            raise ValueError(f"collection field mismatch: {collection_name}")
        rows.extend(collection_rows)
        collection_counts[collection_name] = len(collection_rows)
        observed_hashes[collection_name] = observed_hash

    expected_count = int(config.get("expected_candidate_count", -1))
    if len(rows) != expected_count:
        raise ValueError(
            f"expected {expected_count} exact candidates, got {len(rows)}"
        )
    return rows, {
        "asset_manifest_sha256": observed_manifest_sha256,
        "collection_rows_sha256": observed_hashes,
        "collection_counts": collection_counts,
    }


def audit_candidate_graph_index(
    *, candidate_rows: list[dict[str, Any]], graph_index: dict[str, Any]
) -> dict[str, Any]:
    key_counts = Counter(str(row.get("candidate_key", "")) for row in candidate_rows)
    duplicate_keys = sorted(key for key, count in key_counts.items() if count > 1)
    candidates = graph_index["candidates"]
    candidate_constraints = graph_index["candidate_constraints"]
    constraint_types = Counter(
        constraint["constraint_type"]
        for constraints in candidate_constraints.values()
        for constraint in constraints
    )
    collection_counts = Counter(
        str(row["collection"]) for row in candidates.values()
    )
    source_pages = {
        (str(row["source_file"]), int(row["page_number"]))
        for row in candidates.values()
    }
    return {
        "audit_version": "phase7-c1c4e2-graph-candidate-index-audit-v0.1",
        "graph_index_version": graph_index["graph_index_version"],
        "constraint_ruleset_version": graph_index["constraint_ruleset_version"],
        "input_candidate_count": len(candidate_rows),
        "unique_candidate_count": len(candidates),
        "collection_counts": dict(sorted(collection_counts.items())),
        "source_count": len({row["source_file"] for row in candidates.values()}),
        "source_page_count": len(source_pages),
        "duplicate_candidate_key_count": len(duplicate_keys),
        "duplicate_candidate_keys": duplicate_keys,
        "empty_constraint_candidate_count": sum(
            1 for constraints in candidate_constraints.values() if not constraints
        ),
        "constraint_type_counts": dict(sorted(constraint_types.items())),
        "constraint_posting_key_count": len(graph_index["constraint_postings"]),
        "page_posting_key_count": len(graph_index["page_postings"]),
        "adjacent_page_key_count": len(graph_index["adjacent_pages"]),
    }


def build_graph_candidate_index_assets(
    *, exact_asset_dir: Path, output_dir: Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Build deterministic graph assets without reading benchmark Gold fields."""
    if str(config.get("graph_index_version", "")) != GRAPH_INDEX_VERSION:
        raise ValueError("configured graph index version mismatch")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"graph candidate index directory is not empty: {output_dir}")
    candidate_rows, input_audit = _load_exact_candidates(
        exact_asset_dir=exact_asset_dir,
        config=config,
    )
    graph_index = build_candidate_graph_index(candidate_rows)
    if graph_index["graph_index_version"] != GRAPH_INDEX_VERSION:
        raise RuntimeError("graph index version mismatch")
    audit = audit_candidate_graph_index(
        candidate_rows=candidate_rows,
        graph_index=graph_index,
    )
    if audit["duplicate_candidate_key_count"]:
        raise ValueError("duplicate candidate_key values are not allowed")

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / INDEX_FILENAME
    audit_path = output_dir / AUDIT_FILENAME
    _atomic_write(index_path, _json_bytes(graph_index))
    audit = {
        **audit,
        **input_audit,
        "graph_index_sha256": sha256_file(index_path),
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0.0,
    }
    _atomic_write(audit_path, _json_bytes(audit))
    manifest = {
        "manifest_version": "phase7-c1c4e2-graph-candidate-index-manifest-v0.1",
        "ready": True,
        "graph_index_version": GRAPH_INDEX_VERSION,
        "input_asset_manifest_sha256": input_audit["asset_manifest_sha256"],
        "files": {
            "graph_index": {
                "path": INDEX_FILENAME,
                "sha256": sha256_file(index_path),
            },
            "audit": {
                "path": AUDIT_FILENAME,
                "sha256": sha256_file(audit_path),
            },
        },
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0.0,
    }
    _atomic_write(output_dir / MANIFEST_FILENAME, _json_bytes(manifest))
    return {"audit": audit, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repo_root.resolve()
    config = _read_json(args.config)
    result = build_graph_candidate_index_assets(
        exact_asset_dir=root / config["exact_asset_dir"],
        output_dir=root / config["output_dir"],
        config=config,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
