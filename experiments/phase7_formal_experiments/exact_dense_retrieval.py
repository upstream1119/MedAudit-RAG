"""Deterministic exhaustive dense retrieval for experimental controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


COLLECTIONS = ("detail_128", "concept_512", "context_1024")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_exact_dense_assets(
    asset_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load exact-dense assets after checking their manifest and file hashes."""
    manifest_path = asset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("ready") or manifest.get("distance_metric") != "squared_l2":
        raise RuntimeError("exact dense asset manifest is not ready")

    assets: dict[str, dict[str, Any]] = {}
    for collection_name, entry in manifest.get("collections", {}).items():
        embeddings_path = asset_dir / str(entry["embeddings_file"])
        rows_path = asset_dir / str(entry["rows_file"])
        if _sha256_file(embeddings_path) != entry["embeddings_sha256"]:
            raise ValueError(f"embedding asset SHA-256 mismatch: {collection_name}")
        if _sha256_file(rows_path) != entry["rows_sha256"]:
            raise ValueError(f"row asset SHA-256 mismatch: {collection_name}")
        embeddings = np.load(embeddings_path, allow_pickle=False)
        rows = _read_jsonl(rows_path)
        if embeddings.ndim != 2 or embeddings.shape != (
            int(entry["count"]),
            int(entry["dimension"]),
        ):
            raise ValueError(f"embedding asset shape mismatch: {collection_name}")
        if len(rows) != int(entry["count"]):
            raise ValueError(f"row asset count mismatch: {collection_name}")
        assets[collection_name] = {"embeddings": embeddings, "rows": rows}
    return assets, manifest


def rank_exact_top_k(
    *,
    embeddings: np.ndarray,
    candidate_keys: np.ndarray,
    query_embedding: np.ndarray,
    top_k: int,
) -> list[dict[str, Any]]:
    """Return exact squared-L2 Top-K with candidate-key tie breaking."""
    matrix = np.asarray(embeddings, dtype=np.float64)
    keys = np.asarray(candidate_keys, dtype=str).reshape(-1)
    query = np.asarray(query_embedding, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional matrix")
    if matrix.shape[0] != keys.shape[0]:
        raise ValueError("candidate key count does not match embedding rows")
    if matrix.shape[1] != query.shape[0]:
        raise ValueError("query and document embedding dimensions differ")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    distances = np.sum((matrix - query) ** 2, axis=1, dtype=np.float64)
    ranked_indices = np.lexsort((keys, distances))[: min(top_k, len(keys))]
    return [
        {
            "candidate_key": str(keys[index]),
            "distance": float(distances[index]),
        }
        for index in ranked_indices.tolist()
    ]


def retrieve_exact_dense(
    *,
    assets: dict[str, dict[str, Any]],
    query_embedding: np.ndarray,
    top_k: int,
) -> list[dict[str, Any]]:
    """Retrieve exact Top-K candidates independently from each granularity."""
    candidates: list[dict[str, Any]] = []
    for collection_name in COLLECTIONS:
        asset = assets.get(collection_name)
        if asset is None:
            continue
        rows = list(asset["rows"])
        keys = np.asarray([str(row["candidate_key"]) for row in rows])
        rows_by_key = {str(row["candidate_key"]): row for row in rows}
        if len(rows_by_key) != len(rows):
            raise ValueError(f"duplicate candidate keys: {collection_name}")
        ranked = rank_exact_top_k(
            embeddings=np.asarray(asset["embeddings"]),
            candidate_keys=keys,
            query_embedding=query_embedding,
            top_k=top_k,
        )
        for route_rank, item in enumerate(ranked, start=1):
            row = rows_by_key[item["candidate_key"]]
            distance = float(item["distance"])
            candidates.append(
                {
                    "candidate_key": str(row["candidate_key"]),
                    "document_id": str(row["document_id"]),
                    "collection": collection_name,
                    "content": str(row.get("content", "")),
                    "source_file": str(row.get("source_file", "")),
                    "page_number": int(row.get("page_number") or 0),
                    "granularity": int(row.get("granularity") or 0),
                    "chapter_title": str(row.get("chapter_title", "")),
                    "block_type": str(row.get("block_type", "")),
                    "route": f"dense_exact:{collection_name}",
                    "route_rank": route_rank,
                    "raw_score": 1.0 - distance,
                    "distance": distance,
                }
            )
    return candidates
