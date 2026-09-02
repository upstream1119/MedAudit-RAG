"""Export deterministic exact-dense assets from the frozen BGE-M3 ChromaDB."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


COLLECTIONS = ("detail_128", "concept_512", "context_1024")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    path.write_text(payload, encoding="utf-8", newline="\n")


def _export_collection(
    *, collection_name: str, collection: Any, batch_size: int
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    records: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    total = int(collection.count())
    for offset in range(0, total, batch_size):
        payload = collection.get(
            limit=min(batch_size, total - offset),
            offset=offset,
            include=["embeddings", "documents", "metadatas"],
        )
        ids = list(payload.get("ids") or [])
        embeddings_value = payload.get("embeddings")
        embeddings = [] if embeddings_value is None else list(embeddings_value)
        documents = list(payload.get("documents") or [])
        metadatas = list(payload.get("metadatas") or [])
        if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
            raise ValueError(f"Chroma batch fields are misaligned: {collection_name}")
        for document_id, embedding, content, metadata_value in zip(
            ids, embeddings, documents, metadatas
        ):
            metadata = dict(metadata_value or {})
            key = f"{collection_name}::{document_id}"
            row = {
                **metadata,
                "candidate_key": key,
                "document_id": str(document_id),
                "collection": collection_name,
                "content": str(content or ""),
            }
            records.append((key, np.asarray(embedding, dtype=np.float32), row))
    if len(records) != total:
        raise ValueError(f"Chroma export count mismatch: {collection_name}")
    records.sort(key=lambda item: item[0])
    keys = [item[0] for item in records]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate candidate keys: {collection_name}")
    if not records:
        raise ValueError(f"empty dense collection: {collection_name}")
    matrix = np.stack([item[1] for item in records]).astype(np.float32, copy=False)
    return matrix, [item[2] for item in records]


def build_exact_dense_assets(
    *,
    collections: dict[str, Any],
    output_dir: Path,
    source_status_path: Path,
    batch_size: int,
) -> dict[str, Any]:
    """Write sorted vectors, row mappings and a hash-audited manifest."""
    if not source_status_path.is_file():
        raise FileNotFoundError(f"dense index status is missing: {source_status_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"exact dense asset directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_collections: dict[str, dict[str, Any]] = {}
    for collection_name in COLLECTIONS:
        collection = collections.get(collection_name)
        if collection is None:
            continue
        embeddings, rows = _export_collection(
            collection_name=collection_name,
            collection=collection,
            batch_size=batch_size,
        )
        embeddings_name = f"{collection_name}_embeddings.npy"
        rows_name = f"{collection_name}_rows.jsonl"
        embeddings_path = output_dir / embeddings_name
        rows_path = output_dir / rows_name
        np.save(embeddings_path, embeddings, allow_pickle=False)
        _write_jsonl(rows_path, rows)
        manifest_collections[collection_name] = {
            "count": int(embeddings.shape[0]),
            "dimension": int(embeddings.shape[1]),
            "embeddings_file": embeddings_name,
            "embeddings_sha256": sha256_file(embeddings_path),
            "rows_file": rows_name,
            "rows_sha256": sha256_file(rows_path),
        }

    manifest = {
        "asset_version": "bge-m3-exact-dense-assets-v0.1",
        "ready": True,
        "distance_metric": "squared_l2",
        "source_index_status_sha256": sha256_file(source_status_path),
        "collections": manifest_collections,
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0.0,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chroma-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    import chromadb

    client = chromadb.PersistentClient(path=str(args.chroma_path.resolve()))
    collections = {name: client.get_collection(name) for name in COLLECTIONS}
    manifest = build_exact_dense_assets(
        collections=collections,
        output_dir=args.output_dir.resolve(),
        source_status_path=args.chroma_path.resolve() / "index_status.json",
        batch_size=args.batch_size,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
