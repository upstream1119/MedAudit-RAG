from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


DEFAULT_COLLECTIONS = ["detail_128", "concept_512", "context_1024"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_chroma_rows(collection: Any, collection_name: str) -> list[dict[str, Any]]:
    payload = collection.get(include=["documents", "metadatas"])
    rows = []
    for document_id, content, metadata in zip(
        payload.get("ids") or [],
        payload.get("documents") or [],
        payload.get("metadatas") or [],
    ):
        metadata = metadata or {}
        rows.append(
            {
                "document_id": str(document_id),
                "collection": collection_name,
                "content": content or "",
                "source_file": metadata.get("source_file", ""),
                "page_number": int(metadata.get("page_number") or 0),
                "granularity": int(metadata.get("granularity") or 0),
                "source_hash": metadata.get("source_hash", ""),
                "chapter_title": metadata.get("chapter_title", ""),
                "block_type": metadata.get("block_type", ""),
            }
        )
    return sorted(rows, key=lambda row: row["document_id"])


def lexical_weights_to_csr(
    lexical_weights: list[dict[Any, Any]], vocab_size: int
) -> sparse.csr_matrix:
    data: list[float] = []
    indices: list[int] = []
    indptr = [0]
    for weights in lexical_weights:
        normalized = sorted((int(token_id), float(weight)) for token_id, weight in weights.items())
        for token_id, weight in normalized:
            if token_id < 0 or token_id >= vocab_size:
                raise ValueError(f"Token id {token_id} is outside vocab size {vocab_size}")
            if weight != 0.0:
                indices.append(token_id)
                data.append(weight)
        indptr.append(len(data))
    return sparse.csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(lexical_weights), vocab_size),
        dtype=np.float32,
    )


def encode_collection_weights(
    *,
    encoder: Any,
    rows: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
    collection_name: str,
) -> list[dict[Any, Any]]:
    outputs = encoder.encode(
        [row["content"] for row in rows],
        batch_size=batch_size,
        max_length=max_length,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    weights = outputs.get("lexical_weights") or []
    if len(weights) != len(rows):
        raise RuntimeError(
            f"Sparse output count mismatch for {collection_name}: "
            f"expected {len(rows)}, got {len(weights)}"
        )
    return weights


def audit_sparse_index(
    *,
    rows_by_collection: dict[str, list[dict[str, Any]]],
    matrices_by_collection: dict[str, sparse.csr_matrix],
    expected_collections: list[str],
    expected_sources: set[str],
) -> dict[str, Any]:
    missing_collections = sorted(set(expected_collections) - set(rows_by_collection))
    all_ids: list[str] = []
    empty_vector_count = 0
    row_count_mismatches: dict[str, dict[str, int]] = {}
    missing_sources_by_collection: dict[str, list[str]] = {}
    collections: dict[str, Any] = {}

    for collection_name in expected_collections:
        rows = rows_by_collection.get(collection_name, [])
        matrix = matrices_by_collection.get(collection_name)
        matrix_rows = int(matrix.shape[0]) if matrix is not None else 0
        if len(rows) != matrix_rows:
            row_count_mismatches[collection_name] = {
                "mapping_rows": len(rows),
                "matrix_rows": matrix_rows,
            }
        present_sources = {row.get("source_file", "") for row in rows if row.get("source_file")}
        missing_sources_by_collection[collection_name] = sorted(
            expected_sources - present_sources
        )
        collection_empty_count = (
            int(np.count_nonzero(np.diff(matrix.indptr) == 0)) if matrix is not None else len(rows)
        )
        empty_vector_count += collection_empty_count
        all_ids.extend(row.get("document_id", "") for row in rows)
        collections[collection_name] = {
            "row_count": len(rows),
            "matrix_shape": list(matrix.shape) if matrix is not None else [0, 0],
            "nnz": int(matrix.nnz) if matrix is not None else 0,
            "empty_vector_count": collection_empty_count,
            "source_count": len(present_sources),
            "missing_sources": missing_sources_by_collection[collection_name],
        }

    id_counts = Counter(all_ids)
    duplicate_count = sum(count - 1 for count in id_counts.values() if count > 1)
    missing_sources = sorted(
        expected_sources
        - {
            row.get("source_file", "")
            for rows in rows_by_collection.values()
            for row in rows
            if row.get("source_file")
        }
    )
    ready = not any(
        (
            missing_collections,
            row_count_mismatches,
            missing_sources,
            any(missing_sources_by_collection.values()),
            empty_vector_count,
            duplicate_count,
        )
    )
    return {
        "ready": ready,
        "expected_collections": expected_collections,
        "missing_collections": missing_collections,
        "expected_source_count": len(expected_sources),
        "missing_sources": missing_sources,
        "missing_sources_by_collection": missing_sources_by_collection,
        "row_count_mismatches": row_count_mismatches,
        "empty_vector_count": empty_vector_count,
        "duplicate_document_id_count": duplicate_count,
        "collections": collections,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_sparse_index(
    *,
    chroma_path: Path,
    model_path: Path,
    output_dir: Path,
    batch_size: int,
    max_length: int,
    device: str,
) -> dict[str, Any]:
    import chromadb
    from FlagEmbedding import BGEM3FlagModel

    index_status_path = chroma_path / "index_status.json"
    index_status = json.loads(index_status_path.read_text(encoding="utf-8"))
    if not index_status.get("ready"):
        raise RuntimeError("Dense BGE-M3 index is not ready")
    if index_status.get("embedding_model") != "BAAI/bge-m3":
        raise ValueError("Sparse source index must be the BGE-M3 dense index")
    expected_sources = set(index_status.get("expected_sources") or [])
    if not expected_sources:
        raise ValueError("Dense index status has no expected sources")

    client = chromadb.PersistentClient(path=str(chroma_path))
    rows_by_collection = {
        name: export_chroma_rows(client.get_collection(name), name)
        for name in DEFAULT_COLLECTIONS
    }
    encoder = BGEM3FlagModel(
        str(model_path),
        use_fp16=True,
        devices=[device],
        batch_size=batch_size,
        passage_max_length=max_length,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    vocab_size = len(encoder.tokenizer)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrices_by_collection: dict[str, sparse.csr_matrix] = {}
    file_records: dict[str, Any] = {}

    for collection_name in DEFAULT_COLLECTIONS:
        rows = rows_by_collection[collection_name]
        all_weights = encode_collection_weights(
            encoder=encoder,
            rows=rows,
            batch_size=batch_size,
            max_length=max_length,
            collection_name=collection_name,
        )
        matrix = lexical_weights_to_csr(all_weights, vocab_size)
        matrices_by_collection[collection_name] = matrix
        matrix_path = output_dir / f"{collection_name}.npz"
        mapping_path = output_dir / f"{collection_name}_documents.jsonl"
        sparse.save_npz(matrix_path, matrix, compressed=True)
        _write_jsonl(mapping_path, rows)
        file_records[collection_name] = {
            "matrix_file": matrix_path.name,
            "matrix_sha256": sha256_file(matrix_path),
            "matrix_bytes": matrix_path.stat().st_size,
            "mapping_file": mapping_path.name,
            "mapping_sha256": sha256_file(mapping_path),
            "mapping_bytes": mapping_path.stat().st_size,
        }

    audit = audit_sparse_index(
        rows_by_collection=rows_by_collection,
        matrices_by_collection=matrices_by_collection,
        expected_collections=DEFAULT_COLLECTIONS,
        expected_sources=expected_sources,
    )
    manifest = {
        "index_version": "bge-m3-sparse-index-v0.1",
        "ready": audit["ready"],
        "model_id": "BAAI/bge-m3",
        "model_path": str(model_path),
        "device": device,
        "batch_size": batch_size,
        "max_length": max_length,
        "vocab_size": vocab_size,
        "chroma_path": str(chroma_path),
        "dense_index_status_sha256": sha256_file(index_status_path),
        "files": file_records,
        "audit": audit,
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0.0,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if not audit["ready"]:
        raise RuntimeError("Sparse index audit failed; manifest retained for diagnosis")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chroma-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    manifest = build_sparse_index(
        chroma_path=args.chroma_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    print(json.dumps(manifest["audit"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
