"""Run the local BGE-M3 hybrid retrieval pipeline on Validation40."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


COLLECTIONS = ["detail_128", "concept_512", "context_1024"]
METHODS = [
    "bge_m3_dense",
    "bge_m3_sparse",
    "dense_sparse_rrf",
    "hybrid_reranker_dedup",
]


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
            raise ValueError(f"JSONL row must be an object at line {line_number}: {path}")
        rows.append(row)
    return rows


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", delete=False, dir=path.parent, prefix=f".{path.name}."
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def validate_frozen_hashes(
    *,
    config: dict[str, Any],
    runtime_projection_path: Path,
    dense_status_path: Path,
    sparse_manifest_path: Path,
    pilot_test_path: Path,
) -> dict[str, Any]:
    """Verify frozen assets; the Pilot Test is hashed as bytes and never parsed."""
    checks = [
        (
            "runtime projection",
            runtime_projection_path,
            "expected_runtime_projection_sha256",
        ),
        (
            "dense index status",
            dense_status_path,
            "expected_dense_index_status_sha256",
        ),
        (
            "sparse index manifest",
            sparse_manifest_path,
            "expected_sparse_index_manifest_sha256",
        ),
        ("pilot test", pilot_test_path, "expected_pilot_test_sha256"),
    ]
    observed: dict[str, str] = {}
    for label, path, config_key in checks:
        actual = sha256_file(path)
        expected = str(config.get(config_key, "")).strip().lower()
        if not expected or actual.lower() != expected:
            raise ValueError(f"{label} SHA-256 mismatch")
        observed[config_key.replace("expected_", "observed_")] = actual
    return {
        **observed,
        "pilot_test_accessed": False,
        "pilot_test_sha256_before": observed["observed_pilot_test_sha256"],
    }


def _candidate_from_fields(
    *,
    collection: str,
    document_id: str,
    content: str,
    metadata: dict[str, Any],
    route: str,
    route_rank: int,
    raw_score: float,
) -> dict[str, Any]:
    return {
        "candidate_key": f"{collection}::{document_id}",
        "document_id": document_id,
        "collection": collection,
        "content": content,
        "source_file": str(metadata.get("source_file", "")),
        "page_number": int(metadata.get("page_number") or 0),
        "granularity": int(metadata.get("granularity") or 0),
        "chapter_title": str(metadata.get("chapter_title", "")),
        "block_type": str(metadata.get("block_type", "")),
        "route": route,
        "route_rank": route_rank,
        "raw_score": float(raw_score),
    }


def retrieve_dense(
    *,
    collections: dict[str, Any],
    query_embedding: np.ndarray,
    top_k: int,
) -> list[dict[str, Any]]:
    """Retrieve Top-K dense candidates independently from each granularity."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    candidates: list[dict[str, Any]] = []
    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1).tolist()
    for collection_name in COLLECTIONS:
        collection = collections.get(collection_name)
        if collection is None:
            continue
        payload = collection.query(
            query_embeddings=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = (payload.get("ids") or [[]])[0]
        documents = (payload.get("documents") or [[]])[0]
        metadatas = (payload.get("metadatas") or [[]])[0]
        distances = (payload.get("distances") or [[]])[0]
        for rank, (document_id, content, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances), start=1
        ):
            candidates.append(
                _candidate_from_fields(
                    collection=collection_name,
                    document_id=str(document_id),
                    content=str(content or ""),
                    metadata=metadata or {},
                    route=f"dense:{collection_name}",
                    route_rank=rank,
                    raw_score=1.0 - float(distance),
                )
            )
    return candidates


def lexical_weights_to_query_csr(
    lexical_weights: dict[Any, Any], vocab_size: int
) -> sparse.csr_matrix:
    indices: list[int] = []
    values: list[float] = []
    for token_id, weight in sorted(
        (int(token_id), float(weight)) for token_id, weight in lexical_weights.items()
    ):
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError(f"Token id {token_id} is outside vocab size {vocab_size}")
        if weight != 0.0:
            indices.append(token_id)
            values.append(weight)
    indptr = np.asarray([0, len(indices)], dtype=np.int64)
    return sparse.csr_matrix(
        (
            np.asarray(values, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            indptr,
        ),
        shape=(1, vocab_size),
        dtype=np.float32,
    )


def retrieve_sparse(
    *,
    matrices: dict[str, sparse.csr_matrix],
    documents: dict[str, list[dict[str, Any]]],
    query_vector: sparse.csr_matrix,
    top_k: int,
) -> list[dict[str, Any]]:
    """Retrieve Top-K learned sparse candidates independently per granularity."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    candidates: list[dict[str, Any]] = []
    for collection_name in COLLECTIONS:
        matrix = matrices.get(collection_name)
        rows = documents.get(collection_name)
        if matrix is None or rows is None:
            continue
        if matrix.shape[0] != len(rows):
            raise ValueError(f"Sparse row mapping mismatch: {collection_name}")
        if matrix.shape[1] != query_vector.shape[1]:
            raise ValueError(f"Sparse vocabulary mismatch: {collection_name}")
        scores = np.asarray(matrix.dot(query_vector.T).toarray()).reshape(-1)
        positive = np.flatnonzero(scores > 0.0)
        if positive.size == 0:
            continue
        ranked = positive[np.argsort(-scores[positive], kind="stable")[:top_k]]
        for rank, row_index in enumerate(ranked.tolist(), start=1):
            row = rows[row_index]
            candidates.append(
                _candidate_from_fields(
                    collection=collection_name,
                    document_id=str(row["document_id"]),
                    content=str(row.get("content", "")),
                    metadata=row,
                    route=f"sparse:{collection_name}",
                    route_rank=rank,
                    raw_score=float(scores[row_index]),
                )
            )
    return candidates


def reciprocal_rank_fusion(
    candidates: list[dict[str, Any]], *, rrf_k: int, top_k: int
) -> list[dict[str, Any]]:
    """Fuse ranked lists using rank only; raw route scores are retained for audit."""
    if rrf_k < 0 or top_k <= 0:
        raise ValueError("rrf_k must be non-negative and top_k must be positive")
    fused: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        key = str(candidate["candidate_key"])
        item = fused.setdefault(
            key,
            {
                **{k: v for k, v in candidate.items() if k not in {"route", "route_rank", "raw_score"}},
                "rrf_score": 0.0,
                "route_traces": [],
            },
        )
        rank = int(candidate["route_rank"])
        item["rrf_score"] += 1.0 / (rrf_k + rank)
        item["route_traces"].append(
            {
                "route": str(candidate["route"]),
                "rank": rank,
                "raw_score": float(candidate["raw_score"]),
            }
        )
    ranked = sorted(
        fused.values(),
        key=lambda item: (-float(item["rrf_score"]), str(item["candidate_key"])),
    )[:top_k]
    for rank, item in enumerate(ranked, start=1):
        item["rrf_rank"] = rank
        item["route_traces"] = sorted(
            item["route_traces"], key=lambda trace: (trace["route"], trace["rank"])
        )
    return ranked


def rerank_candidates(
    *,
    question: str,
    candidates: list[dict[str, Any]],
    scorer: Any,
    batch_size: int,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    pairs = [[question, str(item.get("content", ""))] for item in candidates]
    scores = scorer.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(values) != len(candidates):
        raise RuntimeError("Reranker output count does not match candidate count")
    enriched = [
        {
            **item,
            "pre_rerank_rank": rank,
            "reranker_score": float(score),
        }
        for rank, (item, score) in enumerate(zip(candidates, values), start=1)
    ]
    enriched.sort(
        key=lambda item: (
            -float(item["reranker_score"]),
            -float(item.get("rrf_score", 0.0)),
            str(item["candidate_key"]),
        )
    )
    for rank, item in enumerate(enriched, start=1):
        item["post_rerank_rank"] = rank
    return enriched


def _normalize_text(text: object) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", str(text)).casefold()


def _character_ngrams(text: object, n: int) -> set[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return set()
    if len(normalized) <= n:
        return {normalized}
    return {normalized[index : index + n] for index in range(len(normalized) - n + 1)}


def _high_overlap(left: object, right: object, *, n: int, threshold: float) -> bool:
    left_ngrams = _character_ngrams(left, n)
    right_ngrams = _character_ngrams(right, n)
    if not left_ngrams or not right_ngrams:
        return False
    intersection = len(left_ngrams & right_ngrams)
    union = len(left_ngrams | right_ngrams)
    jaccard = intersection / union if union else 0.0
    overlap_coefficient = intersection / min(len(left_ngrams), len(right_ngrams))
    return max(jaccard, overlap_coefficient) >= threshold


def deduplicate_evidence(
    candidates: list[dict[str, Any]],
    *,
    max_evidence: int,
    ngram_size: int,
    overlap_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Keep ranked evidence while removing exact and same-page overlap duplicates."""
    if max_evidence <= 0 or ngram_size <= 0:
        raise ValueError("max_evidence and ngram_size must be positive")
    if not 0.0 <= overlap_threshold <= 1.0:
        raise ValueError("overlap_threshold must be within [0, 1]")
    admitted: list[dict[str, Any]] = []
    normalized_seen: set[str] = set()
    exact_duplicate_count = 0
    same_page_overlap_count = 0
    invalid_provenance_count = 0

    for candidate in candidates:
        content = str(candidate.get("content", "")).strip()
        source = str(candidate.get("source_file", "")).strip()
        page = int(candidate.get("page_number") or 0)
        if not content or not source or page <= 0:
            invalid_provenance_count += 1
            continue
        normalized = _normalize_text(content)
        if normalized in normalized_seen:
            exact_duplicate_count += 1
            continue
        overlaps = any(
            source.casefold() == str(item.get("source_file", "")).casefold()
            and page == int(item.get("page_number") or 0)
            and _high_overlap(
                content,
                item.get("content", ""),
                n=ngram_size,
                threshold=overlap_threshold,
            )
            for item in admitted
        )
        if overlaps:
            same_page_overlap_count += 1
            continue
        normalized_seen.add(normalized)
        admitted.append(dict(candidate))
        if len(admitted) >= max_evidence:
            break

    return admitted, {
        "input_count": len(candidates),
        "output_count": len(admitted),
        "exact_duplicate_count": exact_duplicate_count,
        "same_page_overlap_count": same_page_overlap_count,
        "invalid_provenance_count": invalid_provenance_count,
    }


def _release_cuda_model(model: Any) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _load_sparse_assets(
    sparse_index_dir: Path,
) -> tuple[dict[str, sparse.csr_matrix], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    manifest = _read_json(sparse_index_dir / "manifest.json")
    if not manifest.get("ready"):
        raise RuntimeError("Sparse index manifest is not ready")
    matrices = {
        name: sparse.load_npz(sparse_index_dir / manifest["files"][name]["matrix_file"])
        for name in COLLECTIONS
    }
    documents = {
        name: _read_jsonl(sparse_index_dir / manifest["files"][name]["mapping_file"])
        for name in COLLECTIONS
    }
    return matrices, documents, manifest


def _load_chroma_collections(chroma_path: Path) -> dict[str, Any]:
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_path))
    return {name: client.get_collection(name) for name in COLLECTIONS}


def _encode_dense_queries(
    *, model_path: Path, questions: list[str], batch_size: int, device: str
) -> tuple[np.ndarray, float, int]:
    import torch
    from sentence_transformers import SentenceTransformer

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = SentenceTransformer(str(model_path), device=device, local_files_only=True)
    embeddings = model.encode(
        questions,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    _release_cuda_model(model)
    return embeddings, elapsed, peak


def _encode_sparse_queries(
    *,
    model_path: Path,
    questions: list[str],
    batch_size: int,
    max_length: int,
    device: str,
) -> tuple[list[sparse.csr_matrix], float, int, int]:
    import torch
    from FlagEmbedding import BGEM3FlagModel

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = BGEM3FlagModel(
        str(model_path),
        use_fp16=True,
        devices=[device],
        batch_size=batch_size,
        query_max_length=max_length,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    outputs = model.encode(
        questions,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    vocab_size = len(model.tokenizer)
    vectors = [
        lexical_weights_to_query_csr(weights, vocab_size)
        for weights in outputs.get("lexical_weights") or []
    ]
    if len(vectors) != len(questions):
        raise RuntimeError("Sparse query output count mismatch")
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    _release_cuda_model(model)
    return vectors, elapsed, peak, vocab_size


def _load_reranker(model_path: Path, device: str) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(str(model_path), device=device, local_files_only=True)


def save_query_artifacts(
    *,
    output_dir: Path,
    dense_embeddings: np.ndarray,
    sparse_queries: list[sparse.csr_matrix],
) -> dict[str, Path]:
    """Persist the actual local encoder outputs used by the retrieval run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dense_path = output_dir / "validation40_dense_query_embeddings_v0_1.npy"
    sparse_path = output_dir / "validation40_sparse_query_vectors_v0_1.npz"
    sparse_matrix = (
        sparse.vstack(sparse_queries, format="csr")
        if sparse_queries
        else sparse.csr_matrix((0, 0), dtype=np.float32)
    )
    np.save(dense_path, np.asarray(dense_embeddings, dtype=np.float32))
    sparse.save_npz(sparse_path, sparse_matrix, compressed=True)
    return {"dense_path": dense_path, "sparse_path": sparse_path}


def _fused_method_result(
    candidates: list[dict[str, Any]],
    *,
    rrf_k: int,
    candidate_k: int,
    final_evidence_k: int,
    ngram_size: int,
    overlap_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    fused = reciprocal_rank_fusion(candidates, rrf_k=rrf_k, top_k=candidate_k)
    evidence, audit = deduplicate_evidence(
        fused,
        max_evidence=final_evidence_k,
        ngram_size=ngram_size,
        overlap_threshold=overlap_threshold,
    )
    return fused, evidence, audit


def run_validation40(
    *,
    runtime_projection_path: Path,
    dense_index_path: Path,
    sparse_index_dir: Path,
    embedding_model_path: Path,
    reranker_model_path: Path,
    pilot_test_path: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Execute four real local retrieval configurations without opening Gold labels."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    dense_status_path = dense_index_path / "index_status.json"
    sparse_manifest_path = sparse_index_dir / "manifest.json"
    frozen_hash_audit = validate_frozen_hashes(
        config=config,
        runtime_projection_path=runtime_projection_path,
        dense_status_path=dense_status_path,
        sparse_manifest_path=sparse_manifest_path,
        pilot_test_path=pilot_test_path,
    )
    rows = _read_jsonl(runtime_projection_path)
    expected_count = int(config.get("expected_count", 40))
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} runtime rows, got {len(rows)}")
    allowed_fields = {"sample_id", "question", "dataset_version", "kb_version"}
    forbidden = sorted(set().union(*(set(row) for row in rows)) - allowed_fields)
    if forbidden:
        raise ValueError(f"Runtime projection contains non-whitelisted fields: {forbidden}")

    dense_status = _read_json(dense_status_path)
    if not dense_status.get("ready") or dense_status.get("embedding_model") != "BAAI/bge-m3":
        raise RuntimeError("Dense BGE-M3 index is not ready")
    matrices, documents, sparse_manifest = _load_sparse_assets(sparse_index_dir)
    collections = _load_chroma_collections(dense_index_path)

    questions = [str(row["question"]) for row in rows]
    device = str(config.get("device", "cuda:0"))
    dense_embeddings, dense_encode_seconds, dense_peak = _encode_dense_queries(
        model_path=embedding_model_path,
        questions=questions,
        batch_size=int(config.get("dense_batch_size", 4)),
        device=device,
    )
    sparse_queries, sparse_encode_seconds, sparse_peak, sparse_vocab_size = (
        _encode_sparse_queries(
            model_path=embedding_model_path,
            questions=questions,
            batch_size=int(config.get("sparse_batch_size", 4)),
            max_length=int(config.get("query_max_length", 512)),
            device=device,
        )
    )
    if sparse_vocab_size != int(sparse_manifest["vocab_size"]):
        raise ValueError("Sparse query and passage vocabulary sizes differ")

    route_top_k = int(config.get("route_top_k", 20))
    candidate_k = int(config.get("candidate_k", 20))
    final_evidence_k = int(config.get("final_evidence_k", 4))
    rrf_k = int(config.get("rrf_k", 60))
    ngram_size = int(config.get("dedup_ngram_size", 3))
    overlap_threshold = float(config.get("dedup_overlap_threshold", 0.8))
    per_sample: list[dict[str, Any]] = []
    hybrid_candidates: list[list[dict[str, Any]]] = []

    for index, row in enumerate(rows):
        dense_started = time.perf_counter()
        dense_routes = retrieve_dense(
            collections=collections,
            query_embedding=dense_embeddings[index],
            top_k=route_top_k,
        )
        dense_seconds = time.perf_counter() - dense_started
        sparse_started = time.perf_counter()
        sparse_routes = retrieve_sparse(
            matrices=matrices,
            documents=documents,
            query_vector=sparse_queries[index],
            top_k=route_top_k,
        )
        sparse_seconds = time.perf_counter() - sparse_started

        dense_fused, dense_evidence, dense_dedup = _fused_method_result(
            dense_routes,
            rrf_k=rrf_k,
            candidate_k=candidate_k,
            final_evidence_k=final_evidence_k,
            ngram_size=ngram_size,
            overlap_threshold=overlap_threshold,
        )
        sparse_fused, sparse_evidence, sparse_dedup = _fused_method_result(
            sparse_routes,
            rrf_k=rrf_k,
            candidate_k=candidate_k,
            final_evidence_k=final_evidence_k,
            ngram_size=ngram_size,
            overlap_threshold=overlap_threshold,
        )
        fusion_started = time.perf_counter()
        hybrid_fused = reciprocal_rank_fusion(
            dense_routes + sparse_routes, rrf_k=rrf_k, top_k=candidate_k
        )
        hybrid_rrf_seconds = time.perf_counter() - fusion_started
        rrf_evidence, rrf_dedup = deduplicate_evidence(
            hybrid_fused,
            max_evidence=final_evidence_k,
            ngram_size=ngram_size,
            overlap_threshold=overlap_threshold,
        )
        hybrid_candidates.append(hybrid_fused)
        per_sample.append(
            {
                **row,
                "methods": {
                    "bge_m3_dense": {
                        "candidates_top20": dense_fused,
                        "evidence_top4": dense_evidence,
                        "dedup_audit": dense_dedup,
                        "latency_seconds": dense_seconds,
                    },
                    "bge_m3_sparse": {
                        "candidates_top20": sparse_fused,
                        "evidence_top4": sparse_evidence,
                        "dedup_audit": sparse_dedup,
                        "latency_seconds": sparse_seconds,
                    },
                    "dense_sparse_rrf": {
                        "candidates_top20": hybrid_fused,
                        "evidence_top4": rrf_evidence,
                        "dedup_audit": rrf_dedup,
                        "latency_seconds": dense_seconds + sparse_seconds + hybrid_rrf_seconds,
                    },
                },
            }
        )

    query_artifacts = save_query_artifacts(
        output_dir=output_dir,
        dense_embeddings=dense_embeddings,
        sparse_queries=sparse_queries,
    )
    del dense_embeddings, sparse_queries
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    reranker = _load_reranker(reranker_model_path, device)
    reranker_peak = 0
    for row, fused in zip(per_sample, hybrid_candidates):
        started = time.perf_counter()
        reranked = rerank_candidates(
            question=str(row["question"]),
            candidates=fused,
            scorer=reranker,
            batch_size=int(config.get("reranker_batch_size", 4)),
        )
        evidence, dedup_audit = deduplicate_evidence(
            reranked,
            max_evidence=final_evidence_k,
            ngram_size=ngram_size,
            overlap_threshold=overlap_threshold,
        )
        row["methods"]["hybrid_reranker_dedup"] = {
            "candidates_top20": reranked,
            "evidence_top4": evidence,
            "dedup_audit": dedup_audit,
            "latency_seconds": time.perf_counter() - started,
        }
        if torch.cuda.is_available():
            reranker_peak = max(reranker_peak, int(torch.cuda.max_memory_allocated()))
    _release_cuda_model(reranker)

    results_path = output_dir / "validation40_hybrid_retrieval_results_v0_1.jsonl"
    audit_path = output_dir / "validation40_hybrid_retrieval_audit_v0_1.json"
    _atomic_write(results_path, _jsonl_bytes(per_sample))
    pilot_sha256_after = sha256_file(pilot_test_path)
    if pilot_sha256_after != frozen_hash_audit["pilot_test_sha256_before"]:
        raise RuntimeError("Pilot Test80 changed during Validation40 retrieval")
    audit = {
        "execution_version": "validation40-hybrid-retrieval-v0.1",
        "dataset_version": rows[0]["dataset_version"] if rows else "",
        "kb_version": rows[0]["kb_version"] if rows else "",
        "runtime_projection_sha256": sha256_file(runtime_projection_path),
        "dense_index_status_sha256": sha256_file(dense_status_path),
        "sparse_index_manifest_sha256": sha256_file(sparse_index_dir / "manifest.json"),
        "dense_query_embeddings_sha256": sha256_file(query_artifacts["dense_path"]),
        "sparse_query_vectors_sha256": sha256_file(query_artifacts["sparse_path"]),
        "sample_count": len(rows),
        "methods": METHODS,
        "route_top_k": route_top_k,
        "candidate_k": candidate_k,
        "final_evidence_k": final_evidence_k,
        "rrf_k": rrf_k,
        "dedup_ngram_size": ngram_size,
        "dedup_overlap_threshold": overlap_threshold,
        "embedding_model_id": "BAAI/bge-m3",
        "reranker_model_id": "BAAI/bge-reranker-v2-m3",
        "dense_query_encoding_seconds": dense_encode_seconds,
        "sparse_query_encoding_seconds": sparse_encode_seconds,
        "peak_vram_bytes": {
            "dense_encoder": dense_peak,
            "sparse_encoder": sparse_peak,
            "reranker": reranker_peak,
        },
        "dense_index_bytes": sum(
            path.stat().st_size for path in dense_index_path.rglob("*") if path.is_file()
        ),
        "sparse_index_bytes": sum(
            path.stat().st_size for path in sparse_index_dir.rglob("*") if path.is_file()
        ),
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0.0,
        **frozen_hash_audit,
        "pilot_test_sha256_after": pilot_sha256_after,
    }
    _atomic_write(audit_path, _json_bytes(audit))
    return {"results": per_sample, "audit": audit}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    config = _read_json(args.config)
    root = args.repo_root.resolve()
    result = run_validation40(
        runtime_projection_path=root / config["runtime_projection_path"],
        dense_index_path=root / config["dense_index_path"],
        sparse_index_dir=root / config["sparse_index_dir"],
        embedding_model_path=Path(os.environ[config["embedding_model_path_env"]]),
        reranker_model_path=Path(os.environ[config["reranker_model_path_env"]]),
        pilot_test_path=root / config["pilot_test_path"],
        output_dir=root / config["output_dir"],
        config=config,
    )
    print(json.dumps(result["audit"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
