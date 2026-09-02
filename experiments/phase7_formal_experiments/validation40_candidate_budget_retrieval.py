"""Run isolated true Top-K candidate-budget retrieval on Validation40."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any


METHODS = ("bge_m3_dense", "bge_m3_sparse", "dense_sparse_rrf")


def _load_base_module():
    path = Path(__file__).with_name("validation40_hybrid_retrieval.py")
    spec = importlib.util.spec_from_file_location("validation40_hybrid_retrieval_base", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot load retrieval base module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base_module()


def _load_exact_module():
    path = Path(__file__).with_name("exact_dense_retrieval.py")
    spec = importlib.util.spec_from_file_location("exact_dense_retrieval", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot load exact dense module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXACT = _load_exact_module()


def validate_candidate_budgets(values: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    budgets = tuple(int(value) for value in values)
    if not budgets or any(value <= 0 for value in budgets):
        raise ValueError("candidate budgets must be positive")
    if tuple(sorted(set(budgets))) != budgets:
        raise ValueError("candidate budgets must be strictly increasing")
    return budgets


def _route_prefix(
    candidates: list[dict[str, Any]], budget: int
) -> list[dict[str, Any]]:
    return [item for item in candidates if int(item["route_rank"]) <= budget]


def build_budget_rankings(
    *,
    dense_routes: list[dict[str, Any]],
    sparse_routes: list[dict[str, Any]],
    budgets: list[int] | tuple[int, ...],
    rrf_k: int,
    dense_method_name: str = "bge_m3_dense",
    rrf_method_name: str = "dense_sparse_rrf",
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Recompute each ranking from the corresponding true route prefix."""
    validated = validate_candidate_budgets(budgets)
    rankings: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for budget in validated:
        dense_prefix = _route_prefix(dense_routes, budget)
        sparse_prefix = _route_prefix(sparse_routes, budget)
        rankings[str(budget)] = {
            dense_method_name: BASE.reciprocal_rank_fusion(
                dense_prefix, rrf_k=rrf_k, top_k=budget
            ),
            "bge_m3_sparse": BASE.reciprocal_rank_fusion(
                sparse_prefix, rrf_k=rrf_k, top_k=budget
            ),
            rrf_method_name: BASE.reciprocal_rank_fusion(
                dense_prefix + sparse_prefix, rrf_k=rrf_k, top_k=budget
            ),
        }
    return rankings


def build_sample_result(
    *,
    runtime_row: dict[str, Any],
    candidate_budgets: list[int] | tuple[int, ...],
    methods: dict[str, Any],
) -> dict[str, Any]:
    """Build deterministic raw output; runtime measurements stay in the audit."""
    return {
        **runtime_row,
        "candidate_budgets": list(candidate_budgets),
        "methods": methods,
    }


def resolve_dense_method_names(dense_backend: str) -> tuple[str, str]:
    if dense_backend == "chroma_hnsw":
        return "bge_m3_dense", "dense_sparse_rrf"
    if dense_backend == "exact_numpy":
        return "bge_m3_dense_exact", "dense_exact_sparse_rrf"
    raise ValueError(f"unsupported dense backend: {dense_backend}")


def _canonical_config_sha256(config: dict[str, Any]) -> str:
    payload = {key: value for key, value in config.items() if key != "_config_file_sha256"}
    return hashlib.sha256(BASE._json_bytes(payload)).hexdigest()


def run_candidate_budget_retrieval(
    *,
    runtime_projection_path: Path,
    dense_index_path: Path,
    exact_dense_asset_dir: Path | None,
    sparse_index_dir: Path,
    embedding_model_path: Path,
    pilot_test_path: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Retrieve candidates without opening Validation Gold or Pilot Test content."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    dense_status_path = dense_index_path / "index_status.json"
    sparse_manifest_path = sparse_index_dir / "manifest.json"
    frozen_hash_audit = BASE.validate_frozen_hashes(
        config=config,
        runtime_projection_path=runtime_projection_path,
        dense_status_path=dense_status_path,
        sparse_manifest_path=sparse_manifest_path,
        pilot_test_path=pilot_test_path,
    )
    rows = BASE._read_jsonl(runtime_projection_path)
    expected_count = int(config.get("expected_count", 40))
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} runtime rows, got {len(rows)}")
    allowed_fields = {"sample_id", "question", "dataset_version", "kb_version"}
    forbidden = sorted(set().union(*(set(row) for row in rows)) - allowed_fields)
    if forbidden:
        raise ValueError(f"Runtime projection contains non-whitelisted fields: {forbidden}")

    budgets = validate_candidate_budgets(config.get("candidate_budgets", []))
    max_route_top_k = int(config.get("max_route_top_k", budgets[-1]))
    if max_route_top_k != budgets[-1]:
        raise ValueError("max_route_top_k must equal the largest candidate budget")
    rrf_k = int(config.get("rrf_k", 60))
    device = str(config.get("device", "cuda:0"))
    dense_backend = str(config.get("dense_backend", "chroma_hnsw"))
    dense_method_name, rrf_method_name = resolve_dense_method_names(dense_backend)
    methods = (dense_method_name, "bge_m3_sparse", rrf_method_name)

    dense_status = BASE._read_json(dense_status_path)
    if not dense_status.get("ready") or dense_status.get("embedding_model") != "BAAI/bge-m3":
        raise RuntimeError("Dense BGE-M3 index is not ready")
    matrices, documents, sparse_manifest = BASE._load_sparse_assets(sparse_index_dir)
    collections: dict[str, Any] = {}
    exact_assets: dict[str, dict[str, Any]] = {}
    exact_manifest_sha256 = None
    exact_asset_bytes = 0
    if dense_backend == "chroma_hnsw":
        collections = BASE._load_chroma_collections(dense_index_path)
    else:
        if exact_dense_asset_dir is None:
            raise ValueError("exact_dense_asset_dir is required for exact_numpy")
        exact_assets, exact_manifest = EXACT.load_exact_dense_assets(
            exact_dense_asset_dir
        )
        if exact_manifest.get("source_index_status_sha256") != BASE.sha256_file(
            dense_status_path
        ):
            raise RuntimeError("exact dense assets do not match the frozen dense index")
        exact_manifest_sha256 = BASE.sha256_file(
            exact_dense_asset_dir / "manifest.json"
        )
        expected_exact_manifest_sha256 = str(
            config.get("expected_exact_dense_manifest_sha256", "")
        ).strip().lower()
        if (
            not expected_exact_manifest_sha256
            or exact_manifest_sha256 != expected_exact_manifest_sha256
        ):
            raise ValueError("exact dense manifest SHA-256 mismatch")
        exact_asset_bytes = sum(
            path.stat().st_size
            for path in exact_dense_asset_dir.rglob("*")
            if path.is_file()
        )
    questions = [str(row["question"]) for row in rows]

    dense_embeddings, dense_encode_seconds, dense_peak = BASE._encode_dense_queries(
        model_path=embedding_model_path,
        questions=questions,
        batch_size=int(config.get("dense_batch_size", 4)),
        device=device,
    )
    sparse_queries, sparse_encode_seconds, sparse_peak, sparse_vocab_size = (
        BASE._encode_sparse_queries(
            model_path=embedding_model_path,
            questions=questions,
            batch_size=int(config.get("sparse_batch_size", 4)),
            max_length=int(config.get("query_max_length", 512)),
            device=device,
        )
    )
    if sparse_vocab_size != int(sparse_manifest["vocab_size"]):
        raise ValueError("Sparse query and passage vocabulary sizes differ")

    per_sample: list[dict[str, Any]] = []
    retrieval_seconds = 0.0
    for index, row in enumerate(rows):
        started = time.perf_counter()
        if dense_backend == "exact_numpy":
            dense_routes = EXACT.retrieve_exact_dense(
                assets=exact_assets,
                query_embedding=dense_embeddings[index],
                top_k=max_route_top_k,
            )
        else:
            dense_routes = BASE.retrieve_dense(
                collections=collections,
                query_embedding=dense_embeddings[index],
                top_k=max_route_top_k,
            )
        sparse_routes = BASE.retrieve_sparse(
            matrices=matrices,
            documents=documents,
            query_vector=sparse_queries[index],
            top_k=max_route_top_k,
        )
        budget_rankings = build_budget_rankings(
            dense_routes=dense_routes,
            sparse_routes=sparse_routes,
            budgets=budgets,
            rrf_k=rrf_k,
            dense_method_name=dense_method_name,
            rrf_method_name=rrf_method_name,
        )
        elapsed = time.perf_counter() - started
        retrieval_seconds += elapsed
        methods = {
            method: {
                str(budget): budget_rankings[str(budget)][method]
                for budget in budgets
            }
            for method in methods
        }
        per_sample.append(
            build_sample_result(
                runtime_row=row,
                candidate_budgets=budgets,
                methods=methods,
            )
        )

    query_artifacts = BASE.save_query_artifacts(
        output_dir=output_dir,
        dense_embeddings=dense_embeddings,
        sparse_queries=sparse_queries,
    )
    del dense_embeddings, sparse_queries
    gc.collect()

    results_path = output_dir / str(
        config.get(
            "results_filename", "validation40_candidate_budget_results_v0_1.jsonl"
        )
    )
    audit_path = output_dir / str(
        config.get(
            "audit_filename",
            "validation40_candidate_budget_retrieval_audit_v0_1.json",
        )
    )
    BASE._atomic_write(results_path, BASE._jsonl_bytes(per_sample))
    pilot_sha256_after = BASE.sha256_file(pilot_test_path)
    if pilot_sha256_after != frozen_hash_audit["pilot_test_sha256_before"]:
        raise RuntimeError("Pilot Test80 changed during candidate-budget retrieval")
    audit = {
        "execution_version": str(
            config.get(
                "execution_version", "validation40-candidate-budget-retrieval-v0.1"
            )
        ),
        "phase": "Phase 7-C1c-4e-1",
        "dataset_version": rows[0]["dataset_version"] if rows else "",
        "kb_version": rows[0]["kb_version"] if rows else "",
        "sample_count": len(rows),
        "methods": list(methods),
        "dense_backend": dense_backend,
        "candidate_budgets": list(budgets),
        "max_route_top_k": max_route_top_k,
        "rrf_k": rrf_k,
        "runtime_projection_sha256": BASE.sha256_file(runtime_projection_path),
        "dense_index_status_sha256": BASE.sha256_file(dense_status_path),
        "sparse_index_manifest_sha256": BASE.sha256_file(sparse_manifest_path),
        "dense_query_embeddings_sha256": BASE.sha256_file(query_artifacts["dense_path"]),
        "sparse_query_vectors_sha256": BASE.sha256_file(query_artifacts["sparse_path"]),
        "results_sha256": BASE.sha256_file(results_path),
        "canonical_config_sha256": _canonical_config_sha256(config),
        "config_file_sha256": config.get("_config_file_sha256"),
        "exact_dense_manifest_sha256": exact_manifest_sha256,
        "exact_dense_asset_bytes": exact_asset_bytes,
        "dense_query_encoding_seconds": dense_encode_seconds,
        "sparse_query_encoding_seconds": sparse_encode_seconds,
        "retrieval_seconds": retrieval_seconds,
        "peak_vram_bytes": {"dense_encoder": dense_peak, "sparse_encoder": sparse_peak},
        "dense_index_bytes": sum(
            path.stat().st_size for path in dense_index_path.rglob("*") if path.is_file()
        ),
        "sparse_index_bytes": sum(
            path.stat().st_size for path in sparse_index_dir.rglob("*") if path.is_file()
        ),
        "gold_accessed": False,
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0.0,
        **frozen_hash_audit,
        "pilot_test_sha256_after": pilot_sha256_after,
    }
    BASE._atomic_write(audit_path, BASE._json_bytes(audit))
    return {"results": per_sample, "audit": audit}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    config = BASE._read_json(args.config)
    config["_config_file_sha256"] = BASE.sha256_file(args.config)
    root = args.repo_root.resolve()
    result = run_candidate_budget_retrieval(
        runtime_projection_path=root / config["runtime_projection_path"],
        dense_index_path=root / config["dense_index_path"],
        exact_dense_asset_dir=(
            root / config["exact_dense_asset_dir"]
            if config.get("exact_dense_asset_dir")
            else None
        ),
        sparse_index_dir=root / config["sparse_index_dir"],
        embedding_model_path=Path(os.environ[config["embedding_model_path_env"]]),
        pilot_test_path=root / config["pilot_test_path"],
        output_dir=root / config["output_dir"],
        config=config,
    )
    print(json.dumps(result["audit"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
