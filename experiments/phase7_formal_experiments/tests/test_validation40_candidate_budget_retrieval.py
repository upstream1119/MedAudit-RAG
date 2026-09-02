from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "validation40_candidate_budget_retrieval.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "Candidate budget retrieval module is missing"
    spec = importlib.util.spec_from_file_location(
        "validation40_candidate_budget_retrieval", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(route: str, rank: int, key: str) -> dict:
    return {
        "collection": "detail_128",
        "route_rank": rank,
        "candidate_key": key,
        "source_file": "A.pdf",
        "page_number": rank,
        "content": key,
        "route": f"{route}:detail_128",
        "raw_score": 1.0 / rank,
    }


def test_validates_sorted_unique_candidate_budgets():
    module = _load_module()

    assert module.validate_candidate_budgets([20, 40, 80]) == (20, 40, 80)

    with pytest.raises(ValueError, match="strictly increasing"):
        module.validate_candidate_budgets([20, 20, 80])
    with pytest.raises(ValueError, match="positive"):
        module.validate_candidate_budgets([0, 20])


def test_budget_rankings_recompute_rrf_from_true_route_prefixes():
    module = _load_module()
    dense_routes = [
        _candidate("dense", 1, "shared"),
        _candidate("dense", 2, "dense-2"),
        _candidate("dense", 3, "dense-3"),
    ]
    sparse_routes = [
        _candidate("sparse", 1, "sparse-1"),
        _candidate("sparse", 2, "sparse-2"),
        _candidate("sparse", 3, "shared"),
    ]

    rankings = module.build_budget_rankings(
        dense_routes=dense_routes,
        sparse_routes=sparse_routes,
        budgets=[2, 3],
        rrf_k=60,
    )

    assert [item["candidate_key"] for item in rankings["2"]["bge_m3_dense"]] == [
        "shared",
        "dense-2",
    ]
    assert [item["candidate_key"] for item in rankings["2"]["bge_m3_sparse"]] == [
        "sparse-1",
        "sparse-2",
    ]
    shared_at_2 = next(
        item for item in rankings["2"]["dense_sparse_rrf"]
        if item["candidate_key"] == "shared"
    )
    assert [trace["route"] for trace in shared_at_2["route_traces"]] == [
        "dense:detail_128"
    ]
    assert rankings["3"]["dense_sparse_rrf"][0]["candidate_key"] == "shared"
    assert [
        (trace["route"], trace["rank"])
        for trace in rankings["3"]["dense_sparse_rrf"][0]["route_traces"]
    ] == [("dense:detail_128", 1), ("sparse:detail_128", 3)]
    assert rankings["3"]["dense_sparse_rrf"][0]["rrf_score"] > shared_at_2[
        "rrf_score"
    ]


def test_budget_rankings_are_not_route_union_or_zero_padded():
    module = _load_module()
    rankings = module.build_budget_rankings(
        dense_routes=[_candidate("dense", 1, "dense-only")],
        sparse_routes=[_candidate("sparse", 1, "sparse-only")],
        budgets=[1, 3],
        rrf_k=60,
    )

    assert len(rankings["1"]["bge_m3_dense"]) == 1
    assert len(rankings["3"]["bge_m3_dense"]) == 1
    assert len(rankings["3"]["dense_sparse_rrf"]) == 2
    assert rankings["3"]["dense_sparse_rrf"] != (
        rankings["3"]["bge_m3_dense"] + rankings["3"]["bge_m3_sparse"]
    )


def test_sample_result_excludes_nondeterministic_runtime_measurements():
    module = _load_module()
    methods = {"bge_m3_dense": {"20": [{"candidate_key": "a"}]}}

    result = module.build_sample_result(
        runtime_row={"sample_id": "S1", "question": "Q"},
        candidate_budgets=[20, 40, 80],
        methods=methods,
    )

    assert result == {
        "sample_id": "S1",
        "question": "Q",
        "candidate_budgets": [20, 40, 80],
        "methods": methods,
    }
    assert "retrieval_latency_seconds" not in result


def test_budget_rankings_can_label_exact_dense_as_a_separate_control():
    module = _load_module()
    rankings = module.build_budget_rankings(
        dense_routes=[_candidate("dense_exact", 1, "dense-only")],
        sparse_routes=[_candidate("sparse", 1, "sparse-only")],
        budgets=[2],
        rrf_k=60,
        dense_method_name="bge_m3_dense_exact",
        rrf_method_name="dense_exact_sparse_rrf",
    )

    assert [item["candidate_key"] for item in rankings["2"]["bge_m3_dense_exact"]] == [
        "dense-only"
    ]
    assert [
        item["candidate_key"] for item in rankings["2"]["dense_exact_sparse_rrf"]
    ] == ["dense-only", "sparse-only"]


def test_resolves_exact_dense_method_names_without_overwriting_hnsw_baseline():
    module = _load_module()

    assert module.resolve_dense_method_names("chroma_hnsw") == (
        "bge_m3_dense",
        "dense_sparse_rrf",
    )
    assert module.resolve_dense_method_names("exact_numpy") == (
        "bge_m3_dense_exact",
        "dense_exact_sparse_rrf",
    )
    with pytest.raises(ValueError, match="unsupported dense backend"):
        module.resolve_dense_method_names("unknown")
