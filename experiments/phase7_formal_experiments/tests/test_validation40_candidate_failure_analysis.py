from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "validation40_candidate_failure_analysis.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "Candidate failure analysis module is missing"
    spec = importlib.util.spec_from_file_location(
        "validation40_candidate_failure_analysis", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(source: str, page: int, key: str) -> dict:
    return {
        "source_file": source,
        "page_number": page,
        "candidate_key": key,
        "content": f"evidence-{key}",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_classifies_dense_hit_dropped_by_rrf_as_fusion_failure():
    module = _load_module()
    gold = {"source_filename": "A.pdf", "page_number": 10}
    methods = {
        "bge_m3_dense": {
            "candidates_top20": [_candidate("A.pdf", 10, "dense-gold")]
        },
        "bge_m3_sparse": {
            "candidates_top20": [_candidate("B.pdf", 3, "sparse-other")]
        },
        "dense_sparse_rrf": {
            "candidates_top20": [_candidate("B.pdf", 3, "rrf-other")]
        },
    }

    diagnosis = module.classify_candidate_failure(
        gold=gold,
        methods=methods,
        adjacent_page_tolerance=1,
    )

    assert diagnosis["primary_failure_type"] == "fusion_dropped_valid_dense"
    assert diagnosis["dense_exact_page_hit"] is True
    assert diagnosis["rrf_exact_page_hit"] is False
    assert diagnosis["all_routes_exact_page_miss"] is False


def test_classifies_rrf_adjacent_page_without_exact_page():
    module = _load_module()
    gold = {"source_filename": "A.pdf", "page_number": 10}
    methods = {
        "bge_m3_dense": {
            "candidates_top20": [_candidate("A.pdf", 9, "dense-adjacent")]
        },
        "bge_m3_sparse": {
            "candidates_top20": [_candidate("B.pdf", 3, "sparse-other")]
        },
        "dense_sparse_rrf": {
            "candidates_top20": [_candidate("A.pdf", 9, "rrf-adjacent")]
        },
    }

    diagnosis = module.classify_candidate_failure(
        gold=gold,
        methods=methods,
        adjacent_page_tolerance=1,
    )

    assert diagnosis["primary_failure_type"] == "adjacent_page_only"
    assert diagnosis["rrf_source_hit"] is True
    assert diagnosis["rrf_adjacent_page_hit"] is True
    assert diagnosis["all_routes_exact_page_miss"] is True


def test_classifies_same_source_non_adjacent_page_as_page_absent():
    module = _load_module()
    gold = {"source_filename": "A.pdf", "page_number": 10}
    methods = {
        "bge_m3_dense": {
            "candidates_top20": [_candidate("A.pdf", 20, "dense-far")]
        },
        "bge_m3_sparse": {
            "candidates_top20": [_candidate("B.pdf", 3, "sparse-other")]
        },
        "dense_sparse_rrf": {
            "candidates_top20": [_candidate("A.pdf", 20, "rrf-far")]
        },
    }

    diagnosis = module.classify_candidate_failure(
        gold=gold,
        methods=methods,
        adjacent_page_tolerance=1,
    )

    assert diagnosis["primary_failure_type"] == "source_present_page_absent"
    assert diagnosis["rrf_source_hit"] is True
    assert diagnosis["rrf_adjacent_page_hit"] is False


def test_classifies_missing_gold_source_as_source_absent():
    module = _load_module()
    gold = {"source_filename": "A.pdf", "page_number": 10}
    methods = {
        "bge_m3_dense": {
            "candidates_top20": [_candidate("B.pdf", 3, "dense-other")]
        },
        "bge_m3_sparse": {
            "candidates_top20": [_candidate("C.pdf", 7, "sparse-other")]
        },
        "dense_sparse_rrf": {
            "candidates_top20": [_candidate("B.pdf", 3, "rrf-other")]
        },
    }

    diagnosis = module.classify_candidate_failure(
        gold=gold,
        methods=methods,
        adjacent_page_tolerance=1,
    )

    assert diagnosis["primary_failure_type"] == "source_absent"
    assert diagnosis["rrf_source_hit"] is False
    assert diagnosis["all_routes_exact_page_miss"] is True


def test_route_union_is_deterministic_and_deduplicates_candidate_keys():
    module = _load_module()
    dense_shared = _candidate("A.pdf", 10, "shared")
    sparse_shared = _candidate("A.pdf", 10, "shared")
    methods = {
        "bge_m3_dense": {
            "candidates_top20": [dense_shared, _candidate("B.pdf", 2, "dense-only")]
        },
        "bge_m3_sparse": {
            "candidates_top20": [sparse_shared, _candidate("C.pdf", 3, "sparse-only")]
        },
        "dense_sparse_rrf": {
            "candidates_top20": [_candidate("A.pdf", 10, "shared")]
        },
    }

    first = module.build_route_union(methods)
    second = module.build_route_union(methods)

    assert first == second
    assert [item["candidate_key"] for item in first] == [
        "shared",
        "dense-only",
        "sparse-only",
    ]
    assert first[0]["union_routes"] == ["rrf", "dense", "sparse"]


def test_source_page_aggregation_collapses_overlapping_chunk_identities():
    module = _load_module()
    candidates = [
        _candidate("A.pdf", 10, "concept-a"),
        _candidate("A.pdf", 10, "detail-a"),
        _candidate("A.pdf", 11, "adjacent-a"),
    ]

    aggregated = module.aggregate_source_pages(candidates)

    assert len(aggregated) == 2
    assert aggregated[0]["source_file"] == "A.pdf"
    assert aggregated[0]["page_number"] == 10
    assert aggregated[0]["source_page_member_count"] == 2
    assert aggregated[0]["source_page_member_keys"] == ["concept-a", "detail-a"]
    assert aggregated[1]["source_page_member_count"] == 1


def test_budget_audit_does_not_mislabel_route_union_as_true_top40_or_top80():
    module = _load_module()
    methods = {
        "bge_m3_dense": {
            "candidates_top20": [
                _candidate("A.pdf", index, f"dense-{index}") for index in range(20)
            ]
        },
        "bge_m3_sparse": {
            "candidates_top20": [
                _candidate("B.pdf", index, f"sparse-{index}") for index in range(20)
            ]
        },
        "dense_sparse_rrf": {
            "candidates_top20": [
                _candidate("A.pdf", index, f"dense-{index}") for index in range(20)
            ]
        },
    }

    audit = module.audit_candidate_budget_availability(
        methods=methods,
        requested_budgets=[20, 40, 80],
    )

    assert audit["single_route_exposed_k"] == 20
    assert audit["route_union_exposed_count"] == 40
    assert audit["true_retrieval_budget_status"]["20"] == "available"
    assert audit["true_retrieval_budget_status"]["40"] == "requires_rerun"
    assert audit["true_retrieval_budget_status"]["80"] == "requires_rerun"
    assert audit["route_union_is_true_top_k"] is False


def test_run_analysis_writes_failure_taxonomy_and_non_graph_control_audit(tmp_path: Path):
    module = _load_module()
    gold_path = tmp_path / "gold.jsonl"
    retrieval_path = tmp_path / "retrieval.jsonl"
    retrieval_audit_path = tmp_path / "retrieval_audit.json"
    output_dir = tmp_path / "outputs"

    _write_jsonl(
        gold_path,
        [
            {"candidate_id": "hit", "source_filename": "A.pdf", "page_number": 10},
            {"candidate_id": "miss", "source_filename": "A.pdf", "page_number": 20},
        ],
    )
    _write_jsonl(
        retrieval_path,
        [
            {
                "sample_id": "hit",
                "question": "hit question",
                "methods": {
                    "bge_m3_dense": {"candidates_top20": [_candidate("A.pdf", 10, "h")]},
                    "bge_m3_sparse": {"candidates_top20": [_candidate("A.pdf", 10, "h")]},
                    "dense_sparse_rrf": {"candidates_top20": [_candidate("A.pdf", 10, "h")]},
                },
            },
            {
                "sample_id": "miss",
                "question": "miss question",
                "methods": {
                    "bge_m3_dense": {"candidates_top20": [_candidate("A.pdf", 19, "a19")]},
                    "bge_m3_sparse": {"candidates_top20": [_candidate("B.pdf", 2, "b2")]},
                    "dense_sparse_rrf": {"candidates_top20": [_candidate("A.pdf", 19, "a19")]},
                },
            },
        ],
    )
    retrieval_audit_path.write_text(
        json.dumps({"pilot_test_accessed": False}), encoding="utf-8"
    )
    config = {
        "analysis_version": "test-v0.1",
        "expected_gold_sha256": _sha256(gold_path),
        "expected_retrieval_sha256": _sha256(retrieval_path),
        "expected_retrieval_audit_sha256": _sha256(retrieval_audit_path),
        "adjacent_page_tolerance": 1,
        "requested_candidate_budgets": [1, 2],
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0.0,
    }

    result = module.run_analysis(
        gold_path=gold_path,
        retrieval_path=retrieval_path,
        retrieval_audit_path=retrieval_audit_path,
        output_dir=output_dir,
        config=config,
    )

    assert len(result["failure_cases"]) == 1
    assert result["failure_cases"][0]["sample_id"] == "miss"
    assert result["failure_cases"][0]["primary_failure_type"] == "adjacent_page_only"
    assert result["taxonomy"]["rrf_strict_page_miss_count"] == 1
    assert result["taxonomy"]["rrf_source_absent_count"] == 0
    assert result["taxonomy"]["rrf_source_present_page_absent_count"] == 1
    assert result["taxonomy"]["rrf_adjacent_page_only_count"] == 1
    assert result["taxonomy"]["dense_exact_rrf_miss_count"] == 0
    assert result["controls"]["mean_route_union_candidate_count"] == pytest.approx(1.5)
    assert result["controls"]["mean_source_page_aggregate_count"] == pytest.approx(1.5)
    assert result["controls"]["mean_source_page_duplicates_removed"] == pytest.approx(0.0)
    assert result["audit"]["pilot_test_accessed"] is False
    assert result["audit"]["external_model_calls"] == 0
    assert (output_dir / "candidate_failure_cases_v0_1.jsonl").exists()
    assert (output_dir / "non_graph_candidate_controls_v0_1.json").exists()
    assert (output_dir / "candidate_failure_audit_v0_1.json").exists()


def test_run_analysis_fails_closed_on_retrieval_audit_hash_mismatch(tmp_path: Path):
    module = _load_module()
    gold_path = tmp_path / "gold.jsonl"
    retrieval_path = tmp_path / "retrieval.jsonl"
    retrieval_audit_path = tmp_path / "retrieval_audit.json"
    _write_jsonl(
        gold_path,
        [{"candidate_id": "s1", "source_filename": "A.pdf", "page_number": 10}],
    )
    _write_jsonl(
        retrieval_path,
        [
            {
                "sample_id": "s1",
                "methods": {
                    "bge_m3_dense": {"candidates_top20": [_candidate("A.pdf", 10, "a")]},
                    "bge_m3_sparse": {"candidates_top20": [_candidate("A.pdf", 10, "a")]},
                    "dense_sparse_rrf": {"candidates_top20": [_candidate("A.pdf", 10, "a")]},
                },
            }
        ],
    )
    retrieval_audit_path.write_text(
        json.dumps({"pilot_test_accessed": False}), encoding="utf-8"
    )
    config = {
        "analysis_version": "test-v0.1",
        "expected_gold_sha256": _sha256(gold_path),
        "expected_retrieval_sha256": _sha256(retrieval_path),
        "expected_retrieval_audit_sha256": "0" * 64,
    }

    with pytest.raises(ValueError, match="retrieval audit SHA-256 mismatch"):
        module.run_analysis(
            gold_path=gold_path,
            retrieval_path=retrieval_path,
            retrieval_audit_path=retrieval_audit_path,
            output_dir=tmp_path / "outputs",
            config=config,
        )
