from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "validation40_hybrid_retrieval_evaluation.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "Hybrid retrieval evaluation module is missing"
    spec = importlib.util.spec_from_file_location(
        "validation40_hybrid_retrieval_evaluation", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(source: str, page: int, content: str) -> dict:
    return {"source_file": source, "page_number": page, "content": content}


def test_evaluates_candidate_recall_and_final_evidence_separately():
    module = _load_module()
    gold = {"candidate_id": "s1", "source_filename": "A.pdf", "page_number": 10}
    method = {
        "candidates_top20": [
            _candidate("B.pdf", 3, "other"),
            _candidate("A.pdf", 11, "adjacent"),
            _candidate("A.pdf", 10, "exact"),
        ],
        "evidence_top4": [
            _candidate("A.pdf", 11, "adjacent"),
            _candidate("A.pdf", 10, "exact"),
        ],
        "dedup_audit": {"input_count": 3, "output_count": 2},
        "latency_seconds": 0.25,
    }

    metrics = module.evaluate_method_result(
        sample_id="s1",
        method_id="hybrid",
        gold=gold,
        method_result=method,
        adjacent_page_tolerance=1,
        redundancy_ngram_size=3,
        redundancy_threshold=0.8,
    )

    assert metrics["candidate_source_recall_at_20"] == 1
    assert metrics["candidate_strict_source_page_recall_at_20"] == 1
    assert metrics["candidate_strict_source_page_rank"] == 3
    assert metrics["candidate_strict_source_page_mrr"] == pytest.approx(1 / 3)
    assert metrics["final_source_recall_at_4"] == 1
    assert metrics["final_strict_source_page_recall_at_4"] == 1
    assert metrics["final_adjacent_source_page_recall_at_4"] == 1


def test_redundancy_rate_counts_high_overlap_pairs():
    module = _load_module()
    evidence = [
        _candidate("A.pdf", 1, "治疗后48-72小时无改善应再次评估"),
        _candidate("A.pdf", 1, "治疗后48-72小时无改善应再次评估原因"),
        _candidate("B.pdf", 2, "独立证据"),
    ]

    audit = module.measure_redundancy(
        evidence, ngram_size=3, threshold=0.75
    )

    assert audit["pair_count"] == 3
    assert audit["redundant_pair_count"] == 1
    assert audit["redundant_pair_rate"] == pytest.approx(1 / 3)


def test_selects_winner_by_declared_metric_order():
    module = _load_module()
    summaries = [
        {
            "method_id": "fast",
            "final_strict_source_page_recall_at_4": 0.5,
            "candidate_strict_source_page_mrr": 0.6,
            "final_source_recall_at_4": 0.8,
            "mean_redundant_pair_rate": 0.0,
            "mean_latency_seconds": 0.1,
        },
        {
            "method_id": "better_mrr",
            "final_strict_source_page_recall_at_4": 0.5,
            "candidate_strict_source_page_mrr": 0.7,
            "final_source_recall_at_4": 0.7,
            "mean_redundant_pair_rate": 0.1,
            "mean_latency_seconds": 0.2,
        },
    ]

    winner = module.select_strong_non_graph_config(summaries)

    assert winner["method_id"] == "better_mrr"
    assert winner["selection_rule"] == [
        "final_strict_source_page_recall_at_4",
        "candidate_strict_source_page_mrr",
        "final_source_recall_at_4",
        "mean_redundant_pair_rate",
        "mean_latency_seconds",
    ]


def test_join_fails_closed_when_retrieval_sample_is_missing():
    module = _load_module()
    gold_rows = [{"candidate_id": "s1"}, {"candidate_id": "s2"}]
    retrieval_rows = [{"sample_id": "s1"}]

    with pytest.raises(ValueError, match="sample IDs do not match"):
        module.join_gold_and_retrieval(gold_rows, retrieval_rows)


def test_hybrid_latency_includes_retrieval_and_reranking_stages():
    module = _load_module()
    methods = {
        "dense_sparse_rrf": {"latency_seconds": 0.30},
        "hybrid_reranker_dedup": {"latency_seconds": 0.20},
    }

    effective = module.effective_method_result(
        "hybrid_reranker_dedup", methods
    )

    assert effective["retrieval_latency_seconds"] == pytest.approx(0.30)
    assert effective["reranker_latency_seconds"] == pytest.approx(0.20)
    assert effective["latency_seconds"] == pytest.approx(0.50)


def test_frozen_config_preserves_models_indexes_and_retrieval_parameters():
    module = _load_module()
    winner = {"method_id": "hybrid_reranker_dedup", "sample_count": 40}
    retrieval_audit = {
        "embedding_model_id": "BAAI/bge-m3",
        "reranker_model_id": "BAAI/bge-reranker-v2-m3",
        "dense_index_status_sha256": "dense-hash",
        "sparse_index_manifest_sha256": "sparse-hash",
        "candidate_k": 20,
        "route_top_k": 20,
        "rrf_k": 60,
        "final_evidence_k": 4,
        "dedup_ngram_size": 3,
        "dedup_overlap_threshold": 0.8,
        "pilot_test_sha256_before": "pilot-hash",
    }

    frozen = module.build_frozen_config(
        winner=winner,
        retrieval_audit=retrieval_audit,
        gold_sha256="gold-hash",
        retrieval_sha256="retrieval-hash",
    )

    assert frozen["selected_method"] == "hybrid_reranker_dedup"
    assert frozen["retrieval_contract"]["embedding_model_id"] == "BAAI/bge-m3"
    assert frozen["retrieval_contract"]["rrf_k"] == 60
    assert frozen["retrieval_contract"]["sparse_index_manifest_sha256"] == "sparse-hash"
    assert frozen["input_hashes"]["pilot_test_sha256"] == "pilot-hash"
