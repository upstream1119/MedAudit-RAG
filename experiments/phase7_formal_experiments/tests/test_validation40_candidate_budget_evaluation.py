from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "validation40_candidate_budget_evaluation.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "Candidate budget evaluation module is missing"
    spec = importlib.util.spec_from_file_location(
        "validation40_candidate_budget_evaluation", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(source: str, page: int) -> dict:
    return {"source_file": source, "page_number": page}


def _budget_metric(hit: int) -> dict:
    return {
        "strict_source_page_recall": hit,
        "source_recall": hit,
        "adjacent_source_page_recall": hit,
        "strict_source_page_mrr": float(hit),
    }


def test_evaluates_exact_source_page_gain_at_deeper_budget():
    module = _load_module()
    gold = {"candidate_id": "s1", "source_filename": "A.pdf", "page_number": 10}
    rankings = {
        "20": [_candidate("B.pdf", page) for page in range(1, 21)],
        "40": [
            *[_candidate("B.pdf", page) for page in range(1, 21)],
            _candidate("A.pdf", 10),
        ],
        "80": [
            *[_candidate("B.pdf", page) for page in range(1, 21)],
            _candidate("A.pdf", 10),
            _candidate("A.pdf", 11),
        ],
    }

    metrics = module.evaluate_budget_rankings(
        sample_id="s1",
        method_id="dense_sparse_rrf",
        gold=gold,
        rankings=rankings,
        budgets=[20, 40, 80],
        adjacent_page_tolerance=1,
    )

    assert metrics["budgets"]["20"]["strict_source_page_recall"] == 0
    assert metrics["budgets"]["40"]["strict_source_page_recall"] == 1
    assert metrics["budgets"]["40"]["strict_source_page_rank"] == 21
    assert metrics["budgets"]["40"]["strict_source_page_mrr"] == pytest.approx(1 / 21)
    assert metrics["first_strict_hit_budget"] == 40


def test_summary_counts_incremental_recovery_without_double_counting():
    module = _load_module()
    rows = [
        {
            "sample_id": "a",
            "method_id": "dense",
                "budgets": {
                    "20": _budget_metric(1),
                    "40": _budget_metric(1),
                    "80": _budget_metric(1),
            },
            "first_strict_hit_budget": 20,
        },
        {
            "sample_id": "b",
            "method_id": "dense",
                "budgets": {
                    "20": _budget_metric(0),
                    "40": _budget_metric(1),
                    "80": _budget_metric(1),
            },
            "first_strict_hit_budget": 40,
        },
        {
            "sample_id": "c",
            "method_id": "dense",
                "budgets": {
                    "20": _budget_metric(0),
                    "40": _budget_metric(0),
                    "80": _budget_metric(0),
            },
            "first_strict_hit_budget": None,
        },
    ]

    summary = module.summarize_budget_curves(
        rows=rows,
        methods=["dense"],
        budgets=[20, 40, 80],
    )[0]

    assert summary["strict_hits_at_budget"] == {"20": 1, "40": 2, "80": 2}
    assert summary["new_strict_hits_by_budget"] == {"20": 1, "40": 1, "80": 0}
    assert summary["strict_miss_after_max_budget_count"] == 1


def test_join_fails_closed_when_gold_and_retrieval_ids_differ():
    module = _load_module()

    with pytest.raises(ValueError, match="sample IDs do not match"):
        module.join_gold_and_retrieval(
            [{"candidate_id": "s1"}, {"candidate_id": "s2"}],
            [{"sample_id": "s1"}],
        )
