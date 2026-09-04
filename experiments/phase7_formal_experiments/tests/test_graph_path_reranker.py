from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "graph_path_reranker.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "Graph path reranker module is missing"
    spec = importlib.util.spec_from_file_location("graph_path_reranker", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(key: str, *, origin: str, rank: int, source: str) -> dict:
    collection, document_id = key.split("::", maxsplit=1)
    return {
        "candidate_key": key,
        "collection": collection,
        "document_id": document_id,
        "content": "MPP 在限定情况下可考虑糖皮质激素。",
        "source_file": source,
        "page_number": rank,
        "candidate_origin": origin,
        "post_rerank_rank": rank,
        "reranker_score": float(10 - rank),
    }


def _lexicon() -> dict:
    return {
        "lexicon_version": "fixture-v0.2",
        "entries": [
            {
                "constraint_type": "clinical_condition",
                "normalized_value": "mycoplasma_pneumoniae_pneumonia",
                "aliases": ["MPP"],
                "strong_anchor": True,
            },
            {
                "constraint_type": "medication_class",
                "normalized_value": "corticosteroid",
                "aliases": ["糖皮质激素"],
                "strong_anchor": True,
            },
        ],
    }


def _tier_weights() -> dict[str, float]:
    return {
        "source_condition_exact": 1.0,
        "content_condition_generic_source": 0.75,
        "query_without_condition": 0.5,
        "content_condition_other_source": 0.25,
        "context_only_condition": 0.0,
    }


def test_baseline_and_graph_candidates_receive_the_same_path_treatment():
    module = _load_module()
    candidates = [
        _candidate(
            "detail_128::graph",
            origin="graph_expansion",
            rank=1,
            source="儿科感染指南.pdf",
        ),
        _candidate(
            "detail_128::baseline",
            origin="baseline",
            rank=2,
            source="MPP诊疗指南.pdf",
        ),
    ]
    graph_index = {
        "candidates": {
            row["candidate_key"]: {
                key: value
                for key, value in row.items()
                if key not in {"candidate_origin", "post_rerank_rank", "reranker_score"}
            }
            for row in candidates
        }
    }
    selected_paths = [
        {
            "candidate_key": "detail_128::baseline",
            "source_file": "MPP诊疗指南.pdf",
            "page_number": 2,
            "graph_route_rank": 1,
            "graph_source_condition_tier": 0,
            "graph_source_condition_tier_label": "source_condition_exact",
            "graph_path_constraint_types": [
                "clinical_condition",
                "medication_class",
            ],
        },
        {
            "candidate_key": "detail_128::graph",
            "source_file": "儿科感染指南.pdf",
            "page_number": 1,
            "graph_route_rank": 2,
            "graph_source_condition_tier": 1,
            "graph_source_condition_tier_label": "content_condition_generic_source",
            "graph_path_constraint_types": [
                "clinical_condition",
                "medication_class",
            ],
        },
    ]
    tier_weights = _tier_weights()

    ranked, audit = module.graph_rerank_candidates(
        question="MPP 糖皮质激素治疗是否有依据？",
        candidates=candidates,
        selected_paths=selected_paths,
        graph_index=graph_index,
        lexicon=_lexicon(),
        tier_weights=tier_weights,
        max_rank_shift=2.0,
    )

    by_key = {row["candidate_key"]: row for row in ranked}
    assert set(by_key) == {"detail_128::graph", "detail_128::baseline"}
    assert by_key["detail_128::baseline"]["g2_path_eligible"] is True
    assert by_key["detail_128::graph"]["g2_path_eligible"] is True
    assert by_key["detail_128::baseline"]["g2_graph_path_trace"]
    assert by_key["detail_128::graph"]["g2_graph_path_trace"]
    assert audit["path_eligible_count"] == 2

    swapped_origins = deepcopy(candidates)
    swapped_origins[0]["candidate_origin"] = "baseline"
    swapped_origins[1]["candidate_origin"] = "graph_expansion"
    reranked, _ = module.graph_rerank_candidates(
        question="MPP 糖皮质激素治疗是否有依据？",
        candidates=swapped_origins,
        selected_paths=selected_paths,
        graph_index=graph_index,
        lexicon=_lexicon(),
        tier_weights=tier_weights,
        max_rank_shift=2.0,
    )
    assert [row["candidate_key"] for row in reranked] == [
        row["candidate_key"] for row in ranked
    ]
    assert [row["graph_path_score"] for row in reranked] == [
        row["graph_path_score"] for row in ranked
    ]


def test_zero_rank_shift_preserves_frozen_order_without_mutating_input():
    module = _load_module()
    candidates = [
        _candidate(
            "detail_128::unmatched",
            origin="baseline",
            rank=1,
            source="其他指南.pdf",
        ),
        _candidate(
            "detail_128::matched",
            origin="baseline",
            rank=2,
            source="MPP诊疗指南.pdf",
        ),
    ]
    frozen_input = deepcopy(candidates)
    graph_index = {
        "candidates": {
            "detail_128::matched": {
                key: value
                for key, value in candidates[1].items()
                if key not in {"candidate_origin", "post_rerank_rank", "reranker_score"}
            }
        }
    }
    selected_paths = [
        {
            "candidate_key": "detail_128::matched",
            "source_file": "MPP诊疗指南.pdf",
            "page_number": 2,
            "graph_route_rank": 1,
            "graph_source_condition_tier": 0,
            "graph_source_condition_tier_label": "source_condition_exact",
            "graph_path_constraint_types": [
                "clinical_condition",
                "medication_class",
            ],
        }
    ]

    ranked, audit = module.graph_rerank_candidates(
        question="MPP 糖皮质激素治疗是否有依据？",
        candidates=candidates,
        selected_paths=selected_paths,
        graph_index=graph_index,
        lexicon=_lexicon(),
        tier_weights=_tier_weights(),
        max_rank_shift=0.0,
    )

    assert [row["candidate_key"] for row in ranked] == [
        row["candidate_key"] for row in candidates
    ]
    assert candidates == frozen_input
    assert audit["candidate_set_preserved"] is True


def test_existing_graph_trace_mismatch_fails_closed():
    module = _load_module()
    candidate = _candidate(
        "detail_128::matched",
        origin="graph_expansion",
        rank=1,
        source="MPP诊疗指南.pdf",
    )
    candidate["graph_path_trace"] = {"route_rank": 99}
    graph_index = {
        "candidates": {
            "detail_128::matched": {
                key: value
                for key, value in candidate.items()
                if key
                not in {
                    "candidate_origin",
                    "post_rerank_rank",
                    "reranker_score",
                    "graph_path_trace",
                }
            }
        }
    }
    selected_paths = [
        {
            "candidate_key": "detail_128::matched",
            "source_file": "MPP诊疗指南.pdf",
            "page_number": 1,
            "graph_route_rank": 1,
            "graph_source_condition_tier": 0,
            "graph_source_condition_tier_label": "source_condition_exact",
            "graph_path_constraint_types": [
                "clinical_condition",
                "medication_class",
            ],
        }
    ]

    with pytest.raises(
        ValueError, match="existing graph path trace does not match symmetric rebuild"
    ):
        module.graph_rerank_candidates(
            question="MPP 糖皮质激素治疗是否有依据？",
            candidates=[candidate],
            selected_paths=selected_paths,
            graph_index=graph_index,
            lexicon=_lexicon(),
            tier_weights=_tier_weights(),
            max_rank_shift=2.0,
        )


def test_gold_only_candidate_fields_fail_closed():
    module = _load_module()
    candidate = _candidate(
        "detail_128::gold",
        origin="baseline",
        rank=1,
        source="MPP诊疗指南.pdf",
    )
    candidate["gold_evidence"] = {"source_file": "MPP诊疗指南.pdf"}

    with pytest.raises((AssertionError, ValueError), match="[Gg]old"):
        module.graph_rerank_candidates(
            question="MPP 糖皮质激素治疗是否有依据？",
            candidates=[candidate],
            selected_paths=[],
            graph_index={"candidates": {}},
            lexicon=_lexicon(),
            tier_weights=_tier_weights(),
            max_rank_shift=2.0,
        )


def test_graph_score_and_rank_adjustment_stay_within_configured_bounds():
    module = _load_module()
    candidates = [
        _candidate(
            "detail_128::first",
            origin="baseline",
            rank=1,
            source="其他指南.pdf",
        ),
        _candidate(
            "detail_128::second",
            origin="baseline",
            rank=2,
            source="其他指南.pdf",
        ),
        _candidate(
            "detail_128::matched",
            origin="graph_expansion",
            rank=3,
            source="MPP诊疗指南.pdf",
        ),
    ]
    graph_index = {
        "candidates": {
            "detail_128::matched": {
                key: value
                for key, value in candidates[2].items()
                if key not in {"candidate_origin", "post_rerank_rank", "reranker_score"}
            }
        }
    }
    selected_paths = [
        {
            "candidate_key": "detail_128::matched",
            "source_file": "MPP诊疗指南.pdf",
            "page_number": 3,
            "graph_route_rank": 1,
            "graph_source_condition_tier": 0,
            "graph_source_condition_tier_label": "source_condition_exact",
            "graph_path_constraint_types": [
                "clinical_condition",
                "medication_class",
            ],
        }
    ]

    ranked, audit = module.graph_rerank_candidates(
        question="MPP 糖皮质激素治疗是否有依据？",
        candidates=candidates,
        selected_paths=selected_paths,
        graph_index=graph_index,
        lexicon=_lexicon(),
        tier_weights=_tier_weights(),
        max_rank_shift=2.0,
    )

    by_key = {row["candidate_key"]: row for row in ranked}
    matched = by_key["detail_128::matched"]
    assert 0.0 <= matched["graph_path_score"] <= 1.0
    assert all(
        0.0 <= value <= 1.0
        for value in matched["graph_path_score_components"].values()
    )
    assert matched["reranker_rank_before_graph"] - matched["graph_adjusted_rank_value"] <= 2.0
    assert matched["reranker_rank_before_graph"] - matched["graph_rerank_rank"] <= 2
    assert {row["candidate_key"] for row in ranked} == {
        row["candidate_key"] for row in candidates
    }
    assert audit["candidate_origin_used_for_score"] is False
