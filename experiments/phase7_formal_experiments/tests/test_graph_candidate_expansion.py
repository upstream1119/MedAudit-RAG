from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "graph_candidate_expansion.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "Graph candidate expansion module is missing"
    spec = importlib.util.spec_from_file_location(
        "graph_candidate_expansion",
        MODULE_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(
    candidate_key: str,
    content: str,
    *,
    source_file: str = "儿童社区获得性肺炎诊疗规范.pdf",
    page_number: int = 1,
) -> dict:
    collection, document_id = candidate_key.split("::", maxsplit=1)
    return {
        "candidate_key": candidate_key,
        "collection": collection,
        "document_id": document_id,
        "content": content,
        "source_file": source_file,
        "page_number": page_number,
        "granularity": 128,
    }


def _source_routing_lexicon() -> dict:
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


def test_source_routed_expansion_prefers_condition_specific_source():
    module = _load_module()
    router = __import__(
        "experiments.phase7_formal_experiments.runtime_graph_path_router",
        fromlist=["build_runtime_path_catalog"],
    )
    baseline = [
        _candidate("detail_128::base-1", "基线证据1。", page_number=1),
        _candidate("detail_128::base-2", "基线证据2。", page_number=2),
    ]
    graph_rows = [
        _candidate(
            "detail_128::generic",
            "MPP在限定情况下可考虑糖皮质激素。",
            source_file="儿童常见感染综述.pdf",
            page_number=8,
        ),
        _candidate(
            "detail_128::specific",
            "MPP在限定情况下可考虑糖皮质激素。",
            source_file="MPP诊疗指南.pdf",
            page_number=10,
        ),
    ]
    graph_index = module.build_candidate_graph_index([*baseline, *graph_rows])
    lexicon = _source_routing_lexicon()
    catalog = router.build_runtime_path_catalog(graph_index, lexicon)
    baseline[0]["graph_path_trace"] = {"stale": True}

    expanded = module.expand_candidates(
        "MPP糖皮质激素治疗是否有依据？",
        baseline,
        graph_index,
        total_budget=2,
        graph_quota=1,
        runtime_path_catalog=catalog,
        runtime_lexicon=lexicon,
        routing_policy={
            "allow_specific_condition_class_path": True,
            "max_total_paths": 20,
            "max_paths_per_source": 2,
            "max_paths_per_source_page": 1,
        },
    )

    assert [row["candidate_key"] for row in expanded] == [
        "detail_128::base-1",
        "detail_128::specific",
    ]
    assert expanded[1]["candidate_origin"] == "graph_expansion"
    assert expanded[1]["graph_source_condition_tier_label"] == (
        "source_condition_exact"
    )
    assert "graph_path_trace" not in expanded[0]
    assert expanded[1]["graph_path_trace"]["candidate"] == {
        "candidate_key": "detail_128::specific",
        "source_file": "MPP诊疗指南.pdf",
        "page_number": 10,
    }
    assert expanded[1]["graph_path_trace"]["route_decision"] == "selected"


def test_constraint_path_recovers_candidate_missing_from_baseline():
    module = _load_module()
    baseline = [
        _candidate("detail_128::base-1", "儿童肺炎的一般定义。", page_number=3),
        _candidate("detail_128::base-2", "儿童发热的一般评估。", page_number=4),
    ]
    graph_only = _candidate(
        "detail_128::graph-1",
        "治疗48-72小时症状无改善时，应再次进行临床或实验室评估。",
        page_number=26,
    )
    graph_index = module.build_candidate_graph_index([*baseline, graph_only])

    expanded = module.expand_candidates(
        "儿童肺炎治疗48-72小时症状无改善时，是否需要再次评估？",
        baseline,
        graph_index,
        total_budget=2,
        graph_quota=1,
    )

    assert [row["candidate_key"] for row in expanded] == [
        "detail_128::base-1",
        "detail_128::graph-1",
    ]
    assert expanded[1]["candidate_origin"] == "graph_expansion"
    assert expanded[1]["graph_path_match_count"] == 3
    assert expanded[1]["graph_path_constraint_types"] == [
        "monitoring_action",
        "monitoring_trigger",
        "monitoring_window",
    ]


@pytest.mark.parametrize(
    "gold_field",
    [
        "gold_evidence",
        "expected_decision",
        "allowed_claims",
        "forbidden_claims",
    ],
)
def test_gold_only_fields_are_rejected_fail_closed(gold_field):
    module = _load_module()
    row = _candidate("detail_128::unsafe", "治疗后应再次评估。")
    row[gold_field] = "forbidden evaluation annotation"

    with pytest.raises(ValueError, match="gold-only"):
        module.build_candidate_graph_index([row])


def test_no_reliable_graph_match_preserves_baseline_order():
    module = _load_module()
    baseline = [
        _candidate("detail_128::base-1", "儿童肺炎的一般定义。"),
        _candidate("detail_128::base-2", "儿童发热的一般评估。"),
    ]
    graph_index = module.build_candidate_graph_index(
        [
            *baseline,
            _candidate(
                "detail_128::unrelated",
                "治疗48-72小时症状无改善时，应再次评估。",
            ),
        ]
    )

    expanded = module.expand_candidates(
        "儿童肺炎的一般治疗原则有哪些？",
        baseline,
        graph_index,
        total_budget=2,
        graph_quota=1,
    )

    assert [row["candidate_key"] for row in expanded] == [
        "detail_128::base-1",
        "detail_128::base-2",
    ]
    assert all(row["candidate_origin"] == "baseline" for row in expanded)


def test_budget_deduplication_and_order_are_deterministic():
    module = _load_module()
    baseline = [
        _candidate("detail_128::base-1", "儿童肺炎的一般定义。"),
        _candidate("detail_128::base-2", "儿童发热的一般评估。"),
    ]
    graph_rows = [
        _candidate(
            "detail_128::graph-b",
            "治疗48-72小时症状无改善时，应再次评估。",
            page_number=27,
        ),
        _candidate(
            "detail_128::graph-a",
            "治疗48-72小时症状无改善时，应重新评估。",
            page_number=26,
        ),
    ]
    question = "治疗48-72小时症状无改善时是否需要再次评估？"

    first = module.expand_candidates(
        question,
        baseline,
        module.build_candidate_graph_index([*baseline, *graph_rows]),
        total_budget=3,
        graph_quota=2,
    )
    second = module.expand_candidates(
        question,
        list(reversed(baseline)),
        module.build_candidate_graph_index(list(reversed([*baseline, *graph_rows]))),
        total_budget=3,
        graph_quota=2,
    )

    assert [row["candidate_key"] for row in first] == [
        "detail_128::base-1",
        "detail_128::graph-a",
        "detail_128::graph-b",
    ]
    assert [row["candidate_key"] for row in second[1:]] == [
        "detail_128::graph-a",
        "detail_128::graph-b",
    ]
    assert len({row["candidate_key"] for row in first}) == len(first) == 3


def test_invalid_candidate_budget_is_rejected():
    module = _load_module()
    graph_index = module.build_candidate_graph_index([])

    with pytest.raises(ValueError, match="graph_quota"):
        module.expand_candidates(
            "治疗后是否需要评估？",
            [],
            graph_index,
            total_budget=2,
            graph_quota=3,
        )
