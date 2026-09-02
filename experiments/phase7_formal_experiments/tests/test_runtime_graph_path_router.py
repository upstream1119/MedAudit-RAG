from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "runtime_graph_path_router.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "Runtime graph path router is missing"
    spec = importlib.util.spec_from_file_location(
        "runtime_graph_path_router",
        MODULE_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_source_condition_match_outranks_generic_source():
    module = _load_module()
    graph_index = {
        "graph_index_version": "fixture-v0.1",
        "candidates": {
            "detail::generic": {
                "candidate_key": "detail::generic",
                "content": "MPP在限定情况下可考虑糖皮质激素。",
                "source_file": "儿童常见感染综述.pdf",
                "page_number": 8,
            },
            "detail::specific": {
                "candidate_key": "detail::specific",
                "content": "MPP在限定情况下可考虑糖皮质激素。",
                "source_file": "MPP诊疗指南.pdf",
                "page_number": 10,
            },
        },
    }

    catalog = module.build_runtime_path_catalog(graph_index, _lexicon())
    routed = module.route_graph_paths(
        "MPP糖皮质激素治疗是否有依据？",
        catalog=catalog,
        lexicon=_lexicon(),
        allow_specific_condition_class_path=True,
        max_total_paths=20,
        max_paths_per_source=2,
        max_paths_per_source_page=1,
    )

    assert [row["candidate_key"] for row in routed["selected_paths"]] == [
        "detail::specific",
        "detail::generic",
    ]
    assert routed["selected_paths"][0]["graph_source_condition_tier"] == 0
    assert routed["selected_paths"][1]["graph_source_condition_tier"] == 1


def test_routing_enforces_page_source_and_total_quotas():
    module = _load_module()
    candidates = {}
    for key, source, page in [
        ("a-p1-first", "MPP_A指南.pdf", 1),
        ("a-p1-second", "MPP_A指南.pdf", 1),
        ("a-p2", "MPP_A指南.pdf", 2),
        ("b-p1-first", "MPP_B指南.pdf", 1),
        ("b-p1-second", "MPP_B指南.pdf", 1),
        ("c-p1", "MPP_C指南.pdf", 1),
    ]:
        candidates[key] = {
            "candidate_key": key,
            "content": "MPP在限定情况下可考虑糖皮质激素。",
            "source_file": source,
            "page_number": page,
        }
    catalog = module.build_runtime_path_catalog(
        {
            "graph_index_version": "fixture-v0.1",
            "candidates": candidates,
        },
        _lexicon(),
    )

    routed = module.route_graph_paths(
        "MPP糖皮质激素治疗是否有依据？",
        catalog=catalog,
        lexicon=_lexicon(),
        allow_specific_condition_class_path=True,
        max_total_paths=3,
        max_paths_per_source=2,
        max_paths_per_source_page=1,
    )

    assert [row["candidate_key"] for row in routed["selected_paths"]] == [
        "a-p1-first",
        "a-p2",
        "b-p1-first",
    ]
    assert routed["drop_reason_counts"] == {
        "source_page_quota": 2,
        "total_quota": 1,
    }
    audit_by_key = {
        row["candidate_key"]: row for row in routed["path_audit"]
    }
    assert audit_by_key["a-p1-second"]["graph_route_drop_reason"] == (
        "source_page_quota"
    )
    assert audit_by_key["c-p1"]["graph_route_drop_reason"] == "total_quota"


def test_routing_enforces_per_source_quota_across_pages():
    module = _load_module()
    graph_index = {
        "graph_index_version": "fixture-v0.1",
        "candidates": {
            f"a-p{page}": {
                "candidate_key": f"a-p{page}",
                "content": "MPP在限定情况下可考虑糖皮质激素。",
                "source_file": "MPP_A指南.pdf",
                "page_number": page,
            }
            for page in (1, 2, 3)
        },
    }
    graph_index["candidates"]["b-p1"] = {
        "candidate_key": "b-p1",
        "content": "MPP在限定情况下可考虑糖皮质激素。",
        "source_file": "MPP_B指南.pdf",
        "page_number": 1,
    }
    catalog = module.build_runtime_path_catalog(graph_index, _lexicon())

    routed = module.route_graph_paths(
        "MPP糖皮质激素治疗是否有依据？",
        catalog=catalog,
        lexicon=_lexicon(),
        allow_specific_condition_class_path=True,
        max_total_paths=20,
        max_paths_per_source=2,
        max_paths_per_source_page=1,
    )

    assert [row["candidate_key"] for row in routed["selected_paths"]] == [
        "a-p1",
        "a-p2",
        "b-p1",
    ]
    assert routed["drop_reason_counts"] == {"source_quota": 1}


def test_routing_is_deterministic_and_rejects_gold_only_catalog_inputs():
    module = _load_module()
    graph_index = {
        "graph_index_version": "fixture-v0.1",
        "candidates": {
            "detail::1": {
                "candidate_key": "detail::1",
                "content": "MPP在限定情况下可考虑糖皮质激素。",
                "source_file": "MPP指南.pdf",
                "page_number": 10,
            }
        },
    }
    catalog = module.build_runtime_path_catalog(graph_index, _lexicon())
    kwargs = {
        "catalog": catalog,
        "lexicon": _lexicon(),
        "allow_specific_condition_class_path": True,
        "max_total_paths": 20,
        "max_paths_per_source": 2,
        "max_paths_per_source_page": 1,
    }

    assert module.route_graph_paths("MPP糖皮质激素是否适用？", **kwargs) == (
        module.route_graph_paths("MPP糖皮质激素是否适用？", **kwargs)
    )
    unsafe_graph = {
        **graph_index,
        "gold_evidence": "must not be read",
    }
    with pytest.raises(ValueError, match="gold-only"):
        module.build_runtime_path_catalog(unsafe_graph, _lexicon())


def test_broad_antimicrobial_condition_class_path_remains_blocked():
    module = _load_module()
    lexicon = _lexicon()
    lexicon["entries"].append(
        {
            "constraint_type": "medication_class",
            "normalized_value": "antimicrobial",
            "aliases": ["抗菌药物"],
            "strong_anchor": True,
        }
    )
    graph_index = {
        "graph_index_version": "fixture-v0.1",
        "candidates": {
            "detail::broad": {
                "candidate_key": "detail::broad",
                "content": "MPP治疗可涉及抗菌药物。",
                "source_file": "MPP指南.pdf",
                "page_number": 10,
            }
        },
    }
    catalog = module.build_runtime_path_catalog(graph_index, lexicon)

    routed = module.route_graph_paths(
        "MPP抗菌药物如何选择？",
        catalog=catalog,
        lexicon=lexicon,
        allow_specific_condition_class_path=True,
        max_total_paths=20,
        max_paths_per_source=2,
        max_paths_per_source_page=1,
    )

    assert routed["raw_path_count"] == 0
    assert routed["selected_paths"] == []
