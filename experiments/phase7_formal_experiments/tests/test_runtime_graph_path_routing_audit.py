from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "runtime_graph_path_routing_audit.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "Runtime graph path routing audit is missing"
    spec = importlib.util.spec_from_file_location(
        "runtime_graph_path_routing_audit",
        MODULE_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _fixture_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    runtime_path = tmp_path / "runtime.jsonl"
    graph_path = tmp_path / "graph.json"
    lexicon_path = tmp_path / "lexicon.json"
    _write_jsonl(
        runtime_path,
        [
            {
                "sample_id": "v-001",
                "question": "MPP糖皮质激素治疗是否有依据？",
            },
            {
                "sample_id": "v-002",
                "question": "未收录疾病的治疗原则是什么？",
            },
        ],
    )
    _write_json(
        graph_path,
        {
            "graph_index_version": "fixture-v0.1",
            "candidates": {
                "detail::specific": {
                    "candidate_key": "detail::specific",
                    "content": "MPP在限定情况下可考虑糖皮质激素。",
                    "source_file": "MPP诊疗指南.pdf",
                    "page_number": 10,
                },
                "detail::generic": {
                    "candidate_key": "detail::generic",
                    "content": "MPP在限定情况下可考虑糖皮质激素。",
                    "source_file": "儿童感染综述.pdf",
                    "page_number": 8,
                },
            },
        },
    )
    _write_json(
        lexicon_path,
        {
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
        },
    )
    return runtime_path, graph_path, lexicon_path


def test_routing_audit_reports_gold_free_coverage_and_budget(tmp_path):
    module = _load_module()
    runtime_path, graph_path, lexicon_path = _fixture_files(tmp_path)

    audit = module.build_routing_audit(
        runtime_projection_path=runtime_path,
        graph_index_path=graph_path,
        lexicon_path=lexicon_path,
        allow_specific_condition_class_path=True,
        max_total_paths=20,
        max_paths_per_source=2,
        max_paths_per_source_page=1,
        broad_path_threshold=200,
    )

    assert audit["question_count"] == 2
    assert audit["candidate_count"] == 2
    assert audit["raw_path_coverage_count"] == 1
    assert audit["routed_path_coverage_count"] == 1
    assert audit["raw_zero_path_count"] == 1
    assert audit["routed_zero_path_count"] == 1
    assert audit["raw_path_distribution"]["maximum"] == 2
    assert audit["routed_path_distribution"]["maximum"] == 2
    assert audit["routed_over_total_budget_count"] == 0
    assert audit["source_quota_violation_count"] == 0
    assert audit["source_page_quota_violation_count"] == 0
    assert audit["external_model_calls"] == 0
    assert audit["input_tokens"] == 0
    assert audit["output_tokens"] == 0
    assert audit["estimated_cost"] == 0.0
    assert audit["guards"] == {
        "gold_access": False,
        "pilot_test_content_access": False,
        "external_model_calls": False,
    }


def test_routing_audit_is_deterministic(tmp_path):
    module = _load_module()
    runtime_path, graph_path, lexicon_path = _fixture_files(tmp_path)
    kwargs = {
        "runtime_projection_path": runtime_path,
        "graph_index_path": graph_path,
        "lexicon_path": lexicon_path,
        "allow_specific_condition_class_path": True,
        "max_total_paths": 20,
        "max_paths_per_source": 2,
        "max_paths_per_source_page": 1,
        "broad_path_threshold": 200,
    }

    first = module.build_routing_audit(**kwargs)
    second = module.build_routing_audit(**kwargs)

    assert first == second
    assert first["deterministic_payload_sha256"] == second[
        "deterministic_payload_sha256"
    ]


def test_routing_audit_rejects_gold_only_runtime_fields(tmp_path):
    module = _load_module()
    runtime_path, graph_path, lexicon_path = _fixture_files(tmp_path)
    _write_jsonl(
        runtime_path,
        [
            {
                "sample_id": "unsafe",
                "question": "MPP糖皮质激素治疗是否有依据？",
                "gold_evidence": "must not be read",
            }
        ],
    )

    with pytest.raises(ValueError, match="gold-only"):
        module.build_routing_audit(
            runtime_projection_path=runtime_path,
            graph_index_path=graph_path,
            lexicon_path=lexicon_path,
            allow_specific_condition_class_path=True,
            max_total_paths=20,
            max_paths_per_source=2,
            max_paths_per_source_page=1,
            broad_path_threshold=200,
        )
