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
    / "runtime_constraint_coverage_audit.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "Runtime constraint coverage audit is missing"
    spec = importlib.util.spec_from_file_location(
        "runtime_constraint_coverage_audit",
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
                "question": "MPP神经系统并发症中的阿奇霉素疗程如何理解？",
            },
            {
                "sample_id": "v-002",
                "question": "毛细支气管炎是否需要再次评估？",
            },
        ],
    )
    _write_json(
        graph_path,
        {
            "graph_index_version": "fixture-v0.1",
            "candidates": {
                "detail::1": {
                    "candidate_key": "detail::1",
                    "content": "MPP神经系统并发症可涉及阿奇霉素疗程。",
                    "source_file": "MPP指南.pdf",
                    "page_number": 10,
                },
                "detail::2": {
                    "candidate_key": "detail::2",
                    "content": "毛细支气管炎患儿应结合病情评估。",
                    "source_file": "毛细支气管炎指南.pdf",
                    "page_number": 3,
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
                    "constraint_type": "clinical_condition",
                    "normalized_value": "bronchiolitis",
                    "aliases": ["毛细支气管炎"],
                    "strong_anchor": True,
                },
                {
                    "constraint_type": "medication",
                    "normalized_value": "azithromycin",
                    "aliases": ["阿奇霉素"],
                    "strong_anchor": True,
                },
                {
                    "constraint_type": "evidence_topic",
                    "normalized_value": "neurologic_complication",
                    "aliases": ["神经系统并发症"],
                    "strong_anchor": False,
                },
            ],
        },
    )
    return runtime_path, graph_path, lexicon_path


def test_coverage_audit_reports_gold_free_query_and_candidate_paths(tmp_path):
    module = _load_module()
    runtime_path, graph_path, lexicon_path = _fixture_files(tmp_path)

    audit = module.build_coverage_audit(
        runtime_projection_path=runtime_path,
        graph_index_path=graph_path,
        lexicon_path=lexicon_path,
        minimum_matched_constraint_types=2,
    )

    assert audit["question_count"] == 2
    assert audit["candidate_count"] == 2
    assert audit["zero_query_constraint_count"] == 0
    assert audit["query_two_type_coverage_count"] == 2
    assert audit["candidate_nonempty_constraint_count"] == 2
    assert audit["question_with_potential_path_count"] == 1
    assert audit["zero_potential_path_count"] == 1
    assert audit["zero_path_diagnosis_counts"] == {
        "insufficient_posting_types": 1
    }
    assert audit["cross_condition_conflict_count"] == 2
    assert audit["potential_path_distribution"]["maximum"] == 1
    assert audit["potential_path_distribution"]["median"] == 0.5
    assert audit["potential_path_distribution"]["p95"] == 1
    assert audit["broad_path_question_count"] == 0
    assert audit["path_failure_reason_counts"]["condition_conflict"] == 2
    assert audit["question_details"][0]["candidate_source_count"] == 1
    assert audit["question_details"][0]["matched_signature_counts"] == {
        "clinical_condition+evidence_topic+medication": 1
    }
    zero_detail = audit["question_details"][1]
    assert zero_detail["zero_path_reason"] == "no_posting"
    assert zero_detail["zero_path_diagnosis"] == "insufficient_posting_types"
    assert zero_detail["maximum_candidate_matched_type_count"] == 1
    assert zero_detail["constraint_posting_counts"] == {
        "clinical_condition::bronchiolitis": {
            "content": 1,
            "context": 1,
        },
        "monitoring_action::reassess": {
            "content": 0,
            "context": 0,
        },
    }
    assert zero_detail["top_failure_reasons"]
    assert audit["external_model_calls"] == 0
    assert audit["input_tokens"] == 0
    assert audit["output_tokens"] == 0
    assert "path_policy" not in audit


def test_coverage_audit_is_deterministic(tmp_path):
    module = _load_module()
    runtime_path, graph_path, lexicon_path = _fixture_files(tmp_path)

    first = module.build_coverage_audit(
        runtime_projection_path=runtime_path,
        graph_index_path=graph_path,
        lexicon_path=lexicon_path,
        minimum_matched_constraint_types=2,
    )
    second = module.build_coverage_audit(
        runtime_projection_path=runtime_path,
        graph_index_path=graph_path,
        lexicon_path=lexicon_path,
        minimum_matched_constraint_types=2,
    )

    assert first == second
    assert first["deterministic_payload_sha256"] == second[
        "deterministic_payload_sha256"
    ]


def test_coverage_audit_records_specific_condition_class_policy(tmp_path):
    module = _load_module()
    runtime_path, graph_path, lexicon_path = _fixture_files(tmp_path)
    _write_jsonl(
        runtime_path,
        [
            {
                "sample_id": "v-policy",
                "question": "MPP糖皮质激素治疗是否有依据？",
            }
        ],
    )
    _write_json(
        graph_path,
        {
            "graph_index_version": "fixture-v0.1",
            "candidates": {
                "detail::policy": {
                    "candidate_key": "detail::policy",
                    "content": "MPP可在限定情况下考虑糖皮质激素。",
                    "source_file": "MPP指南.pdf",
                    "page_number": 10,
                }
            },
        },
    )
    lexicon = json.loads(lexicon_path.read_text(encoding="utf-8"))
    lexicon["entries"].append(
        {
            "constraint_type": "medication_class",
            "normalized_value": "corticosteroid",
            "aliases": ["糖皮质激素"],
            "strong_anchor": True,
        }
    )
    _write_json(lexicon_path, lexicon)

    audit = module.build_coverage_audit(
        runtime_projection_path=runtime_path,
        graph_index_path=graph_path,
        lexicon_path=lexicon_path,
        minimum_matched_constraint_types=2,
        allow_specific_condition_class_path=True,
        audit_version="phase7-runtime-constraint-coverage-v0.3",
    )

    assert audit["audit_version"] == "phase7-runtime-constraint-coverage-v0.3"
    assert audit["question_with_potential_path_count"] == 1
    assert audit["path_policy"] == {
        "allow_specific_condition_class_path": True,
        "broad_condition_class_remains_blocked": True,
    }


def test_coverage_audit_rejects_gold_only_runtime_fields(tmp_path):
    module = _load_module()
    runtime_path, graph_path, lexicon_path = _fixture_files(tmp_path)
    _write_jsonl(
        runtime_path,
        [
            {
                "sample_id": "unsafe",
                "question": "MPP如何用药？",
                "gold_evidence": "must not be read",
            }
        ],
    )

    with pytest.raises(ValueError, match="gold-only"):
        module.build_coverage_audit(
            runtime_projection_path=runtime_path,
            graph_index_path=graph_path,
            lexicon_path=lexicon_path,
            minimum_matched_constraint_types=2,
        )
