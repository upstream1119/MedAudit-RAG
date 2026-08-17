from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "benchmark120_gold_freeze.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "Benchmark120 Gold freeze module is missing"
    spec = importlib.util.spec_from_file_location("benchmark120_gold_freeze", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selection(index: int, *, second_pass: bool) -> dict:
    split = "validation" if index <= 2 else "pilot_test"
    decision = "review_required" if second_pass else "answer"
    return {
        "candidate_id": f"PMSQA-GOLD-{index:03d}",
        "question": f"第 {index} 条指南约束审计问题",
        "dataset_split": split,
        "dataset_version": "benchmark-v1.0-selection-draft-v0.1",
        "kb_version": "KB-medium-v1",
        "selection_version": "benchmark120-selection-draft-v0.1",
        "source_id": f"SRC-{index:03d}",
        "source_title": f"测试指南 {index}",
        "source_filename": f"source-{index}.pdf",
        "source_sha256": f"source-sha-{index}",
        "page_number": index,
        "anchor_text_span": f"第 {index} 条可定位证据。",
        "fact_cluster_id": f"FC-{index:03d}",
        "evidence_anchor_group_id": f"EAG-{index:03d}",
        "evidence_anchor_ids": [f"ANCH-{index:03d}"],
        "independence_unit_id": f"IU-{index:03d}",
        "expected_decision": decision,
        "current_kb_support": "partial_current_kb_support" if second_pass else "supported_by_current_kb",
        "gold_evidence_status": "page_span_located",
        "required_evidence_type": [],
        "required_claims": [],
        "allowed_claims": ["只允许证据范围内结论"],
        "forbidden_claims": ["不得外推"],
        "missing_evidence_type": [],
        "missing_information": [],
        "risk_labels": ["review_required"] if second_pass else ["evidence_supported"],
        "requires_second_pass": second_pass,
        "freeze_status": "draft",
        "split_status": "proposal",
    }


def _adjudication(selection: dict, *, boundary: bool = False) -> dict[str, str]:
    decision = "boundary_refusal" if boundary else "answer"
    support = "policy_rule" if boundary else "supported_by_current_kb"
    evidence = "policy_rule" if boundary else "page_span_located"
    return {
        "candidate_id": selection["candidate_id"],
        "question": selection["question"],
        "source_id": selection["source_id"],
        "source_title": selection["source_title"],
        "source_filename": selection["source_filename"],
        "source_sha256": selection["source_sha256"],
        "page_number": str(selection["page_number"]),
        "anchor_text_span": selection["anchor_text_span"],
        "dataset_split": selection["dataset_split"],
        "resolution_reviewer_id": "PYH",
        "resolution_annotator_role": "author",
        "resolution_reviewed_at": "2026-08-16T20:00:00+08:00",
        "resolution_status": "accepted",
        "resolution_final_decision": decision,
        "resolution_final_kb_support": support,
        "resolution_final_gold_evidence_status": evidence,
        "resolution_final_required_evidence_type": json.dumps(
            ["safety_policy"] if boundary else ["direct_page_span"]
        ),
        "resolution_final_required_claims": json.dumps(["受限审计结论"]),
        "resolution_final_allowed_claims": json.dumps(["可指出证据边界"]),
        "resolution_final_forbidden_claims": json.dumps(["不得生成处方"]),
        "resolution_final_missing_evidence_type": "[]",
        "resolution_final_missing_information": "[]",
        "resolution_final_risk_labels": json.dumps(
            ["prescription_boundary"] if boundary else ["claim_overreach"]
        ),
        "resolution_reason": "逐条核对问题、来源、页码和证据跨度后确认。",
    }


def _config() -> dict:
    return {
        "config_version": "benchmark120-gold-freeze-config-v1.0",
        "gold_version": "benchmark-v1.0-guideline-grounded-author-adjudicated",
        "freeze_version": "benchmark-v1.0-freeze-v1.0",
        "kb_version": "KB-medium-v1",
        "protocol_version": "benchmark-protocol-v0.1",
        "expected_selection_count": 4,
        "expected_adjudication_count": 2,
        "expected_split_counts": {"validation": 2, "pilot_test": 2},
        "expected_input_sha256": {
            "selection": "selection-hash",
            "split": "split-hash",
            "adjudication": "adjudication-hash",
            "adjudication_audit": "audit-hash",
        },
        "external_model_calls": 0,
    }


def _assets():
    selection = [
        _selection(1, second_pass=False),
        _selection(2, second_pass=True),
        _selection(3, second_pass=False),
        _selection(4, second_pass=True),
    ]
    adjudication = [_adjudication(selection[1]), _adjudication(selection[3], boundary=True)]
    split = {
        "validation_candidate_ids": [selection[0]["candidate_id"], selection[1]["candidate_id"]],
        "pilot_test_candidate_ids": [selection[2]["candidate_id"], selection[3]["candidate_id"]],
    }
    return selection, adjudication, split


def test_build_promotes_resolved_fields_and_freezes_exact_splits():
    module = _load_module()
    selection, adjudication, split = _assets()
    config = _config()
    result = module.build_gold_freeze(
        selection,
        adjudication,
        split,
        config,
        dict(config["expected_input_sha256"]),
    )

    assert len(result["gold_rows"]) == 4
    assert len(result["validation_rows"]) == 2
    assert len(result["pilot_test_rows"]) == 2
    assert all(row["freeze_status"] == "frozen" for row in result["gold_rows"])
    assert all(row["gold_promotion_status"] == "promoted" for row in result["gold_rows"])
    assert all(row["clinically_validated"] is False for row in result["gold_rows"])
    resolved = {row["candidate_id"]: row for row in result["gold_rows"]}
    assert resolved[selection[1]["candidate_id"]]["expected_decision"] == "answer"
    assert resolved[selection[3]["candidate_id"]]["expected_decision"] == "boundary_refusal"
    assert result["summary"]["gold_promotion_performed"] is True
    assert result["summary"]["freeze_performed"] is True
    assert result["summary"]["clinically_validated"] is False
    assert result["summary"]["usage"] == module.zero_usage()


def test_build_fails_closed_on_hash_or_adjudication_coverage_drift():
    module = _load_module()
    selection, adjudication, split = _assets()
    config = _config()
    with pytest.raises(ValueError, match="hash"):
        module.build_gold_freeze(
            selection,
            adjudication,
            split,
            config,
            {**config["expected_input_sha256"], "selection": "wrong"},
        )
    with pytest.raises(ValueError, match="完整覆盖|coverage"):
        module.build_gold_freeze(
            selection,
            adjudication[:-1],
            split,
            config,
            dict(config["expected_input_sha256"]),
        )


def test_build_rejects_split_leakage_and_source_drift():
    module = _load_module()
    selection, adjudication, split = _assets()
    config = _config()
    leaked = deepcopy(selection)
    leaked[2]["independence_unit_id"] = leaked[0]["independence_unit_id"]
    with pytest.raises(ValueError, match="泄漏|leakage"):
        module.build_gold_freeze(
            leaked,
            adjudication,
            split,
            config,
            dict(config["expected_input_sha256"]),
        )
    drifted = deepcopy(adjudication)
    drifted[0]["page_number"] = "999"
    with pytest.raises(ValueError, match="漂移"):
        module.build_gold_freeze(
            selection,
            drifted,
            split,
            config,
            dict(config["expected_input_sha256"]),
        )


def test_boundary_refusal_must_remain_policy_grounded():
    module = _load_module()
    selection, adjudication, split = _assets()
    config = _config()
    adjudication[-1]["resolution_final_gold_evidence_status"] = "page_span_located"
    with pytest.raises(ValueError, match="boundary_refusal.*policy_rule"):
        module.build_gold_freeze(
            selection,
            adjudication,
            split,
            config,
            dict(config["expected_input_sha256"]),
        )


def test_legacy_boundary_policy_fields_are_canonicalized_and_audited():
    module = _load_module()
    selection, adjudication, split = _assets()
    config = _config()
    legacy = selection[0]
    legacy["expected_decision"] = "boundary_refusal"
    legacy["current_kb_support"] = "policy_rule"
    legacy["gold_evidence_status"] = "page_span_located"
    legacy["required_evidence_type"] = [
        "emergency_antibiotic_indication",
        "medical_safety_policy",
    ]

    result = module.build_gold_freeze(
        selection,
        adjudication,
        split,
        config,
        dict(config["expected_input_sha256"]),
    )

    promoted = {row["candidate_id"]: row for row in result["gold_rows"]}
    normalized = promoted[legacy["candidate_id"]]
    assert normalized["gold_evidence_status"] == "policy_rule"
    assert "safety_policy" in normalized["required_evidence_type"]
    assert "medical_safety_policy" not in normalized["required_evidence_type"]
    assert normalized["boundary_policy_normalization_status"] == "canonicalized"
    assert result["summary"]["boundary_policy_normalization_count"] == 1
    assert result["summary"]["boundary_policy_normalized_candidate_ids"] == [
        legacy["candidate_id"]
    ]


def test_write_outputs_are_immutable_and_idempotent(tmp_path: Path):
    module = _load_module()
    selection, adjudication, split = _assets()
    config = _config()
    result = module.build_gold_freeze(
        selection,
        adjudication,
        split,
        config,
        dict(config["expected_input_sha256"]),
    )
    first = module.write_frozen_outputs(result, tmp_path)
    second = module.write_frozen_outputs(result, tmp_path)
    assert first == second
    changed = deepcopy(result)
    changed["gold_rows"][0]["question"] = "被篡改的问题"
    with pytest.raises(FileExistsError, match="冻结|immutable"):
        module.write_frozen_outputs(changed, tmp_path)


def test_load_adjudication_reads_utf8_bom_csv(tmp_path: Path):
    module = _load_module()
    path = tmp_path / "adjudication.csv"
    path.write_text(
        "candidate_id,resolution_status,resolution_reason\n"
        "PMSQA-GOLD-001,accepted,作者确认\n",
        encoding="utf-8-sig",
    )

    rows = module._load_adjudication(path)

    assert rows == [
        {
            "candidate_id": "PMSQA-GOLD-001",
            "resolution_status": "accepted",
            "resolution_reason": "作者确认",
        }
    ]


def test_adjudication_audit_must_be_complete_and_match_csv_hash():
    module = _load_module()
    config = _config()
    audit = {
        "status": "author_adjudication_complete",
        "adjudicated_count": config["expected_adjudication_count"],
        "adjudication_csv_sha256": "adjudication-hash",
        "gold_promotion_performed": False,
        "freeze_performed": False,
        "clinically_validated": False,
    }

    module.validate_adjudication_audit(
        audit,
        config,
        adjudication_sha256="adjudication-hash",
    )

    with pytest.raises(ValueError, match="审计|hash"):
        module.validate_adjudication_audit(
            {**audit, "adjudication_csv_sha256": "wrong"},
            config,
            adjudication_sha256="adjudication-hash",
        )
    with pytest.raises(ValueError, match="审计|状态"):
        module.validate_adjudication_audit(
            {**audit, "status": "pending"},
            config,
            adjudication_sha256="adjudication-hash",
        )
