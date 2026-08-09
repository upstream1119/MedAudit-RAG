from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "benchmark_pass2_queue.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_pass2_queue", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(
    candidate_id: str,
    decision: str,
    *,
    outcome: str = "accepted",
    unit_id: str | None = None,
    reaudit_status: str = "",
) -> dict[str, str]:
    return {
        "annotation_order": "1",
        "candidate_id": candidate_id,
        "independence_unit_id": unit_id or f"IU-{candidate_id}",
        "question": f"原始问题 {candidate_id}",
        "source_id": "SRC-1",
        "source_title": "儿科用药指南",
        "source_filename": "guide.pdf",
        "source_sha256": "a" * 64,
        "source_type": "clinical_guideline",
        "source_year": "2025",
        "jurisdiction": "CN",
        "source_can_support": '["剂量审核"]',
        "source_cannot_support": '["个体处方"]',
        "page_number": "8",
        "anchor_text_span": "可定位的原文证据。",
        "evidence_scope": "仅支持剂量审核。",
        "age_scope": "儿童",
        "applicability_conditions": "需结合年龄与体重。",
        "supported_claim_types": '["dose"]',
        "scope_check": "within_can_support",
        "evidence_anchor_ids": '["ANCH-1"]',
        "evidence_anchor_group_id": "EAG-1",
        "provisional_fact_cluster_id": "FC-1",
        "policy_rule_ids": "[]",
        "candidate_role": "direct_support",
        "challenge_type": "",
        "provisional_expected_decision": decision,
        "provisional_risk_labels": '["dose"]',
        "current_kb_support": "supported_by_current_kb",
        "missing_evidence_type": "[]",
        "pass1_reviewer_id": "author-pyh",
        "pass1_reviewed_at": "2026-08-02T10:00:00+08:00",
        "pass1_outcome": outcome,
        "pass1_final_question": f"最终问题 {candidate_id}",
        "pass1_expected_decision": decision,
        "pass1_current_kb_support": "supported_by_current_kb",
        "pass1_review_reason": "第一轮核验理由不得泄露。",
        "pass1_overlap_reaudit_status": reaudit_status,
    }


def _config() -> dict:
    return {
        "config_version": "benchmark-pass2-queue-config-v0.1",
        "dataset_version": "benchmark-v1.0-pass2-pending",
        "kb_version": "KB-medium-v1",
        "protocol_version": "benchmark-protocol-v0.1",
        "blind_seed": 20260802,
        "expected_original_promotable_before_scope": 3,
        "expected_scope_exclusions": 1,
        "expected_supplement_promotable": 1,
        "expected_merged_count": 3,
        "target_final_count": 3,
        "target_final_decision_distribution": {
            "answer": 1,
            "boundary_refusal": 1,
            "review_required": 1,
        },
        "external_model_calls": 0,
    }


def test_build_pass2_queue_is_deterministic_blinded_and_keeps_evidence():
    module = _load_module()
    original = [
        _row("C-1", "answer"),
        _row("C-2", "review_required", outcome="revise", reaudit_status="clear"),
        _row("C-3", "answer"),
    ]
    supplement = [_row("S-1", "boundary_refusal")]
    scope_audit = {"flagged_rows": [{"candidate_id": "C-3"}]}
    original_copy = deepcopy(original)

    first = module.build_pass2_artifacts(original, supplement, scope_audit, _config())
    second = module.build_pass2_artifacts(original, supplement, scope_audit, _config())

    assert first == second
    assert original == original_copy
    assert len(first["review_queue"]) == 3
    assert len(first["linkage_records"]) == 3
    assert {row["question"] for row in first["review_queue"]} == {
        "最终问题 C-1",
        "最终问题 C-2",
        "最终问题 S-1",
    }
    visible = first["review_queue"][0]
    assert visible["source_title"] == "儿科用药指南"
    assert visible["page_number"] == "8"
    assert visible["anchor_text_span"] == "可定位的原文证据。"
    assert "candidate_id" not in visible
    assert not any(key.startswith("pass1_") for key in visible)
    assert not any(key.startswith("provisional_") for key in visible)
    assert "current_kb_support" not in visible
    assert "missing_evidence_type" not in visible
    assert "candidate_role" not in visible
    assert "challenge_type" not in visible
    assert visible["pass2_outcome"] == ""
    assert visible["pass2_expected_decision"] == ""


def test_linkage_preserves_first_pass_history_and_one_to_one_mapping():
    module = _load_module()
    original = [
        _row("C-1", "answer"),
        _row("C-2", "review_required"),
        _row("C-3", "answer"),
    ]
    supplement = [_row("S-1", "boundary_refusal")]
    artifacts = module.build_pass2_artifacts(
        original,
        supplement,
        {"flagged_rows": [{"candidate_id": "C-3"}]},
        _config(),
    )

    blind_ids = {row["pass2_item_id"] for row in artifacts["review_queue"]}
    linkage_ids = {row["pass2_item_id"] for row in artifacts["linkage_records"]}
    assert blind_ids == linkage_ids
    assert {row["candidate_id"] for row in artifacts["linkage_records"]} == {
        "C-1",
        "C-2",
        "S-1",
    }
    assert {row["origin_pool"] for row in artifacts["linkage_records"]} == {
        "original",
        "supplement",
    }
    assert all(row["pass1_expected_decision"] for row in artifacts["linkage_records"])
    assert artifacts["summary"]["decision_distribution"] == {
        "answer": 1,
        "boundary_refusal": 1,
        "review_required": 1,
    }
    assert artifacts["summary"]["target_distribution_feasible"] is True
    assert artifacts["summary"]["surplus_by_decision"] == {
        "answer": 0,
        "boundary_refusal": 0,
        "review_required": 0,
    }


def test_fail_closed_on_invalid_scope_or_promotion_state():
    module = _load_module()
    original = [
        _row("C-1", "answer"),
        _row("C-2", "review_required", outcome="revise", reaudit_status="pending"),
        _row("C-3", "answer"),
    ]
    supplement = [_row("S-1", "boundary_refusal")]

    with pytest.raises(ValueError, match="复审"):
        module.build_pass2_artifacts(
            original,
            supplement,
            {"flagged_rows": [{"candidate_id": "C-3"}]},
            _config(),
        )

    original[1] = _row("C-2", "review_required")
    with pytest.raises(ValueError, match="范围审计"):
        module.build_pass2_artifacts(
            original,
            supplement,
            {"flagged_rows": [{"candidate_id": "UNKNOWN"}]},
            _config(),
        )


def test_fail_closed_on_duplicate_ids_or_unexpected_counts():
    module = _load_module()
    original = [
        _row("C-1", "answer"),
        _row("C-2", "review_required"),
        _row("C-3", "answer"),
    ]
    supplement = [_row("C-1", "boundary_refusal")]

    with pytest.raises(ValueError, match="重复 candidate_id"):
        module.build_pass2_artifacts(
            original,
            supplement,
            {"flagged_rows": [{"candidate_id": "C-3"}]},
            _config(),
        )

    supplement = [_row("S-1", "boundary_refusal")]
    config = _config()
    config["expected_merged_count"] = 99
    with pytest.raises(ValueError, match="合并候选数量"):
        module.build_pass2_artifacts(
            original,
            supplement,
            {"flagged_rows": [{"candidate_id": "C-3"}]},
            config,
        )


def test_usage_and_workflow_boundary_are_explicit():
    module = _load_module()
    original = [
        _row("C-1", "answer"),
        _row("C-2", "review_required"),
        _row("C-3", "answer"),
    ]
    artifacts = module.build_pass2_artifacts(
        original,
        [_row("S-1", "boundary_refusal")],
        {"flagged_rows": [{"candidate_id": "C-3"}]},
        _config(),
    )

    assert artifacts["summary"]["usage"] == {
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
    }
    assert "not frozen" in artifacts["summary"]["workflow_boundary"]
    assert "not expert-validated" in artifacts["summary"]["workflow_boundary"]


def test_fail_closed_when_final_target_is_not_feasible():
    module = _load_module()
    original = [
        _row("C-1", "answer"),
        _row("C-2", "review_required"),
        _row("C-3", "answer"),
    ]
    config = _config()
    config["target_final_decision_distribution"] = {
        "answer": 1,
        "boundary_refusal": 2,
    }

    with pytest.raises(ValueError, match="目标决策分布不可满足"):
        module.build_pass2_artifacts(
            original,
            [_row("S-1", "boundary_refusal")],
            {"flagged_rows": [{"candidate_id": "C-3"}]},
            config,
        )
