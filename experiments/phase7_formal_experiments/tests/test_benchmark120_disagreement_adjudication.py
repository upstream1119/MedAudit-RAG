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
    / "benchmark120_disagreement_adjudication.py"
)
FORMAL_CONFIG_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "configs"
    / "benchmark120_disagreement_adjudication_v0_1.json"
)


def _load_module():
    assert MODULE_PATH.exists(), "Benchmark120 adjudication module is missing"
    spec = importlib.util.spec_from_file_location(
        "benchmark120_disagreement_adjudication",
        MODULE_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(index: int) -> dict[str, str]:
    decision = "boundary_refusal" if index in {5, 11} else "answer"
    support = "policy_rule" if decision == "boundary_refusal" else "supported_by_current_kb"
    evidence = "policy_rule" if decision == "boundary_refusal" else "page_span_located"
    return {
        "resolution_order": str(index),
        "resolution_version": "benchmark120-second-pass-resolution-v0.1",
        "dataset_version": "benchmark-v1.0-selection-draft-v0.1",
        "kb_version": "KB-medium-v1",
        "protocol_version": "benchmark-protocol-v0.1",
        "candidate_id": f"PMSQA-TEST-{index:03d}",
        "pass2_item_id": f"B120-P2-{index:03d}",
        "question": f"第 {index} 条待审回答是否能由当前证据直接审核？",
        "dataset_split": "validation" if index <= 10 else "pilot_test",
        "selection_row_sha256": f"selection-{index:03d}",
        "source_id": "POLICY-SAFETY-001" if decision == "boundary_refusal" else "SRC-001",
        "source_title": "医疗安全策略" if decision == "boundary_refusal" else "测试指南",
        "source_filename": "medical_safety_policy.md" if decision == "boundary_refusal" else "test.pdf",
        "source_sha256": f"source-{index:03d}",
        "page_number": "0" if decision == "boundary_refusal" else str(index),
        "anchor_text_span": "证据只支持受限审计结论。",
        "policy_rule_ids": '["POLICY-SAFETY-001"]' if decision == "boundary_refusal" else "[]",
        "policy_evidence_text": "不得生成个体处方。" if decision == "boundary_refusal" else "",
        "first_pass_expected_decision": "boundary_refusal" if decision == "boundary_refusal" else "review_required",
        "first_pass_current_kb_support": "policy_rule" if decision == "boundary_refusal" else "partial_current_kb_support",
        "first_pass_gold_evidence_status": "page_span_located",
        "first_pass_required_evidence_type": "[]",
        "first_pass_required_claims": "[]",
        "first_pass_allowed_claims": '["仅允许受限结论"]',
        "first_pass_forbidden_claims": '["不得外推"]',
        "first_pass_missing_evidence_type": "[]",
        "first_pass_missing_information": "[]",
        "first_pass_risk_labels": '["review_required"]',
        "pass2_expected_decision": decision,
        "pass2_current_kb_support": support,
        "pass2_gold_evidence_status": evidence,
        "pass2_required_evidence_type": '["safety_policy"]' if decision == "boundary_refusal" else '["direct_page_span"]',
        "pass2_required_claims": '["当前证据支持受限审计结论"]',
        "pass2_allowed_claims": '["可以指出待审回答不受证据支持"]',
        "pass2_forbidden_claims": '["不得生成个体化治疗方案"]',
        "pass2_missing_evidence_type": "[]",
        "pass2_missing_information": "[]",
        "pass2_risk_labels": '["prescription_boundary"]' if decision == "boundary_refusal" else '["claim_overreach"]',
        "pass2_review_reason": "第二轮依据证据跨度形成更具体的受限审计结论。",
        "disagreement_fields": '["expected_decision","allowed_claims"]',
        "resolution_reviewer_id": "",
        "resolution_annotator_role": "",
        "resolution_reviewed_at": "",
        "resolution_status": "",
        "resolution_final_decision": "",
        "resolution_final_kb_support": "",
        "resolution_final_gold_evidence_status": "",
        "resolution_final_required_evidence_type": "",
        "resolution_final_required_claims": "",
        "resolution_final_allowed_claims": "",
        "resolution_final_forbidden_claims": "",
        "resolution_final_missing_evidence_type": "",
        "resolution_final_missing_information": "",
        "resolution_final_risk_labels": "",
        "resolution_reason": "",
    }


def _config(expected_count: int = 31) -> dict:
    return {
        "config_version": "benchmark120-disagreement-adjudication-config-v0.1",
        "adjudication_version": "benchmark120-disagreement-adjudication-v0.1",
        "dataset_version": "benchmark-v1.0-selection-draft-v0.1",
        "kb_version": "KB-medium-v1",
        "protocol_version": "benchmark-protocol-v0.1",
        "expected_candidate_count": expected_count,
        "batch_sizes": [8, 8, 8, 7] if expected_count == 31 else [expected_count],
        "allowed_resolution_status": ["accepted"],
        "allowed_decisions": [
            "answer",
            "review_required",
            "insufficient_evidence",
            "boundary_refusal",
        ],
        "allowed_kb_support": [
            "supported_by_current_kb",
            "partial_current_kb_support",
            "not_supported_by_current_kb",
            "policy_rule",
        ],
        "allowed_gold_evidence_status": ["page_span_located", "policy_rule"],
        "expected_input_sha256": {
            "resolution_queue": "queue-hash",
            "resolution_summary": "summary-hash",
        },
        "external_model_calls": 0,
    }


def _accept_all(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    reviewed = deepcopy(rows)
    for row in reviewed:
        row.update(
            {
                "resolution_reviewer_id": "PYH",
                "resolution_annotator_role": "author",
                "resolution_reviewed_at": "2026-08-16T18:00:00+08:00",
                "resolution_status": "accepted",
                "resolution_final_decision": row["pass2_expected_decision"],
                "resolution_final_kb_support": row["pass2_current_kb_support"],
                "resolution_final_gold_evidence_status": row[
                    "pass2_gold_evidence_status"
                ],
                "resolution_final_required_evidence_type": row[
                    "pass2_required_evidence_type"
                ],
                "resolution_final_required_claims": row["pass2_required_claims"],
                "resolution_final_allowed_claims": row["pass2_allowed_claims"],
                "resolution_final_forbidden_claims": row[
                    "pass2_forbidden_claims"
                ],
                "resolution_final_missing_evidence_type": row[
                    "pass2_missing_evidence_type"
                ],
                "resolution_final_missing_information": row[
                    "pass2_missing_information"
                ],
                "resolution_final_risk_labels": row["pass2_risk_labels"],
                "resolution_reason": "逐条核对两轮字段和原始证据后，采用更具体的受限审计口径。",
            }
        )
    return reviewed


def test_prepare_is_deterministic_and_keeps_author_fields_blank():
    module = _load_module()
    rows = [_row(index) for index in range(1, 32)]
    config = _config()
    observed = dict(config["expected_input_sha256"])

    first = module.prepare_adjudication_batches(rows, config, observed)
    second = module.prepare_adjudication_batches(list(reversed(rows)), config, observed)

    assert first == second
    assert [len(batch["rows"]) for batch in first] == [8, 8, 8, 7]
    assert all(
        not str(row.get(field, "")).strip()
        for batch in first
        for row in batch["rows"]
        for field in module.RESOLUTION_AUTHOR_FIELDS
    )


def test_prepare_fails_closed_on_hash_drift_or_precompleted_author_fields():
    module = _load_module()
    rows = [_row(index) for index in range(1, 3)]
    config = _config(expected_count=2)

    with pytest.raises(ValueError, match="hash"):
        module.prepare_adjudication_batches(
            rows,
            config,
            {"resolution_queue": "wrong", "resolution_summary": "summary-hash"},
        )

    rows[0]["resolution_reviewer_id"] = "PYH"
    with pytest.raises(ValueError, match="预填|非空"):
        module.prepare_adjudication_batches(
            rows,
            config,
            dict(config["expected_input_sha256"]),
        )


def test_validate_requires_complete_valid_author_adjudication():
    module = _load_module()
    rows = [_row(index) for index in range(1, 3)]
    config = _config(expected_count=2)
    observed = dict(config["expected_input_sha256"])
    prepared = module.prepare_adjudication_batches(rows, config, observed)[0]["rows"]
    reviewed = _accept_all(prepared)
    reviewed[0]["resolution_final_forbidden_claims"] = "not-json"

    with pytest.raises(ValueError, match="JSON"):
        module.validate_adjudication_batch(reviewed, prepared, config, observed)

    reviewed = _accept_all(prepared)
    reviewed[0]["source_id"] = "SRC-DRIFT"
    with pytest.raises(ValueError, match="漂移"):
        module.validate_adjudication_batch(reviewed, prepared, config, observed)


def test_boundary_refusal_must_use_policy_rule_evidence():
    module = _load_module()
    rows = [_row(5)]
    config = _config(expected_count=1)
    observed = dict(config["expected_input_sha256"])
    prepared = module.prepare_adjudication_batches(rows, config, observed)[0]["rows"]
    reviewed = _accept_all(prepared)
    reviewed[0]["resolution_final_gold_evidence_status"] = "page_span_located"

    with pytest.raises(ValueError, match="boundary_refusal.*policy_rule"):
        module.validate_adjudication_batch(reviewed, prepared, config, observed)


def test_finalize_requires_exact_coverage_and_never_promotes_or_freezes():
    module = _load_module()
    rows = [_row(index) for index in range(1, 5)]
    config = _config(expected_count=4)
    observed = dict(config["expected_input_sha256"])
    prepared_batches = module.prepare_adjudication_batches(rows, config, observed)
    prepared = [row for batch in prepared_batches for row in batch["rows"]]
    reviewed = _accept_all(prepared)

    with pytest.raises(ValueError, match="4.*exactly once|完整覆盖"):
        module.finalize_adjudications(reviewed[:-1], prepared, config, observed)

    result = module.finalize_adjudications(reviewed, prepared, config, observed)

    assert result["summary"]["status"] == "author_adjudication_complete"
    assert result["summary"]["adjudicated_count"] == 4
    assert result["summary"]["gold_promotion_performed"] is False
    assert result["summary"]["freeze_performed"] is False
    assert result["summary"]["usage"] == module.zero_usage()


def test_formal_config_locks_current_resolution_assets():
    module = _load_module()
    assert FORMAL_CONFIG_PATH.exists(), "Formal adjudication config is missing"
    config = module.load_config(FORMAL_CONFIG_PATH)

    assert config["expected_candidate_count"] == 31
    assert config["batch_sizes"] == [8, 8, 8, 7]
    assert config["expected_input_sha256"] == {
        "resolution_queue": "5d10786a259b5a71ecb4da85c5092bdfba0c6bc09c469ee14def31213c90487a",
        "resolution_summary": "4576d544b9f0dbd1175cc6c53481003828ae732294bc67a81fb71e23419bc932",
    }
    assert config["external_model_calls"] == 0
