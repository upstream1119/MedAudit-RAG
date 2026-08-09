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
    / "benchmark_pass2_annotation.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_pass2_annotation", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _queue_row(order: int = 1, item_id: str = "P2-1", policy: str = "[]") -> dict:
    return {
        "pass2_order": str(order),
        "pass2_item_id": item_id,
        "question": "儿童用药证据是否足够？",
        "source_title": "儿科指南",
        "page_number": "8",
        "anchor_text_span": "可定位的证据。",
        "policy_rule_ids": policy,
        "pass2_reviewer_id": "",
        "pass2_annotator_role": "",
        "pass2_reviewed_at": "",
        "pass2_outcome": "",
        "pass2_final_question": "",
        "pass2_expected_decision": "",
        "pass2_current_kb_support": "",
        "pass2_gold_evidence_status": "",
        "pass2_required_evidence_type": "",
        "pass2_required_claims": "",
        "pass2_allowed_claims": "",
        "pass2_forbidden_claims": "",
        "pass2_missing_evidence_type": "",
        "pass2_missing_information": "",
        "pass2_risk_labels": "",
        "pass2_issues_found": "",
        "pass2_review_reason": "",
    }


def _record(
    item_id: str = "P2-1",
    *,
    outcome: str = "accepted",
    decision: str = "answer",
    question: str = "儿童用药证据是否足够？",
) -> dict:
    support = {
        "answer": "supported_by_current_kb",
        "review_required": "partial_current_kb_support",
        "insufficient_evidence": "not_supported_by_current_kb",
        "boundary_refusal": "policy_rule",
    }[decision]
    return {
        "pass2_item_id": item_id,
        "pass2_reviewer_id": "author-pyh",
        "pass2_annotator_role": "author",
        "pass2_reviewed_at": "2026-08-02T18:00:00+08:00",
        "pass2_outcome": outcome,
        "pass2_final_question": question,
        "pass2_expected_decision": decision,
        "pass2_current_kb_support": support,
        "pass2_gold_evidence_status": "page_span_located",
        "pass2_required_evidence_type": ["dose_evidence"],
        "pass2_required_claims": ["仅支持证据范围内结论"],
        "pass2_allowed_claims": ["说明证据边界"],
        "pass2_forbidden_claims": ["不得给出无证据处方"],
        "pass2_missing_evidence_type": [],
        "pass2_missing_information": [],
        "pass2_risk_labels": [decision],
        "pass2_issues_found": [],
        "pass2_review_reason": "第二轮独立重新核对来源、页码和证据边界。",
    }


def _config(count: int = 1) -> dict:
    return {
        "config_version": "benchmark-pass2-annotation-config-v0.1",
        "annotation_version": "benchmark-pass2-annotation-v0.1",
        "dataset_version": "benchmark-v1.0-pass2-pending",
        "kb_version": "KB-medium-v1",
        "expected_candidate_count": count,
        "annotator_role": "author",
        "allowed_outcomes": ["accepted", "revise", "reject"],
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
        "policy_rule_id": "POLICY-SAFETY-001",
        "external_model_calls": 0,
    }


def _batch(records: list[dict]) -> dict:
    return {
        "annotation_version": "benchmark-pass2-annotation-v0.1",
        "batch_id": "phase7-b3-pass2-batch01",
        "batch_scope": "pass2_order 1-1",
        "dataset_version": "benchmark-v1.0-pass2-pending",
        "kb_version": "KB-medium-v1",
        "record_count": len(records),
        "review_mode": "blinded second-pass author verification",
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
        "records": records,
    }


def test_apply_batch_updates_only_matching_blind_items_without_mutating_inputs():
    module = _load_module()
    queue = [_queue_row()]
    batch = _batch([_record()])
    queue_copy = deepcopy(queue)

    updated, summary = module.apply_pass2_batch(queue, batch, _config())

    assert queue == queue_copy
    assert updated[0]["pass2_outcome"] == "accepted"
    assert updated[0]["pass2_expected_decision"] == "answer"
    assert summary["completed_count"] == 1
    assert summary["pending_count"] == 0
    assert summary["promotable_count"] == 1
    assert summary["decision_distribution"] == {"answer": 1}
    assert summary["usage"]["external_model_calls"] == 0


def test_revise_requires_changed_question_and_is_not_silently_promoted():
    module = _load_module()
    queue = [_queue_row()]
    unchanged = _batch([_record(outcome="revise")])
    with pytest.raises(ValueError, match="修改题必须提供不同的问题"):
        module.apply_pass2_batch(queue, unchanged, _config())

    changed = _batch(
        [_record(outcome="revise", question="儿童用药证据在限定条件下是否足够？")]
    )
    _, summary = module.apply_pass2_batch(queue, changed, _config())
    assert summary["revision_required_count"] == 1
    assert summary["promotable_count"] == 0


def test_boundary_refusal_requires_bound_policy_rule():
    module = _load_module()
    queue = [_queue_row()]
    batch = _batch([_record(decision="boundary_refusal")])
    with pytest.raises(ValueError, match="安全政策"):
        module.apply_pass2_batch(queue, batch, _config())

    queue = [_queue_row(policy='["POLICY-SAFETY-001"]')]
    updated, _ = module.apply_pass2_batch(queue, batch, _config())
    assert updated[0]["pass2_expected_decision"] == "boundary_refusal"


def test_fail_closed_on_wrong_order_duplicate_id_or_garbled_metadata():
    module = _load_module()
    queue = [_queue_row(), _queue_row(order=2, item_id="P2-2")]
    duplicate = _batch([_record(), _record()])
    with pytest.raises(ValueError, match="重复 blind ID"):
        module.apply_pass2_batch(queue, duplicate, _config(count=2))

    wrong_scope = _batch([_record(item_id="P2-2")])
    with pytest.raises(ValueError, match="连续顺序"):
        module.apply_pass2_batch(queue, wrong_scope, _config(count=2))

    garbled = _batch([_record()])
    garbled["records"][0]["pass2_review_reason"] = "???"
    with pytest.raises(ValueError, match="乱码"):
        module.apply_pass2_batch(queue[:1], garbled, _config())
