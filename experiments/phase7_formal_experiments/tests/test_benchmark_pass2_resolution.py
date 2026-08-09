from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "benchmark_pass2_resolution.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "resolution module must exist"
    spec = importlib.util.spec_from_file_location(
        "benchmark_pass2_resolution", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _linkage_record(
    item_id: str,
    candidate_id: str,
    *,
    decision: str,
    support: str,
    outcome: str = "accepted",
    question: str = "儿童用药证据是否足够？",
) -> dict:
    return {
        "pass2_item_id": item_id,
        "candidate_id": candidate_id,
        "origin_pool": "original",
        "source_row_sha256": f"source-row-{candidate_id}",
        "independence_unit_id": f"IU-{candidate_id}",
        "evidence_anchor_ids": f'["ANCH-{candidate_id}"]',
        "evidence_anchor_group_id": f"EAG-{candidate_id}",
        "provisional_fact_cluster_id": f"FC-{candidate_id}",
        "pass1_outcome": outcome,
        "pass1_expected_decision": decision,
        "pass1_current_kb_support": support,
        "pass1_gold_evidence_status": "page_span_located",
        "pass1_final_question": question,
        "pass1_review_reason": "第一轮逐页核对证据边界。",
    }


def _pass2_row(
    order: int,
    item_id: str,
    *,
    decision: str,
    support: str,
    outcome: str = "accepted",
    question: str = "儿童用药证据是否足够？",
) -> dict:
    return {
        "pass2_order": str(order),
        "pass2_item_id": item_id,
        "question": question,
        "source_id": "SRC-001",
        "source_title": "儿科指南",
        "source_filename": "pediatric_guideline.pdf",
        "source_sha256": "source-file-sha",
        "source_type": "clinical_guideline",
        "source_year": "2024",
        "jurisdiction": "CN",
        "source_can_support": '["儿科用药证据边界"]',
        "source_cannot_support": '["个体化处方"]',
        "page_number": "8",
        "anchor_text_span": "可定位的指南证据片段。",
        "evidence_scope": "仅支持指南证据范围内结论。",
        "age_scope": "儿童",
        "applicability_conditions": "需结合具体适用条件。",
        "supported_claim_types": '["evidence_boundary"]',
        "scope_check": "within_can_support",
        "policy_rule_ids": "[]",
        "pass2_reviewer_id": "author-pyh",
        "pass2_annotator_role": "author",
        "pass2_reviewed_at": "2026-08-09T12:00:00+08:00",
        "pass2_outcome": outcome,
        "pass2_final_question": question,
        "pass2_expected_decision": decision,
        "pass2_current_kb_support": support,
        "pass2_gold_evidence_status": "page_span_located",
        "pass2_required_evidence_type": '["direct_evidence"]',
        "pass2_required_claims": '["说明证据边界"]',
        "pass2_allowed_claims": '["提示人工复核"]',
        "pass2_forbidden_claims": '["无证据处方"]',
        "pass2_missing_evidence_type": "[]",
        "pass2_missing_information": "[]",
        "pass2_risk_labels": '["evidence_scope"]',
        "pass2_issues_found": "[]",
        "pass2_review_reason": "第二轮独立核对来源、页码和证据边界。",
    }


def _inputs() -> tuple[dict, list[dict], dict, dict]:
    linkage = {
        "dataset_version": "benchmark-v1.0-pass2-pending",
        "linkage_version": "benchmark-pass2-linkage-v0.1",
        "records": [
            _linkage_record(
                "P2-1",
                "C-1",
                decision="answer",
                support="supported_by_current_kb",
            ),
            _linkage_record(
                "P2-2",
                "C-2",
                decision="review_required",
                support="partial_current_kb_support",
            ),
            _linkage_record(
                "P2-3",
                "C-3",
                decision="answer",
                support="supported_by_current_kb",
            ),
        ],
        "workflow_boundary": "第一轮与第二轮通过盲化 ID 关联。",
    }
    pass2_rows = [
        _pass2_row(
            1,
            "P2-1",
            decision="answer",
            support="supported_by_current_kb",
        ),
        _pass2_row(
            2,
            "P2-2",
            decision="insufficient_evidence",
            support="not_supported_by_current_kb",
        ),
        _pass2_row(
            3,
            "P2-3",
            decision="answer",
            support="supported_by_current_kb",
            outcome="reject",
        ),
    ]
    progress = {
        "candidate_count": 3,
        "completed_count": 3,
        "pending_count": 0,
        "promotable_count": 2,
        "status": "complete",
        "queue_sha256": "queue-sha",
    }
    config = {
        "config_version": "benchmark-pass2-resolution-config-v0.1",
        "resolution_version": "benchmark-pass2-resolution-v0.1",
        "dataset_version": "benchmark-v1.0-pass2-pending",
        "kb_version": "KB-medium-v1",
        "expected_candidate_count": 3,
        "expected_disagreement_count": 1,
        "expected_excluded_count": 1,
        "target_final_count": 120,
        "target_final_decision_distribution": {
            "answer": 40,
            "review_required": 40,
            "insufficient_evidence": 24,
            "boundary_refusal": 16,
        },
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
        "external_model_calls": 0,
    }
    return linkage, pass2_rows, progress, config


def _build(module, linkage, rows, progress, config):
    return module.build_resolution_artifacts(
        linkage,
        rows,
        progress,
        config,
        linkage_sha256="linkage-sha",
        pass2_queue_sha256="queue-sha",
        progress_sha256="progress-sha",
    )


def test_resolution_queue_contains_only_promotable_core_disagreements():
    module = _load_module()
    linkage, rows, progress, config = _inputs()

    artifacts = _build(module, linkage, rows, progress, config)

    assert len(artifacts["resolution_queue"]) == 1
    assert artifacts["resolution_queue"][0]["candidate_id"] == "C-2"
    assert json.loads(artifacts["resolution_queue"][0]["disagreement_fields"]) == [
        "expected_decision",
        "current_kb_support",
    ]
    summary = artifacts["summary"]
    assert summary["linked_count"] == 3
    assert summary["core_agreement_count"] == 1
    assert summary["resolution_candidate_count"] == 1
    assert summary["excluded_count"] == 1
    assert summary["status"] == "resolution_pending"


def test_resolution_row_preserves_both_histories_and_leaves_resolution_empty():
    module = _load_module()
    linkage, rows, progress, config = _inputs()
    original_linkage = deepcopy(linkage)
    original_rows = deepcopy(rows)

    row = _build(module, linkage, rows, progress, config)["resolution_queue"][0]

    assert linkage == original_linkage
    assert rows == original_rows
    assert row["pass2_item_id"] == "P2-2"
    assert row["independence_unit_id"] == "IU-C-2"
    assert row["source_title"] == "儿科指南"
    assert row["page_number"] == "8"
    assert row["anchor_text_span"] == "可定位的指南证据片段。"
    assert row["pass1_expected_decision"] == "review_required"
    assert row["pass2_expected_decision"] == "insufficient_evidence"
    assert row["pass1_review_reason"]
    assert row["pass2_review_reason"]
    for field in module.RESOLUTION_EMPTY_FIELDS:
        assert row[field] == ""


def test_fail_closed_until_pass2_is_complete_and_hash_matches():
    module = _load_module()
    linkage, rows, progress, config = _inputs()
    incomplete = deepcopy(progress)
    incomplete["status"] = "in_progress"
    incomplete["pending_count"] = 1
    with pytest.raises(ValueError, match="第二轮尚未完整完成"):
        _build(module, linkage, rows, incomplete, config)

    with pytest.raises(ValueError, match="队列哈希"):
        module.build_resolution_artifacts(
            linkage,
            rows,
            progress,
            config,
            linkage_sha256="linkage-sha",
            pass2_queue_sha256="different-sha",
            progress_sha256="progress-sha",
        )

    wrong_promotable = deepcopy(progress)
    wrong_promotable["promotable_count"] = 1
    with pytest.raises(ValueError, match="可晋升数量"):
        _build(module, linkage, rows, wrong_promotable, config)


def test_fail_closed_on_duplicate_linkage_or_question_drift():
    module = _load_module()
    linkage, rows, progress, config = _inputs()
    duplicate = deepcopy(linkage)
    duplicate["records"].append(deepcopy(duplicate["records"][0]))
    with pytest.raises(ValueError, match="重复 pass2_item_id"):
        _build(module, duplicate, rows, progress, config)

    drifted = deepcopy(rows)
    drifted[1]["pass2_final_question"] = "发生漂移的问题"
    with pytest.raises(ValueError, match="问题文本漂移"):
        _build(module, linkage, drifted, progress, config)


def test_resolution_artifacts_are_deterministic_and_use_no_external_model():
    module = _load_module()
    linkage, rows, progress, config = _inputs()

    first = _build(module, linkage, rows, progress, config)
    second = _build(module, linkage, rows, progress, config)

    assert first == second
    assert first["summary"]["usage"] == {
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
    }
