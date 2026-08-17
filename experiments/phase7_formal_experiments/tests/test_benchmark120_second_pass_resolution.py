from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "benchmark120_second_pass_resolution.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "Benchmark120 resolution module must exist"
    spec = importlib.util.spec_from_file_location(
        "benchmark120_second_pass_resolution", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _canonical_sha256(payload: dict) -> str:
    content = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _selection_row(
    candidate_id: str,
    *,
    decision: str,
    support: str,
    required_claims: list[str],
    allowed_claims: list[str],
) -> dict:
    return {
        "candidate_id": candidate_id,
        "question": f"{candidate_id} 的证据审计问题？",
        "dataset_split": "validation",
        "selection_version": "benchmark120-selection-draft-v0.1",
        "dataset_version": "benchmark-v1.0-selection-draft-v0.1",
        "kb_version": "KB-medium-v1",
        "source_id": "SRC-001",
        "source_title": "儿科临床指南",
        "source_filename": "pediatric_guideline.pdf",
        "source_sha256": "source-file-sha",
        "source_type": "clinical_guideline",
        "source_year": "2024",
        "jurisdiction": "CN",
        "page_number": 8,
        "anchor_text_span": "可定位的指南证据片段。",
        "policy_rule_ids": [],
        "expected_decision": decision,
        "current_kb_support": support,
        "gold_evidence_status": "page_span_located",
        "required_evidence_type": ["direct_evidence", "dose_frequency"],
        "required_claims": required_claims,
        "allowed_claims": allowed_claims,
        "forbidden_claims": ["无证据处方", "夸大证据强度"],
        "missing_evidence_type": [],
        "missing_information": [],
        "risk_labels": ["evidence_scope", "medication_safety"],
        "requires_second_pass": True,
    }


def _linkage_row(item_id: str, selection: dict) -> dict:
    return {
        "pass2_item_id": item_id,
        "candidate_id": selection["candidate_id"],
        "question": selection["question"],
        "dataset_split": selection["dataset_split"],
        "selection_version": selection["selection_version"],
        "source_id": selection["source_id"],
        "page_number": selection["page_number"],
        "selection_row_sha256": _canonical_sha256(selection),
        "first_pass_expected_decision": selection["expected_decision"],
        "first_pass_current_kb_support": selection["current_kb_support"],
        "first_pass_gold_evidence_status": selection["gold_evidence_status"],
        "first_pass_required_evidence_type": selection["required_evidence_type"],
        "first_pass_required_claims": selection["required_claims"],
        "first_pass_allowed_claims": selection["allowed_claims"],
        "first_pass_forbidden_claims": selection["forbidden_claims"],
        "first_pass_missing_evidence_type": selection["missing_evidence_type"],
        "first_pass_missing_information": selection["missing_information"],
        "first_pass_risk_labels": selection["risk_labels"],
    }


def _pass2_row(
    order: int,
    item_id: str,
    selection: dict,
    *,
    decision: str,
    support: str,
    required_claims: list[str],
    allowed_claims: list[str],
) -> dict:
    return {
        "pass2_order": str(order),
        "pass2_item_id": item_id,
        "pass2_annotation_version": "benchmark120-second-pass-v0.1",
        "dataset_version": selection["dataset_version"],
        "kb_version": selection["kb_version"],
        "protocol_version": "benchmark-protocol-v0.1",
        "question": selection["question"],
        "source_id": selection["source_id"],
        "source_title": selection["source_title"],
        "source_filename": selection["source_filename"],
        "source_sha256": selection["source_sha256"],
        "source_type": selection["source_type"],
        "source_year": selection["source_year"],
        "jurisdiction": selection["jurisdiction"],
        "page_number": str(selection["page_number"]),
        "anchor_text_span": selection["anchor_text_span"],
        "policy_rule_ids": "[]",
        "policy_evidence_text": "",
        "pass2_reviewer_id": "PYH",
        "pass2_annotator_role": "author",
        "pass2_reviewed_at": "2026-08-15T12:00:00+08:00",
        "pass2_outcome": "accepted",
        "pass2_final_question": selection["question"],
        "pass2_expected_decision": decision,
        "pass2_current_kb_support": support,
        "pass2_gold_evidence_status": selection["gold_evidence_status"],
        "pass2_required_evidence_type": json.dumps(
            list(reversed(selection["required_evidence_type"])), ensure_ascii=False
        ),
        "pass2_required_claims": json.dumps(required_claims, ensure_ascii=False),
        "pass2_allowed_claims": json.dumps(allowed_claims, ensure_ascii=False),
        "pass2_forbidden_claims": json.dumps(
            list(reversed(selection["forbidden_claims"])), ensure_ascii=False
        ),
        "pass2_missing_evidence_type": "[]",
        "pass2_missing_information": "[]",
        "pass2_risk_labels": json.dumps(
            list(reversed(selection["risk_labels"])), ensure_ascii=False
        ),
        "pass2_issues_found": "[]",
        "pass2_review_reason": "第二轮独立复核证据边界。",
    }


def _inputs() -> tuple[list[dict], dict, list[dict], dict, dict]:
    first = _selection_row(
        "C-1",
        decision="review_required",
        support="partial_current_kb_support",
        required_claims=["第一轮必要主张"],
        allowed_claims=["第一轮允许主张"],
    )
    second = _selection_row(
        "C-2",
        decision="answer",
        support="supported_by_current_kb",
        required_claims=["主张甲", "主张乙"],
        allowed_claims=["边界甲", "边界乙"],
    )
    selection_rows = [first, second]
    linkage = {
        "linkage_version": "benchmark120-second-pass-v0.1",
        "record_count": 2,
        "records": [
            _linkage_row("P2-1", first),
            _linkage_row("P2-2", second),
        ],
    }
    pass2_rows = [
        _pass2_row(
            1,
            "P2-1",
            first,
            decision="answer",
            support="supported_by_current_kb",
            required_claims=["第二轮必要主张"],
            allowed_claims=["第二轮允许主张"],
        ),
        _pass2_row(
            2,
            "P2-2",
            second,
            decision="answer",
            support="supported_by_current_kb",
            required_claims=list(reversed(second["required_claims"])),
            allowed_claims=list(reversed(second["allowed_claims"])),
        ),
    ]
    progress = {
        "annotation_version": "benchmark120-second-pass-v0.1",
        "dataset_version": "benchmark-v1.0-selection-draft-v0.1",
        "kb_version": "KB-medium-v1",
        "candidate_count": 2,
        "completed_count": 2,
        "pending_count": 0,
        "promotable_count": 2,
        "revision_required_count": 0,
        "outcome_distribution": {"accepted": 2},
        "status": "complete",
        "queue_sha256": "queue-sha",
    }
    config = {
        "config_version": "benchmark120-second-pass-resolution-config-v0.1",
        "resolution_version": "benchmark120-second-pass-resolution-v0.1",
        "dataset_version": "benchmark-v1.0-selection-draft-v0.1",
        "kb_version": "KB-medium-v1",
        "protocol_version": "benchmark-protocol-v0.1",
        "expected_selection_count": 2,
        "expected_second_pass_count": 2,
        "expected_resolution_count": 1,
        "expected_full_agreement_count": 1,
        "allowed_decisions": ["answer", "review_required", "boundary_refusal"],
        "allowed_kb_support": [
            "supported_by_current_kb",
            "partial_current_kb_support",
            "policy_rule",
        ],
        "allowed_gold_evidence_status": ["page_span_located", "policy_rule"],
        "external_model_calls": 0,
    }
    return selection_rows, linkage, pass2_rows, progress, config


def _build(module, selection_rows, linkage, rows, progress, config):
    return module.build_resolution_artifacts(
        selection_rows,
        linkage,
        rows,
        progress,
        config,
        selection_sha256="selection-sha",
        linkage_sha256="linkage-sha",
        pass2_queue_sha256="queue-sha",
        progress_sha256="progress-sha",
    )


def test_only_true_disagreements_enter_resolution_queue():
    module = _load_module()
    selection_rows, linkage, rows, progress, config = _inputs()

    artifacts = _build(
        module, selection_rows, linkage, rows, progress, config
    )

    assert len(artifacts["linked_comparison"]) == 2
    assert len(artifacts["resolution_queue"]) == 1
    resolution = artifacts["resolution_queue"][0]
    assert resolution["candidate_id"] == "C-1"
    assert json.loads(resolution["disagreement_fields"]) == [
        "expected_decision",
        "current_kb_support",
        "required_claims",
        "allowed_claims",
    ]
    agreement = artifacts["linked_comparison"][1]
    assert agreement["candidate_id"] == "C-2"
    assert agreement["disagreement_fields"] == []
    assert artifacts["summary"]["linked_count"] == 2
    assert artifacts["summary"]["full_agreement_count"] == 1
    assert artifacts["summary"]["resolution_candidate_count"] == 1


def test_selection_hash_is_recomputed_before_linkage_is_trusted():
    module = _load_module()
    selection_rows, linkage, rows, progress, config = _inputs()
    tampered = deepcopy(selection_rows)
    tampered[0]["question"] = "篡改后的问题"

    with pytest.raises(ValueError, match="selection_row_sha256"):
        _build(module, tampered, linkage, rows, progress, config)


def test_fail_closed_on_incomplete_progress_duplicate_ids_and_question_drift():
    module = _load_module()
    selection_rows, linkage, rows, progress, config = _inputs()

    incomplete = deepcopy(progress)
    incomplete["pending_count"] = 1
    incomplete["status"] = "in_progress"
    with pytest.raises(ValueError, match="第二轮尚未完整完成"):
        _build(module, selection_rows, linkage, rows, incomplete, config)

    duplicate = deepcopy(linkage)
    duplicate["records"].append(deepcopy(duplicate["records"][0]))
    duplicate["record_count"] = 3
    with pytest.raises(ValueError, match="重复 pass2_item_id"):
        _build(module, selection_rows, duplicate, rows, progress, config)

    drifted = deepcopy(rows)
    drifted[0]["pass2_final_question"] = "发生漂移的问题"
    with pytest.raises(ValueError, match="问题文本漂移"):
        _build(module, selection_rows, linkage, drifted, progress, config)


def test_invalid_json_list_fails_closed():
    module = _load_module()
    selection_rows, linkage, rows, progress, config = _inputs()
    invalid = deepcopy(rows)
    invalid[0]["pass2_required_claims"] = "not-json"

    with pytest.raises(ValueError, match="pass2_required_claims 必须是 JSON 数组"):
        _build(module, selection_rows, linkage, invalid, progress, config)


def test_resolution_is_deterministic_and_does_not_promote_or_freeze():
    module = _load_module()
    selection_rows, linkage, rows, progress, config = _inputs()

    first = _build(module, selection_rows, linkage, rows, progress, config)
    second = _build(module, selection_rows, linkage, rows, progress, config)

    assert first == second
    assert first["summary"]["gold_promotion_performed"] is False
    assert first["summary"]["freeze_performed"] is False
    assert first["summary"]["usage"] == {
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
    }
    for field in module.RESOLUTION_EMPTY_FIELDS:
        assert first["resolution_queue"][0][field] == ""
