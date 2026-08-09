from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "benchmark_resolution_confirmation.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "confirmation module must exist"
    spec = importlib.util.spec_from_file_location(
        "benchmark_resolution_confirmation", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _queue_row(order: int, candidate_id: str) -> dict[str, str]:
    return {
        "resolution_order": str(order),
        "resolution_version": "benchmark-resolution-v0.1",
        "dataset_version": "benchmark-v1.0-pass2-pending",
        "kb_version": "kb-medium-v1-2026-07-13",
        "candidate_id": candidate_id,
        "source_row_sha256": f"source-row-{order}",
        "source_id": f"SRC-{order}",
        "source_title": f"Source {order}",
        "source_filename": f"source-{order}.pdf",
        "source_sha256": f"source-file-{order}",
        "page_number": str(order),
        "evidence_anchor_ids": f"[\"anchor-{order}\"]",
        "question": f"Question {order}?",
        "anchor_text_span": f"Evidence {order}",
        "policy_rule_ids": "[]",
        "pass1_expected_decision": "review_required",
        "pass1_current_kb_support": "partial_support",
        "pass1_review_reason": "Pass 1 reason",
        "pass2_expected_decision": "insufficient_evidence",
        "pass2_current_kb_support": "unsupported",
        "pass2_gold_evidence_status": "page_span_located",
        "pass2_review_reason": "Pass 2 reason",
        "pass2_missing_evidence_type": "direct evidence",
        "resolution_reviewer_id": "",
        "resolution_annotator_role": "",
        "resolution_reviewed_at": "",
        "resolution_status": "",
        "resolution_final_decision": "",
        "resolution_final_kb_support": "",
        "resolution_final_gold_evidence_status": "",
        "resolution_reason": "",
    }


def _draft_row(
    order: int,
    candidate_id: str,
    *,
    confidence: str,
) -> dict[str, str]:
    row = _queue_row(order, candidate_id)
    return {
        "draft_order": str(order),
        "draft_version": "annotation-resolution-assistant-draft-v0.1",
        "decision_semantics_version": "semantics-v1",
        "dataset_version": row["dataset_version"],
        "kb_version": row["kb_version"],
        "candidate_id": candidate_id,
        "source_row_sha256": row["source_row_sha256"],
        "source_id": row["source_id"],
        "source_title": row["source_title"],
        "source_filename": row["source_filename"],
        "source_sha256": row["source_sha256"],
        "page_number": row["page_number"],
        "evidence_anchor_ids": row["evidence_anchor_ids"],
        "question": row["question"],
        "anchor_text_span": row["anchor_text_span"],
        "policy_rule_ids": row["policy_rule_ids"],
        "pass1_expected_decision": row["pass1_expected_decision"],
        "pass1_current_kb_support": row["pass1_current_kb_support"],
        "pass1_review_reason": row["pass1_review_reason"],
        "pass2_expected_decision": row["pass2_expected_decision"],
        "pass2_current_kb_support": row["pass2_current_kb_support"],
        "pass2_review_reason": row["pass2_review_reason"],
        "pass2_missing_evidence_type": row["pass2_missing_evidence_type"],
        "assistant_system": "Codex",
        "assistant_recommended_decision": "review_required",
        "assistant_recommended_kb_support": "partial_support",
        "assistant_recommended_gold_evidence_status": "page_span_located",
        "assistant_rationale": "Assistant rationale",
        "assistant_confidence": confidence,
        "assistant_confidence_score": "0.9" if confidence == "high" else "0.7",
        "draft_status": "pending_author_confirmation",
        "author_confirmed_decision": "",
        "author_confirmation_reason": "",
        "author_confirmed_at": "",
    }


def _config() -> dict:
    return {
        "config_version": "benchmark-resolution-confirmation-config-v0.1",
        "review_version": "annotation-resolution-author-review-v0.1",
        "resolved_version": "benchmark-resolution-resolved-v0.1",
        "dataset_version": "benchmark-v1.0-pass2-pending",
        "kb_version": "kb-medium-v1-2026-07-13",
        "expected_resolution_count": 2,
        "expected_resolution_queue_sha256": "queue-hash",
        "expected_assistant_draft_sha256": "draft-hash",
        "allowed_decisions": [
            "answer",
            "review_required",
            "insufficient_evidence",
            "boundary_refusal",
        ],
        "allowed_kb_support": ["supported", "partial_support", "unsupported"],
        "allowed_gold_evidence_status": ["page_span_located", "policy_rule"],
        "allowed_confidence": ["high", "medium"],
        "required_confirmation_status": "confirmed",
        "required_annotator_role": "author",
        "external_model_calls": 0,
    }


def _inputs():
    queue = [_queue_row(1, "C-1"), _queue_row(2, "C-2")]
    drafts = [
        _draft_row(1, "C-1", confidence="medium"),
        _draft_row(2, "C-2", confidence="high"),
    ]
    return queue, drafts, _config()


def _build(module, queue, drafts, config):
    return module.build_author_review_pack(
        queue,
        drafts,
        config,
        resolution_queue_sha256="queue-hash",
        assistant_draft_sha256="draft-hash",
    )


def _confirm(review_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    confirmed = deepcopy(review_rows)
    for row in confirmed:
        row["author_reviewer_id"] = "author-001"
        row["author_annotator_role"] = "author"
        row["author_confirmation_status"] = "confirmed"
        row["author_final_decision"] = row["assistant_recommended_decision"]
        row["author_final_kb_support"] = row["assistant_recommended_kb_support"]
        row["author_final_gold_evidence_status"] = row[
            "assistant_recommended_gold_evidence_status"
        ]
        row["author_reason"] = "Author independently confirmed the evidence boundary."
        row["author_reviewed_at"] = "2026-08-09T12:00:00+08:00"
    return confirmed


def test_review_pack_groups_high_before_medium_and_leaves_author_fields_blank():
    module = _load_module()
    queue, drafts, config = _inputs()

    artifacts = _build(module, queue, drafts, config)
    rows = artifacts["review_rows"]

    assert [row["candidate_id"] for row in rows] == ["C-2", "C-1"]
    assert [row["confidence_group"] for row in rows] == ["high", "medium"]
    assert all(not row[field] for row in rows for field in module.AUTHOR_FIELDS)
    assert artifacts["summary"]["status"] == "pending_author_confirmation"
    assert artifacts["summary"]["usage"] == module.zero_usage()


def test_review_pack_fails_closed_on_hash_or_candidate_mismatch():
    module = _load_module()
    queue, drafts, config = _inputs()

    with pytest.raises(ValueError, match="hash"):
        module.build_author_review_pack(
            queue,
            drafts,
            config,
            resolution_queue_sha256="changed",
            assistant_draft_sha256="draft-hash",
        )

    mismatched = deepcopy(drafts)
    mismatched[0]["question"] = "Drifted question"
    with pytest.raises(ValueError, match="C-1"):
        _build(module, queue, mismatched, config)


def test_review_pack_rejects_prefilled_author_confirmation_in_assistant_draft():
    module = _load_module()
    queue, drafts, config = _inputs()
    drafts[0]["author_confirmed_decision"] = "review_required"

    with pytest.raises(ValueError, match="AI"):
        _build(module, queue, drafts, config)


def test_apply_fails_closed_until_every_row_is_explicitly_confirmed():
    module = _load_module()
    queue, drafts, config = _inputs()
    review_rows = _build(module, queue, drafts, config)["review_rows"]
    partial = _confirm(review_rows)
    partial[1]["author_confirmation_status"] = ""

    with pytest.raises(ValueError, match="all 2"):
        module.apply_author_confirmations(
            queue,
            drafts,
            partial,
            config,
            resolution_queue_sha256="queue-hash",
            assistant_draft_sha256="draft-hash",
        )


def test_apply_rejects_invalid_enum_and_immutable_field_drift():
    module = _load_module()
    queue, drafts, config = _inputs()
    review_rows = _confirm(_build(module, queue, drafts, config)["review_rows"])

    invalid = deepcopy(review_rows)
    invalid[0]["author_final_decision"] = "force_answer"
    with pytest.raises(ValueError, match="author_final_decision"):
        module.apply_author_confirmations(
            queue,
            drafts,
            invalid,
            config,
            resolution_queue_sha256="queue-hash",
            assistant_draft_sha256="draft-hash",
        )

    drifted = deepcopy(review_rows)
    drifted[0]["question"] = "Changed after review pack generation"
    with pytest.raises(ValueError, match="immutable"):
        module.apply_author_confirmations(
            queue,
            drafts,
            drifted,
            config,
            resolution_queue_sha256="queue-hash",
            assistant_draft_sha256="draft-hash",
        )


def test_apply_creates_new_resolved_rows_without_mutating_sources():
    module = _load_module()
    queue, drafts, config = _inputs()
    original_queue = deepcopy(queue)
    original_drafts = deepcopy(drafts)
    review_rows = _confirm(_build(module, queue, drafts, config)["review_rows"])

    first = module.apply_author_confirmations(
        queue,
        drafts,
        review_rows,
        config,
        resolution_queue_sha256="queue-hash",
        assistant_draft_sha256="draft-hash",
    )
    second = module.apply_author_confirmations(
        queue,
        drafts,
        review_rows,
        config,
        resolution_queue_sha256="queue-hash",
        assistant_draft_sha256="draft-hash",
    )

    assert queue == original_queue
    assert drafts == original_drafts
    assert first == second
    assert first["summary"]["status"] == "author_resolution_complete"
    assert first["summary"]["resolved_count"] == 2
    assert all(row["resolution_status"] == "resolved" for row in first["resolved_rows"])
    assert all(row["resolution_annotator_role"] == "author" for row in first["resolved_rows"])

