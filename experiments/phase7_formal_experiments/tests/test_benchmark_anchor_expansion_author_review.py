from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "benchmark_anchor_expansion_author_review.py"
)
FORMAL_CONFIG_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "configs"
    / "benchmark_anchor_expansion_author_review_v0_2.json"
)


def _load_module():
    assert MODULE_PATH.exists(), "Anchor expansion author-review module is missing"
    spec = importlib.util.spec_from_file_location(
        "benchmark_anchor_expansion_author_review",
        MODULE_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assistant_row(index: int, *, outcome: str = "accepted_draft") -> dict[str, str]:
    evidence = f"儿童用药证据片段 {index}：应依据原始指南核对剂量、频次、给药途径和适用人群。"
    row = {
        "candidate_id": f"CAND-{index:03d}",
        "source_id": f"SRC-{(index % 2) + 1:03d}",
        "source_title": "测试儿科用药指南",
        "source_filename": "test-guideline.pdf",
        "source_sha256": "source-sha256",
        "page_number": str(index),
        "block_type": "text",
        "granularity": "512",
        "candidate_text": evidence,
        "context_candidate_id": f"CAND-{index:03d}",
        "context_text": evidence,
        "matched_topics": '["dose", "frequency"]',
        "matched_terms": '{"dose": ["剂量"], "frequency": ["频次"]}',
        "review_status": "pending_author_review",
        "dev50_overlap_anchor_ids": "[]",
        "selection_rank_within_source": "1",
        "parser_version": "parser-v1",
        "chunker_version": "chunker-v1",
        "candidate_config_version": "candidate-v1",
        "review_config_version": "benchmark-anchor-expansion-v0.2",
        "reviewer_id": "",
        "author_reviewed_at": "",
        "author_review_outcome": "",
        "author_review_reason": "",
        "verified_text_span": "",
        "supported_claim_types": "",
        "evidence_scope": "",
        "age_scope": "",
        "applicability_conditions": "",
        "scope_check": "",
        "assistant_review_outcome": outcome,
        "assistant_review_reason": "AI 草稿仅供作者逐条核验。",
        "assistant_verified_text_span": evidence if outcome == "accepted_draft" else "",
        "assistant_supported_claim_types": '["dose", "frequency"]' if outcome == "accepted_draft" else "",
        "assistant_evidence_scope": "儿科剂量与频次核对" if outcome == "accepted_draft" else "",
        "assistant_age_scope": "pediatric_unspecified" if outcome == "accepted_draft" else "",
        "assistant_applicability_conditions": "仅限原指南所述场景" if outcome == "accepted_draft" else "",
        "assistant_scope_check": "within_can_support" if outcome == "accepted_draft" else "",
    }
    return row


def _parent_row(assistant_row: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in assistant_row.items()
        if not key.startswith("assistant_")
    }


def _config(expected_count: int = 4) -> dict:
    return {
        "config_version": "benchmark-anchor-expansion-author-review-config-v0.2",
        "review_version": "benchmark-anchor-expansion-author-review-v0.2",
        "dataset_version": "benchmark-v1.0-pre-freeze",
        "kb_version": "KB-medium-v1",
        "expected_candidate_count": expected_count,
        "batch_size": 2,
        "expected_parent_queue_sha256": "parent-hash",
        "expected_assistant_draft_sha256": "draft-hash",
        "min_verified_text_chars": 10,
        "allowed_author_outcomes": ["accepted", "rejected"],
        "required_scope_check": "within_can_support",
    }


def _inputs(count: int = 4):
    drafts = [_assistant_row(index) for index in range(1, count + 1)]
    parents = [_parent_row(row) for row in drafts]
    return parents, drafts, _config(count)


def _confirm(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    confirmed = deepcopy(rows)
    for row in confirmed:
        row["reviewer_id"] = "author-001"
        row["author_reviewed_at"] = "2026-08-10T12:00:00+08:00"
        row["author_review_outcome"] = "accepted"
        row["author_review_reason"] = "作者已逐页核对并确认该窄范围证据可追溯。"
        row["verified_text_span"] = row["assistant_verified_text_span"]
        row["supported_claim_types"] = row["assistant_supported_claim_types"]
        row["evidence_scope"] = row["assistant_evidence_scope"]
        row["age_scope"] = row["assistant_age_scope"]
        row["applicability_conditions"] = row[
            "assistant_applicability_conditions"
        ]
        row["scope_check"] = "within_can_support"
    return confirmed


def test_prepare_author_batches_is_deterministic_and_keeps_author_fields_blank():
    module = _load_module()
    parents, drafts, config = _inputs()

    first = module.prepare_author_batches(
        parents,
        drafts,
        config,
        parent_queue_sha256="parent-hash",
        assistant_draft_sha256="draft-hash",
    )
    second = module.prepare_author_batches(
        parents,
        drafts,
        config,
        parent_queue_sha256="parent-hash",
        assistant_draft_sha256="draft-hash",
    )

    assert first == second
    assert [len(batch["rows"]) for batch in first] == [2, 2]
    assert all(
        not row[field]
        for batch in first
        for row in batch["rows"]
        for field in module.AUTHOR_REVIEW_FIELDS
    )


def test_prepare_fails_closed_on_hash_or_assistant_draft_drift():
    module = _load_module()
    parents, drafts, config = _inputs()

    with pytest.raises(ValueError, match="hash"):
        module.prepare_author_batches(
            parents,
            drafts,
            config,
            parent_queue_sha256="changed",
            assistant_draft_sha256="draft-hash",
        )

    drifted = deepcopy(drafts)
    drifted[0]["assistant_review_reason"] = "被修改的 AI 草稿"
    prepared = module.prepare_author_batches(
        parents,
        drafts,
        config,
        parent_queue_sha256="parent-hash",
        assistant_draft_sha256="draft-hash",
    )
    author_rows = _confirm([row for batch in prepared for row in batch["rows"]])
    with pytest.raises(ValueError, match="immutable"):
        module.validate_author_batch(
            author_rows,
            drifted,
            config,
            parent_queue_sha256="parent-hash",
            assistant_draft_sha256="draft-hash",
        )


def test_accepted_author_review_requires_traceable_complete_evidence():
    module = _load_module()
    parents, drafts, config = _inputs()
    prepared = module.prepare_author_batches(
        parents,
        drafts,
        config,
        parent_queue_sha256="parent-hash",
        assistant_draft_sha256="draft-hash",
    )
    author_rows = _confirm([row for batch in prepared for row in batch["rows"]])
    author_rows[0]["verified_text_span"] = "原始候选和上下文中不存在的证据"

    with pytest.raises(ValueError, match="不可追溯"):
        module.validate_author_batch(
            author_rows,
            drafts,
            config,
            parent_queue_sha256="parent-hash",
            assistant_draft_sha256="draft-hash",
        )


def test_rejected_author_review_requires_explicit_reason():
    module = _load_module()
    parents, drafts, config = _inputs()
    prepared = module.prepare_author_batches(
        parents,
        drafts,
        config,
        parent_queue_sha256="parent-hash",
        assistant_draft_sha256="draft-hash",
    )
    author_rows = _confirm([row for batch in prepared for row in batch["rows"]])
    author_rows[0]["author_review_outcome"] = "rejected"
    author_rows[0]["author_review_reason"] = ""

    with pytest.raises(ValueError, match="拒绝理由"):
        module.validate_author_batch(
            author_rows,
            drafts,
            config,
            parent_queue_sha256="parent-hash",
            assistant_draft_sha256="draft-hash",
        )


def test_finalize_requires_exactly_once_full_coverage_and_never_promotes():
    module = _load_module()
    parents, drafts, config = _inputs()
    prepared = module.prepare_author_batches(
        parents,
        drafts,
        config,
        parent_queue_sha256="parent-hash",
        assistant_draft_sha256="draft-hash",
    )
    author_rows = _confirm([row for batch in prepared for row in batch["rows"]])

    with pytest.raises(ValueError, match="all 4"):
        module.finalize_author_reviews(
            author_rows[:-1],
            parents,
            drafts,
            config,
            parent_queue_sha256="parent-hash",
            assistant_draft_sha256="draft-hash",
        )

    result = module.finalize_author_reviews(
        author_rows,
        parents,
        drafts,
        config,
        parent_queue_sha256="parent-hash",
        assistant_draft_sha256="draft-hash",
    )

    assert result["summary"]["status"] == "author_review_complete"
    assert result["summary"]["author_reviewed_count"] == 4
    assert result["summary"]["anchor_promotion_performed"] is False
    assert result["summary"]["usage"] == module.zero_usage()
    assert "gold" in result["summary"]["medical_boundary"].lower()


def test_formal_config_locks_the_real_58_candidate_inputs():
    module = _load_module()
    config = module.load_config(FORMAL_CONFIG_PATH)

    assert config["expected_candidate_count"] == 58
    assert config["batch_size"] == 10
    assert config["expected_parent_queue_sha256"] == (
        "423198aefbb6110fb988f14317493c4c7a55802e1f50c7231947e26224bd9b1b"
    )
    assert config["expected_assistant_draft_sha256"] == (
        "62826dfa7f01c2ec039ad659a743ce3868e1a0a43b980ee1d9226d2ad3924a3a"
    )
