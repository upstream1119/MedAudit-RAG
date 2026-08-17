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
    / "benchmark_anchor_expansion_promotion.py"
)

ANCHOR_FIELDS = {
    "age_scope",
    "anchor_id",
    "applicability_conditions",
    "author_review_reason",
    "author_reviewed_at",
    "candidate_config_version",
    "candidate_id",
    "chunker_version",
    "evidence_scope",
    "page_number",
    "parser_version",
    "review_config_version",
    "reviewer_id",
    "scope_check",
    "source_filename",
    "source_id",
    "source_sha256",
    "source_title",
    "supported_claim_types",
    "text_span",
    "verification_status",
}


def _load_module():
    assert MODULE_PATH.exists(), "Anchor expansion promotion module is missing"
    spec = importlib.util.spec_from_file_location(
        "benchmark_anchor_expansion_promotion",
        MODULE_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _author_row(
    candidate_id: str,
    *,
    source_id: str = "SRC-001",
    page_number: int = 2,
    outcome: str = "accepted",
) -> dict[str, str]:
    evidence = (
        f"候选 {candidate_id} 的可追溯儿科用药证据片段，"
        "仅支持原指南所列剂量、频次、途径和适用人群。"
    )
    accepted = outcome == "accepted"
    return {
        "candidate_id": candidate_id,
        "source_id": source_id,
        "source_title": "测试儿科用药指南",
        "source_filename": "test-guideline.pdf",
        "source_sha256": "source-sha256",
        "page_number": str(page_number),
        "block_type": "text",
        "granularity": "512",
        "candidate_text": evidence,
        "context_candidate_id": candidate_id,
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
        "reviewer_id": "PYH",
        "author_reviewed_at": "2026-08-10T19:13:14+08:00",
        "author_review_outcome": outcome,
        "author_review_reason": (
            "作者已逐页核对并确认该窄范围证据可追溯。"
            if accepted
            else "原页不能支持该候选主张，排除出正式锚点池。"
        ),
        "verified_text_span": evidence if accepted else "",
        "supported_claim_types": '["dose", "frequency"]' if accepted else "",
        "evidence_scope": "儿科剂量与频次核对" if accepted else "",
        "age_scope": "pediatric_unspecified" if accepted else "",
        "applicability_conditions": "仅限原指南所述场景" if accepted else "",
        "scope_check": "within_can_support" if accepted else "outside_can_support",
        "assistant_review_outcome": "accepted_draft",
        "assistant_review_reason": "AI 草稿仅供作者逐条核验。",
        "assistant_verified_text_span": evidence,
        "assistant_supported_claim_types": '["dose", "frequency"]',
        "assistant_evidence_scope": "儿科剂量与频次核对",
        "assistant_age_scope": "pediatric_unspecified",
        "assistant_applicability_conditions": "仅限原指南所述场景",
        "assistant_scope_check": "within_can_support",
    }


def _parent_row(author_row: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in author_row.items()
        if not key.startswith("assistant_")
        and key
        not in {
            "reviewer_id",
            "author_reviewed_at",
            "author_review_outcome",
            "author_review_reason",
            "verified_text_span",
            "supported_claim_types",
            "evidence_scope",
            "age_scope",
            "applicability_conditions",
            "scope_check",
        }
    } | {
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
    }


def _existing_anchor() -> dict:
    return {
        "age_scope": "pediatric_unspecified",
        "anchor_id": "ANCH-existing",
        "applicability_conditions": "仅限既有证据场景",
        "author_review_reason": "既有作者核验锚点。",
        "author_reviewed_at": "2026-07-29T12:00:00+08:00",
        "candidate_config_version": "candidate-v0.1",
        "candidate_id": "CAND-existing",
        "chunker_version": "chunker-v1",
        "evidence_scope": "既有证据范围",
        "page_number": 1,
        "parser_version": "parser-v1",
        "review_config_version": "benchmark-anchor-review-v0.1",
        "reviewer_id": "author-pyh",
        "scope_check": "within_can_support",
        "source_filename": "existing-guideline.pdf",
        "source_id": "SRC-002",
        "source_sha256": "existing-source-sha256",
        "source_title": "既有儿科指南",
        "supported_claim_types": ["treatment_principle"],
        "text_span": "既有锚点的可追溯证据片段。",
        "verification_status": "author_verified_anchor",
    }


def _config() -> dict:
    return {
        "config_version": "benchmark-anchor-expansion-promotion-config-v0.2",
        "promotion_version": "benchmark-anchor-expansion-promotion-v0.2",
        "dataset_version": "benchmark-v1.0-pre-freeze",
        "kb_version": "KB-medium-v1",
        "expected_review_count": 2,
        "expected_accepted_count": 1,
        "expected_rejected_count": 1,
        "expected_existing_anchor_count": 1,
        "expected_expansion_anchor_count": 1,
        "expected_merged_anchor_count": 2,
        "expected_parent_queue_sha256": "parent-hash",
        "expected_author_review_sha256": "author-hash",
        "expected_author_audit_sha256": "author-audit-hash",
        "expected_coverage_sha256": "coverage-hash",
        "expected_existing_pool_sha256": "pool-hash",
        "expected_dev50_registry_sha256": "dev50-hash",
        "min_verified_text_chars": 20,
        "required_scope_check": "within_can_support",
    }


def _inputs():
    accepted = _author_row("CAND-accepted", page_number=2)
    rejected = _author_row("CAND-rejected", page_number=3, outcome="rejected")
    authors = [accepted, rejected]
    parents = [_parent_row(row) for row in authors]
    coverage = [
        {
            "source_id": "SRC-001",
            "title": "测试儿科用药指南",
            "filename": "test-guideline.pdf",
            "actual_sha256": "source-sha256",
            "recorded_sha256": "source-sha256",
            "can_support": ["儿科剂量与频次核对"],
            "cannot_support": ["个体处方"],
            "included_in_kb": True,
        }
    ]
    artifact_hashes = {
        "parent_queue_sha256": "parent-hash",
        "author_review_sha256": "author-hash",
        "author_audit_sha256": "author-audit-hash",
        "coverage_sha256": "coverage-hash",
        "existing_pool_sha256": "pool-hash",
        "dev50_registry_sha256": "dev50-hash",
    }
    return (
        authors,
        parents,
        coverage,
        [_existing_anchor()],
        {},
        {"SRC-001": 10},
        _config(),
        artifact_hashes,
    )


def _build(module, inputs=None):
    args = inputs or _inputs()
    return module.build_promotion_result(
        author_rows=args[0],
        parent_rows=args[1],
        coverage_rows=args[2],
        existing_anchors=args[3],
        dev50_pairs=args[4],
        page_counts=args[5],
        config=args[6],
        artifact_hashes=args[7],
    )


def test_promotion_is_deterministic_and_excludes_rejected_rows():
    module = _load_module()

    first = _build(module)
    second = _build(module)

    assert first == second
    assert len(first["new_anchors"]) == 1
    assert len(first["merged_anchors"]) == 2
    assert len(first["decisions"]) == 2
    assert first["audit"]["accepted_count"] == 1
    assert first["audit"]["rejected_count"] == 1
    assert first["audit"]["api_calls"] == 0
    assert first["audit"]["input_tokens"] == 0
    assert first["audit"]["output_tokens"] == 0
    assert first["audit"]["estimated_cost"] == 0
    assert set(first["new_anchors"][0]) == ANCHOR_FIELDS
    assert first["new_anchors"][0]["verification_status"] == "author_verified_anchor"
    assert first["decisions"][1]["promotion_decision"] == "excluded_author_rejected"


def test_promotion_fails_closed_on_parent_hash_drift():
    module = _load_module()
    inputs = list(_inputs())
    inputs[7] = dict(inputs[7], parent_queue_sha256="changed")

    with pytest.raises(ValueError, match="hash"):
        _build(module, tuple(inputs))


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda rows: rows[0].__setitem__("scope_check", "outside_can_support"), "scope"),
        (lambda rows: rows[0].__setitem__("verified_text_span", "不可追溯的伪证据片段"), "trace"),
        (lambda rows: rows[0].__setitem__("page_number", "99"), "page"),
        (lambda rows: rows[0].__setitem__("source_sha256", "changed"), "provenance"),
    ],
)
def test_accepted_anchor_requires_valid_scope_text_page_and_provenance(mutator, message):
    module = _load_module()
    inputs = list(_inputs())
    inputs[0] = deepcopy(inputs[0])
    mutator(inputs[0])

    with pytest.raises(ValueError, match=message):
        _build(module, tuple(inputs))


def test_promotion_rejects_dev50_overlap_and_existing_source_page_duplicate():
    module = _load_module()
    inputs = list(_inputs())
    inputs[4] = {("SRC-001", 2): ["DEV-001"]}
    with pytest.raises(ValueError, match="Dev50"):
        _build(module, tuple(inputs))

    inputs = list(_inputs())
    old_anchor = deepcopy(inputs[3][0])
    old_anchor.update({"source_id": "SRC-001", "page_number": 2})
    inputs[3] = [old_anchor]
    with pytest.raises(ValueError, match="duplicate"):
        _build(module, tuple(inputs))


def test_build_does_not_mutate_existing_pool():
    module = _load_module()
    inputs = _inputs()
    before = deepcopy(inputs[3])

    _build(module, inputs)

    assert inputs[3] == before
