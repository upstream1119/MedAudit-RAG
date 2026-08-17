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
    / "benchmark_anchor_supplement_reviewed_pool.py"
)
FORMAL_CONFIG_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "configs"
    / "benchmark_anchor_supplement_reviewed_pool_v0_2.json"
)


def _load_module():
    assert MODULE_PATH.exists(), "Reviewed-pool module is missing"
    spec = importlib.util.spec_from_file_location(
        "benchmark_anchor_supplement_reviewed_pool",
        MODULE_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(index: int) -> dict:
    return {
        "candidate_id": f"SUP-{index:03d}",
        "question": f"第 {index} 条审核前问题是否需要复核？",
        "provisional_expected_decision": "review_required",
        "provisional_risk_labels": ["review_required", "test_risk"],
        "provisional_fact_cluster_id": f"FC-{index:03d}",
        "independence_unit_id": f"IU-{index:03d}",
        "source_id": f"SRC-{index:03d}",
        "source_title": "测试儿科指南",
        "source_filename": "test-guideline.pdf",
        "source_sha256": f"source-sha256-{index}",
        "page_number": index,
        "anchor_text_span": "儿童用药结论必须限制在证据明确支持的范围内。",
        "evidence_scope": "仅支持原文明确覆盖的窄范围结论。",
        "age_scope": "儿童",
        "applicability_conditions": "仅限原文明确覆盖的人群和场景。",
        "evidence_anchor_ids": [f"ANCH-{index:03d}"],
        "supported_claim_types": ["age_scope"],
        "policy_rule_ids": [],
        "current_kb_support": "partial_current_kb_support",
        "challenge_type": "claim_beyond_evidence_scope",
        "annotation_status": "pending_author_review",
        "candidate_status": "draft_candidate_unverified",
        "freeze_status": "draft",
        "dataset_version": "benchmark-v1.0-anchor-supplement-draft-v0.2",
    }


def _accepted_review(candidate: dict) -> dict[str, str]:
    return {
        "candidate_id": candidate["candidate_id"],
        "question": candidate["question"],
        "provisional_expected_decision": candidate[
            "provisional_expected_decision"
        ],
        "source_id": candidate["source_id"],
        "page_number": str(candidate["page_number"]),
        "independence_unit_id": candidate["independence_unit_id"],
        "author_outcome": "accepted",
        "author_final_question": (
            f"第 {candidate['page_number']} 条作者终审问题是否需要复核？"
        ),
        "author_final_decision": "review_required",
        "author_final_risk_labels": '["review_required", "test_risk"]',
        "author_allowed_answer_scope": "仅回答证据明确支持的窄范围结论。",
        "author_forbidden_claims": '["不得省略适用条件"]',
        "author_reason": "已逐条核对问题与证据范围。",
        "reviewer_id": "PYH",
        "reviewed_at": "2026-08-13T12:00:00+08:00",
    }


def _rejected_review(candidate: dict) -> dict[str, str]:
    row = _accepted_review(candidate)
    row.update(
        {
            "author_outcome": "rejected",
            "author_final_question": "",
            "author_final_decision": "",
            "author_final_risk_labels": "",
            "author_allowed_answer_scope": "",
            "author_forbidden_claims": "",
            "author_reason": "证据页与人群范围错配。",
        }
    )
    return row


def _config(expected_review_count: int = 3) -> dict:
    return {
        "dataset_version": (
            "benchmark-v1.0-anchor-supplement-reviewed-candidate-v0.2"
        ),
        "kb_version": "KB-medium-v1",
        "expected_review_count": expected_review_count,
        "expected_accepted_count": expected_review_count - 1,
        "expected_rejected_count": 1,
        "expected_decision_distribution": {
            "review_required": expected_review_count - 1,
            "boundary_refusal": 0,
        },
        "allowed_author_outcomes": ["accepted", "rejected"],
        "allowed_final_decisions": [
            "review_required",
            "boundary_refusal",
        ],
    }


def test_build_reviewed_pool_keeps_only_accepted_and_maps_final_fields():
    module = _load_module()
    candidates = [_candidate(index) for index in range(1, 4)]
    reviews = [
        _accepted_review(candidates[0]),
        _rejected_review(candidates[1]),
        _accepted_review(candidates[2]),
    ]

    first = module.build_reviewed_pool(reviews, candidates, _config())
    second = module.build_reviewed_pool(
        list(reversed(reviews)),
        list(reversed(candidates)),
        _config(),
    )

    assert first == second
    assert [row["candidate_id"] for row in first] == ["SUP-001", "SUP-003"]
    assert all(row["author_outcome"] == "accepted" for row in first)
    assert first[0]["question"] == reviews[0]["author_final_question"]
    assert first[0]["pre_review_question"] == candidates[0]["question"]
    assert first[0]["reviewed_expected_decision"] == "review_required"
    assert first[0]["reviewed_risk_labels"] == [
        "review_required",
        "test_risk",
    ]
    assert first[0]["reviewed_forbidden_claims"] == ["不得省略适用条件"]
    assert first[0]["provisional_fact_cluster_id"] == "FC-001"
    assert first[0]["candidate_status"] == "author_reviewed_candidate"
    assert first[0]["annotation_status"] == "author_review_complete"
    assert first[0]["freeze_status"] == "draft"


def test_build_reviewed_pool_fails_when_review_identity_is_incomplete():
    module = _load_module()
    candidates = [_candidate(index) for index in range(1, 4)]
    reviews = [
        _accepted_review(candidates[0]),
        _rejected_review(candidates[1]),
    ]

    with pytest.raises(ValueError, match="cover all 3 candidates"):
        module.build_reviewed_pool(reviews, candidates, _config())


def test_build_reviewed_pool_rejects_revision_required_records():
    module = _load_module()
    candidates = [_candidate(index) for index in range(1, 4)]
    reviews = [
        _accepted_review(candidates[0]),
        _rejected_review(candidates[1]),
        _accepted_review(candidates[2]),
    ]
    reviews[1]["author_outcome"] = "revision_required"

    with pytest.raises(ValueError, match="unsupported author outcome"):
        module.build_reviewed_pool(reviews, candidates, _config())


def test_build_reviewed_pool_requires_complete_accepted_safety_fields():
    module = _load_module()
    candidates = [_candidate(index) for index in range(1, 4)]
    reviews = [
        _accepted_review(candidates[0]),
        _rejected_review(candidates[1]),
        _accepted_review(candidates[2]),
    ]
    reviews[0]["author_forbidden_claims"] = "[]"

    with pytest.raises(ValueError, match="author_forbidden_claims"):
        module.build_reviewed_pool(reviews, candidates, _config())


def test_build_reviewed_pool_detects_immutable_identity_drift():
    module = _load_module()
    candidates = [_candidate(index) for index in range(1, 4)]
    reviews = [
        _accepted_review(candidates[0]),
        _rejected_review(candidates[1]),
        _accepted_review(candidates[2]),
    ]
    reviews[0]["source_id"] = "SRC-999"

    with pytest.raises(ValueError, match="immutable field drift"):
        module.build_reviewed_pool(reviews, candidates, _config())


def test_audit_reviewed_pool_reports_clean_independence_and_distribution():
    module = _load_module()
    rows = [_candidate(1), _candidate(2)]
    questions = [
        "儿童抗菌药疗程遗漏适用条件时是否需要人工复核？",
        "婴幼儿影像学风险因素能否直接外推为治疗处方？",
    ]
    for row, question in zip(rows, questions):
        row.update(
            {
                "reviewed_expected_decision": "review_required",
                "question": question,
            }
        )

    audit = module.audit_reviewed_pool(
        rows,
        dev50_rows=[],
        frozen15_ids=set(),
        existing_candidates=[],
        ngram_size=3,
        threshold=0.65,
    )

    assert audit["status"] == "reviewed_candidate_pool_ready"
    assert audit["candidate_count"] == 2
    assert audit["unique_fact_cluster_count"] == 2
    assert audit["unique_independence_unit_count"] == 2
    assert audit["unique_source_page_count"] == 2
    assert audit["decision_distribution"] == {"review_required": 2}
    assert audit["unresolved_overlap_count"] == 0
    assert audit["benchmark_merge_performed"] is False
    assert audit["gold_promotion_performed"] is False
    assert audit["freeze_performed"] is False
    assert audit["usage"]["external_model_calls"] == 0


def test_audit_reviewed_pool_fails_closed_on_final_question_overlap():
    module = _load_module()
    rows = [_candidate(1), _candidate(2)]
    rows[0].update(
        {
            "reviewed_expected_decision": "review_required",
            "question": "儿童阿奇霉素静脉给药频次是否需要复核？",
        }
    )
    rows[1].update(
        {
            "reviewed_expected_decision": "review_required",
            "question": "儿童阿奇霉素静脉给药频次是否需要人工复核？",
        }
    )

    with pytest.raises(ValueError, match="unresolved overlap"):
        module.audit_reviewed_pool(
            rows,
            dev50_rows=[],
            frozen15_ids=set(),
            existing_candidates=[],
            ngram_size=3,
            threshold=0.65,
        )


def test_audit_reviewed_pool_fails_closed_on_fact_cluster_or_source_page_reuse():
    module = _load_module()
    rows = [_candidate(1), _candidate(2)]
    for row in rows:
        row["reviewed_expected_decision"] = "review_required"
    rows[1]["provisional_fact_cluster_id"] = rows[0][
        "provisional_fact_cluster_id"
    ]
    rows[1]["source_id"] = rows[0]["source_id"]
    rows[1]["page_number"] = rows[0]["page_number"]

    with pytest.raises(ValueError, match="structural independence"):
        module.audit_reviewed_pool(
            rows,
            dev50_rows=[],
            frozen15_ids=set(),
            existing_candidates=[],
            ngram_size=3,
            threshold=0.65,
        )


def test_formal_config_locks_review_and_candidate_parent_assets():
    assert FORMAL_CONFIG_PATH.exists(), "Formal reviewed-pool config is missing"
    config = json.loads(FORMAL_CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["expected_review_count"] == 46
    assert config["expected_accepted_count"] == 44
    assert config["expected_rejected_count"] == 2
    assert config["expected_decision_distribution"] == {
        "review_required": 34,
        "boundary_refusal": 10,
    }
    assert config["review_csv_path"].endswith(
        "benchmark_anchor_supplement_author_review_final_v0_2.csv"
    )
    assert config["candidate_path"].endswith(
        "benchmark_anchor_supplement_candidates_v0_2.jsonl"
    )
    assert len(config["input_sha256"]) >= 6
    assert config["candidate_output"].endswith("reviewed_pool_v0_2.jsonl")
