from __future__ import annotations

import importlib.util
import json
import csv
from copy import deepcopy
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "benchmark_anchor_supplement_author_review.py"
)
FORMAL_CONFIG_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "configs"
    / "benchmark_anchor_supplement_author_review_v0_2.json"
)


def _load_module():
    assert MODULE_PATH.exists(), "Supplement author-review module is missing"
    spec = importlib.util.spec_from_file_location(
        "benchmark_anchor_supplement_author_review",
        MODULE_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(index: int) -> dict:
    return {
        "candidate_id": f"SUP-{index:03d}",
        "question": f"第 {index} 条儿科用药审核候选问题是否需要人工复核？",
        "provisional_expected_decision": "review_required",
        "challenge_type": "omitted_applicability_condition",
        "provisional_risk_labels": ["review_required", "test_risk"],
        "current_kb_support": "partial_current_kb_support",
        "policy_rule_ids": [],
        "supported_claim_types": ["age_scope"],
        "scope_check": "within_can_support",
        "source_id": f"SRC-{(index % 3) + 1:03d}",
        "source_title": "测试儿科指南",
        "source_filename": "test-guideline.pdf",
        "source_sha256": "source-sha256",
        "page_number": index,
        "anchor_text_span": "儿童用药结论必须限制在指南证据明确支持的适用范围内。",
        "evidence_scope": "仅支持原文明确覆盖的儿科用药审核结论。",
        "age_scope": "儿童",
        "applicability_conditions": "仅限原文明确覆盖的人群与场景。",
        "evidence_anchor_ids": [f"ANCH-{index:03d}"],
        "independence_unit_id": f"IU-{index:03d}",
        "annotation_status": "pending_author_review",
        "freeze_status": "draft",
    }


def _review_row(candidate: dict) -> dict[str, str]:
    return {
        "candidate_id": candidate["candidate_id"],
        "question": candidate["question"],
        "provisional_expected_decision": candidate[
            "provisional_expected_decision"
        ],
        "challenge_type": candidate["challenge_type"],
        "source_id": candidate["source_id"],
        "source_title": candidate["source_title"],
        "page_number": str(candidate["page_number"]),
        "anchor_text_span": candidate["anchor_text_span"],
        "evidence_scope": candidate["evidence_scope"],
        "age_scope": candidate["age_scope"],
        "applicability_conditions": candidate["applicability_conditions"],
        "evidence_anchor_ids": json.dumps(
            candidate["evidence_anchor_ids"], ensure_ascii=False
        ),
        "independence_unit_id": candidate["independence_unit_id"],
        "author_outcome": "",
        "author_final_decision": "",
        "author_reason": "",
        "reviewer_id": "",
        "reviewed_at": "",
    }


def _config(expected_count: int = 46) -> dict:
    return {
        "config_version": "benchmark-anchor-supplement-author-review-config-v0.2",
        "review_version": "benchmark-anchor-supplement-author-review-v0.2",
        "dataset_version": "benchmark-v1.0-anchor-supplement-pre-review",
        "kb_version": "KB-medium-v1",
        "expected_candidate_count": expected_count,
        "batch_sizes": [10, 10, 10, 10, 6],
        "expected_input_sha256": {
            "candidates": "candidate-hash",
            "review_queue": "queue-hash",
            "audit": "audit-hash",
            "summary": "summary-hash",
        },
        "allowed_author_outcomes": [
            "accepted",
            "revision_required",
            "rejected",
        ],
        "allowed_final_decisions": ["review_required", "boundary_refusal"],
    }


def _accept_all(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    accepted = deepcopy(rows)
    for row in accepted:
        row.update(
            {
                "author_outcome": "accepted",
                "author_final_question": row["question"],
                "author_final_decision": row[
                    "provisional_expected_decision"
                ],
                "author_final_risk_labels": '["review_required", "test_risk"]',
                "author_allowed_answer_scope": "仅回答证据明确支持的窄范围结论。",
                "author_forbidden_claims": '["不得省略适用条件"]',
                "author_reason": "已逐条核对问题与证据范围。",
                "reviewer_id": "author-001",
                "reviewed_at": "2026-08-11T12:00:00+08:00",
            }
        )
    return accepted


def test_prepare_creates_five_deterministic_batches_with_blank_author_fields():
    module = _load_module()
    candidates = [_candidate(index) for index in range(1, 47)]
    review_rows = [_review_row(candidate) for candidate in candidates]
    config = _config()
    observed_hashes = dict(config["expected_input_sha256"])

    first = module.prepare_author_batches(
        candidates,
        review_rows,
        config,
        observed_input_sha256=observed_hashes,
    )
    second = module.prepare_author_batches(
        list(reversed(candidates)),
        list(reversed(review_rows)),
        config,
        observed_input_sha256=observed_hashes,
    )

    assert first == second
    assert [len(batch["rows"]) for batch in first] == [10, 10, 10, 10, 6]
    assert all(
        not str(row.get(field, "")).strip()
        for batch in first
        for row in batch["rows"]
        for field in module.AUTHOR_REVIEW_FIELDS
    )


def test_accepted_review_requires_complete_safety_fields():
    module = _load_module()
    assert hasattr(module, "validate_author_batch"), "Validation gate is missing"
    candidates = [_candidate(index) for index in range(1, 3)]
    review_rows = [_review_row(candidate) for candidate in candidates]
    config = _config(expected_count=2)
    config["batch_sizes"] = [2]
    observed_hashes = dict(config["expected_input_sha256"])
    prepared = module.prepare_author_batches(
        candidates,
        review_rows,
        config,
        observed_input_sha256=observed_hashes,
    )[0]["rows"]
    accepted = deepcopy(prepared)
    for row in accepted:
        row.update(
            {
                "author_outcome": "accepted",
                "author_final_question": row["question"],
                "author_final_decision": row[
                    "provisional_expected_decision"
                ],
                "author_final_risk_labels": '["review_required", "test_risk"]',
                "author_allowed_answer_scope": "仅回答证据明确支持的窄范围结论。",
                "author_forbidden_claims": '["不得省略适用条件"]',
                "author_reason": "已逐条核对问题与证据范围。",
                "reviewer_id": "author-001",
                "reviewed_at": "2026-08-11T12:00:00+08:00",
            }
        )
    accepted[0]["author_forbidden_claims"] = ""

    with pytest.raises(ValueError, match="author_forbidden_claims"):
        module.validate_author_batch(
            accepted,
            prepared,
            config,
            observed_input_sha256=observed_hashes,
        )


def test_nonaccepted_reviews_cannot_carry_promotable_fields():
    module = _load_module()
    candidates = [_candidate(1), _candidate(2)]
    review_rows = [_review_row(candidate) for candidate in candidates]
    config = _config(expected_count=2)
    config["batch_sizes"] = [2]
    observed_hashes = dict(config["expected_input_sha256"])
    prepared = module.prepare_author_batches(
        candidates,
        review_rows,
        config,
        observed_input_sha256=observed_hashes,
    )[0]["rows"]
    reviewed = deepcopy(prepared)
    reviewed[0].update(
        {
            "author_outcome": "revision_required",
            "author_final_question": "修改后仍需再次审核的问题草案。",
            "author_reason": "原问题的适用条件表达不完整。",
            "reviewer_id": "author-001",
            "reviewed_at": "2026-08-11T12:00:00+08:00",
        }
    )
    reviewed[1].update(
        {
            "author_outcome": "rejected",
            "author_final_decision": "review_required",
            "author_reason": "问题无法由当前证据锚点支持。",
            "reviewer_id": "author-001",
            "reviewed_at": "2026-08-11T12:00:00+08:00",
        }
    )

    with pytest.raises(ValueError, match="rejected.*可晋升字段"):
        module.validate_author_batch(
            reviewed,
            prepared,
            config,
            observed_input_sha256=observed_hashes,
        )


def test_finalize_requires_exact_full_coverage_and_never_promotes():
    module = _load_module()
    assert hasattr(module, "finalize_author_reviews"), "Finalize gate is missing"
    candidates = [_candidate(index) for index in range(1, 5)]
    review_rows = [_review_row(candidate) for candidate in candidates]
    config = _config(expected_count=4)
    config["batch_sizes"] = [2, 2]
    observed_hashes = dict(config["expected_input_sha256"])
    batches = module.prepare_author_batches(
        candidates,
        review_rows,
        config,
        observed_input_sha256=observed_hashes,
    )
    prepared = [row for batch in batches for row in batch["rows"]]
    accepted = _accept_all(prepared)

    with pytest.raises(ValueError, match="all 4.*exactly once"):
        module.finalize_author_reviews(
            accepted[:-1],
            prepared,
            config,
            observed_input_sha256=observed_hashes,
        )

    result = module.finalize_author_reviews(
        accepted,
        prepared,
        config,
        observed_input_sha256=observed_hashes,
    )

    assert result["summary"]["status"] == "author_review_complete"
    assert result["summary"]["author_reviewed_count"] == 4
    assert result["summary"]["candidate_merge_performed"] is False
    assert result["summary"]["gold_promotion_performed"] is False
    assert result["summary"]["freeze_performed"] is False
    assert result["summary"]["usage"] == module.zero_usage()


def test_prepare_refuses_to_overwrite_any_completed_author_field(tmp_path):
    module = _load_module()
    assert hasattr(
        module, "write_prepared_author_batches"
    ), "Batch writer is missing"
    candidates = [_candidate(index) for index in range(1, 5)]
    review_rows = [_review_row(candidate) for candidate in candidates]
    config = _config(expected_count=4)
    config["batch_sizes"] = [2, 2]
    observed_hashes = dict(config["expected_input_sha256"])
    batches = module.prepare_author_batches(
        candidates,
        review_rows,
        config,
        observed_input_sha256=observed_hashes,
    )
    outputs = module.write_prepared_author_batches(batches, tmp_path)
    batch_path = Path(outputs["batch_dir"]) / "batch_01.csv"
    with batch_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["reviewer_id"] = "author-001"
    with batch_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        module.write_prepared_author_batches(batches, tmp_path)


def test_formal_config_locks_all_46_candidates_and_four_parent_artifacts():
    module = _load_module()
    assert FORMAL_CONFIG_PATH.exists(), "Formal author-review config is missing"
    config = module.load_config(FORMAL_CONFIG_PATH)

    assert config["expected_candidate_count"] == 46
    assert config["batch_sizes"] == [10, 10, 10, 10, 6]
    assert config["expected_input_sha256"] == {
        "candidates": "4083e6c41efbc9dd9ace72b73a783bde6c930310dec0d8e1a0afc767ec26d678",
        "review_queue": "2e25f89295cad29f725636ea08d591a9c239e0475fbbc03c0eee47e01381321d",
        "audit": "663c7f4968bf34e59af63faba59caf760bfc08542b3f2f2c074fb8ea62315ae3",
        "summary": "9ec4a6d36ee07e2fd66a857180eb7fcbdf8bccff971e7f2af34d4bacadb0079c",
    }


def test_prepare_from_formal_config_writes_a_pending_review_package(tmp_path):
    module = _load_module()
    assert hasattr(module, "prepare_from_config"), "Prepare entrypoint is missing"

    outputs = module.prepare_from_config(FORMAL_CONFIG_PATH, output_dir=tmp_path)
    manifest_path = Path(outputs["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "pending_author_review"
    assert manifest["candidate_count"] == 46
    assert manifest["batch_count"] == 5
    assert manifest["author_reviewed_count"] == 0
    assert manifest["candidate_merge_performed"] is False
    assert manifest["gold_promotion_performed"] is False
    assert manifest["freeze_performed"] is False
    assert len(list(Path(outputs["batch_dir"]).glob("batch_*.csv"))) == 5


def test_prepare_includes_auditable_risk_policy_and_source_context():
    module = _load_module()
    candidate = _candidate(1)
    candidate.update(
        {
            "provisional_risk_labels": ["review_required", "age_scope"],
            "current_kb_support": "partial_current_kb_support",
            "policy_rule_ids": ["POLICY-SAFETY-001"],
            "supported_claim_types": ["age_scope"],
            "scope_check": "within_can_support",
            "source_filename": "test-guideline.pdf",
            "source_sha256": "source-sha256",
        }
    )
    config = _config(expected_count=1)
    config["batch_sizes"] = [1]

    row = module.prepare_author_batches(
        [candidate],
        [_review_row(candidate)],
        config,
        observed_input_sha256=dict(config["expected_input_sha256"]),
    )[0]["rows"][0]

    assert json.loads(row["provisional_risk_labels"]) == [
        "review_required",
        "age_scope",
    ]
    assert row["current_kb_support"] == "partial_current_kb_support"
    assert json.loads(row["policy_rule_ids"]) == ["POLICY-SAFETY-001"]
    assert json.loads(row["supported_claim_types"]) == ["age_scope"]
    assert row["scope_check"] == "within_can_support"
    assert row["source_filename"] == "test-guideline.pdf"
    assert row["source_sha256"] == "source-sha256"


def test_prepare_fails_closed_on_hash_drift_or_duplicate_candidate_id():
    module = _load_module()
    candidates = [_candidate(1), _candidate(2)]
    review_rows = [_review_row(candidate) for candidate in candidates]
    config = _config(expected_count=2)
    config["batch_sizes"] = [2]

    with pytest.raises(ValueError, match="hash mismatch"):
        module.prepare_author_batches(
            candidates,
            review_rows,
            config,
            observed_input_sha256={
                **config["expected_input_sha256"],
                "candidates": "drifted-hash",
            },
        )

    with pytest.raises(ValueError, match="重复 candidate_id"):
        module.prepare_author_batches(
            candidates,
            [review_rows[0], deepcopy(review_rows[0])],
            config,
            observed_input_sha256=dict(config["expected_input_sha256"]),
        )
