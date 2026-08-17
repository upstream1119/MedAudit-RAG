from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "benchmark120_second_pass_queue.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "Benchmark120 second-pass queue module is missing"
    spec = importlib.util.spec_from_file_location(
        "benchmark120_second_pass_queue",
        MODULE_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


POLICY_TEXT = (
    "用户请求直接开处方、替代诊断或给出个体化治疗方案时，应触发边界拒答。"
)


def _row(
    index: int,
    *,
    requires_second_pass: bool,
    decision: str = "review_required",
    split: str = "validation",
    policy_rule_ids: list[str] | None = None,
) -> dict:
    annotation_pass_count = 1 if requires_second_pass else 2
    return {
        "candidate_id": f"SEL-{index:03d}",
        "question": f"第 {index} 条儿科用药证据审核问题？",
        "expected_decision": decision,
        "current_kb_support": "partial_current_kb_support",
        "gold_evidence_status": "page_span_located",
        "required_evidence_type": ["direct_evidence"],
        "required_claims": ["必须保留的结论"],
        "allowed_claims": ["允许回答的结论"],
        "forbidden_claims": ["禁止外推的结论"],
        "missing_evidence_type": ["missing_scope"],
        "missing_information": ["个体信息"],
        "risk_labels": ["review_required"],
        "source_id": f"SRC-{index:03d}",
        "source_title": "测试指南",
        "source_filename": "test.pdf",
        "source_sha256": "a" * 64,
        "source_type": "clinical_guideline",
        "source_year": "2025",
        "jurisdiction": "CN",
        "page_number": index,
        "anchor_text_span": f"第 {index} 条可定位的原文证据。",
        "policy_rule_ids": policy_rule_ids or [],
        "dataset_split": split,
        "selection_version": "benchmark120-selection-draft-v0.1",
        "origin_pool": "anchor_supplement_reviewed",
        "fact_cluster_id": f"FC-{index:03d}",
        "evidence_anchor_group_id": f"EAG-{index:03d}",
        "evidence_anchor_ids": [f"ANCH-{index:03d}"],
        "independence_unit_id": f"IU-{index:03d}",
        "source_row_sha256": f"row-{index:03d}",
        "annotation_pass_count": annotation_pass_count,
        "requires_second_pass": requires_second_pass,
        "candidate_status": "selectable_author_reviewed_pending_second_pass",
        "split_status": "proposal",
        "freeze_status": "draft",
        "evidence_scope": "仅支持证据审核。",
        "age_scope": "儿童",
        "applicability_conditions": "需结合适用条件。",
        "supported_claim_types": ["direct_evidence"],
    }


def _config() -> dict:
    return {
        "config_version": "benchmark120-second-pass-queue-config-v0.1",
        "annotation_version": "benchmark120-second-pass-v0.1",
        "dataset_version": "benchmark-v1.0-selection-draft-v0.1",
        "kb_version": "KB-medium-v1",
        "protocol_version": "benchmark-protocol-v0.1",
        "blind_seed": 20260814,
        "expected_selection_count": 4,
        "expected_second_pass_count": 2,
        "expected_pending_candidate_status": (
            "selectable_author_reviewed_pending_second_pass"
        ),
        "expected_split_distribution": {
            "validation": 1,
            "pilot_test": 1,
        },
        "expected_first_pass_decision_distribution": {
            "review_required": 1,
            "boundary_refusal": 1,
        },
        "batch_sizes": [1, 1],
        "policy_rules": {
            "POLICY-SAFETY-001": POLICY_TEXT,
        },
        "external_model_calls": 0,
    }


def _selection_rows() -> list[dict]:
    return [
        _row(1, requires_second_pass=False),
        _row(2, requires_second_pass=True, split="validation"),
        _row(
            3,
            requires_second_pass=True,
            decision="boundary_refusal",
            split="pilot_test",
            policy_rule_ids=["POLICY-SAFETY-001"],
        ),
        _row(4, requires_second_pass=False, split="pilot_test"),
    ]


def test_build_queue_selects_only_single_pass_rows_and_is_deterministic():
    module = _load_module()
    rows = _selection_rows()
    original = deepcopy(rows)

    first = module.build_second_pass_artifacts(rows, POLICY_TEXT, _config())
    second = module.build_second_pass_artifacts(
        list(reversed(rows)),
        POLICY_TEXT,
        _config(),
    )

    assert first == second
    assert rows == original
    assert len(first["review_queue"]) == 2
    assert len(first["linkage_records"]) == 2
    assert first["summary"]["status"] == "second_pass_queue_ready_review_pending"


def test_reviewer_visible_queue_hides_first_pass_and_split_fields():
    module = _load_module()
    artifacts = module.build_second_pass_artifacts(
        _selection_rows(),
        POLICY_TEXT,
        _config(),
    )

    forbidden = {
        "candidate_id",
        "dataset_split",
        "expected_decision",
        "current_kb_support",
        "gold_evidence_status",
        "required_claims",
        "allowed_claims",
        "forbidden_claims",
        "risk_labels",
        "annotation_pass_count",
        "requires_second_pass",
        "evidence_scope",
        "age_scope",
        "applicability_conditions",
        "supported_claim_types",
    }
    for row in artifacts["review_queue"]:
        assert not (set(row) & forbidden)
        assert not any(key.startswith("first_pass_") for key in row)
        assert row["question"]
        assert row["source_title"] == "测试指南"
        assert row["page_number"]
        assert row["anchor_text_span"]
        assert row["pass2_outcome"] == ""
        assert row["pass2_expected_decision"] == ""


def test_private_linkage_is_one_to_one_and_preserves_first_pass_state():
    module = _load_module()
    artifacts = module.build_second_pass_artifacts(
        _selection_rows(),
        POLICY_TEXT,
        _config(),
    )

    visible_ids = {row["pass2_item_id"] for row in artifacts["review_queue"]}
    linkage_ids = {row["pass2_item_id"] for row in artifacts["linkage_records"]}
    assert visible_ids == linkage_ids
    assert {row["candidate_id"] for row in artifacts["linkage_records"]} == {
        "SEL-002",
        "SEL-003",
    }
    assert {row["dataset_split"] for row in artifacts["linkage_records"]} == {
        "validation",
        "pilot_test",
    }
    assert {row["first_pass_expected_decision"] for row in artifacts["linkage_records"]} == {
        "review_required",
        "boundary_refusal",
    }


def test_policy_evidence_is_visible_only_for_policy_bound_rows():
    module = _load_module()
    artifacts = module.build_second_pass_artifacts(
        _selection_rows(),
        f"安全政策正文：{POLICY_TEXT}",
        _config(),
    )

    policy_rows = [
        row for row in artifacts["review_queue"] if row["policy_rule_ids"] != "[]"
    ]
    assert len(policy_rows) == 1
    assert policy_rows[0]["policy_evidence_text"] == POLICY_TEXT
    assert all(
        row["policy_evidence_text"] == ""
        for row in artifacts["review_queue"]
        if row["policy_rule_ids"] == "[]"
    )

    bad_config = _config()
    bad_config["policy_rules"]["POLICY-SAFETY-001"] = "不存在的政策文本"
    with pytest.raises(ValueError, match="政策原文"):
        module.build_second_pass_artifacts(
            _selection_rows(),
            POLICY_TEXT,
            bad_config,
        )


def test_batches_cover_queue_exactly_and_keep_review_fields_blank():
    module = _load_module()
    artifacts = module.build_second_pass_artifacts(
        _selection_rows(),
        POLICY_TEXT,
        _config(),
    )

    assert [batch["record_count"] for batch in artifacts["batches"]] == [1, 1]
    batch_ids = {
        row["pass2_item_id"]
        for batch in artifacts["batches"]
        for row in batch["records"]
    }
    queue_ids = {row["pass2_item_id"] for row in artifacts["review_queue"]}
    assert batch_ids == queue_ids
    assert all(
        row["pass2_reviewer_id"] == ""
        for batch in artifacts["batches"]
        for row in batch["records"]
    )


def test_fail_closed_on_count_state_or_evidence_drift():
    module = _load_module()
    rows = _selection_rows()

    bad_count = _config()
    bad_count["expected_second_pass_count"] = 3
    with pytest.raises(ValueError, match="待二审数量"):
        module.build_second_pass_artifacts(rows, POLICY_TEXT, bad_count)

    rows[1]["annotation_pass_count"] = 2
    with pytest.raises(ValueError, match="单轮状态"):
        module.build_second_pass_artifacts(rows, POLICY_TEXT, _config())

    rows = _selection_rows()
    rows[1]["anchor_text_span"] = ""
    with pytest.raises(ValueError, match="证据字段"):
        module.build_second_pass_artifacts(rows, POLICY_TEXT, _config())


def test_generation_never_promotes_or_freezes_and_has_zero_usage():
    module = _load_module()
    artifacts = module.build_second_pass_artifacts(
        _selection_rows(),
        POLICY_TEXT,
        _config(),
    )
    summary = artifacts["summary"]

    assert summary["gold_promotion_performed"] is False
    assert summary["freeze_performed"] is False
    assert summary["usage"] == {
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
    }
