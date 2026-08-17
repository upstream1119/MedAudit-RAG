from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "benchmark120_selection.py"
)
FORMAL_CONFIG_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "configs"
    / "benchmark120_selection_v0_1.json"
)


def _load_module():
    assert MODULE_PATH.exists(), "Benchmark120 selection module is missing"
    spec = importlib.util.spec_from_file_location(
        "benchmark120_selection",
        MODULE_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pass2_row(item_id: str, *, decision: str = "answer") -> dict[str, str]:
    return {
        "pass2_item_id": item_id,
        "pass2_outcome": "accepted",
        "pass2_final_question": f"{item_id} 的最终问题",
        "pass2_expected_decision": decision,
        "pass2_current_kb_support": "supported_by_current_kb",
        "pass2_gold_evidence_status": "page_span_located",
        "pass2_required_evidence_type": '["direct_evidence"]',
        "pass2_required_claims": '["必须保留的结论"]',
        "pass2_allowed_claims": '["允许回答的结论"]',
        "pass2_forbidden_claims": '["禁止外推的结论"]',
        "pass2_missing_evidence_type": "[]",
        "pass2_missing_information": "[]",
        "pass2_risk_labels": '["direct_support"]',
        "source_id": "SRC-001",
        "source_title": "测试指南",
        "source_filename": "test.pdf",
        "source_sha256": "source-hash",
        "source_type": "clinical_guideline",
        "source_year": "2025",
        "jurisdiction": "CN",
        "page_number": "10",
        "anchor_text_span": "测试证据原文。",
        "evidence_scope": "仅支持测试范围。",
        "age_scope": "儿童",
        "applicability_conditions": "仅限测试条件。",
        "supported_claim_types": '["direct_evidence"]',
        "policy_rule_ids": "[]",
    }


def _linkage(
    candidate_id: str,
    item_id: str,
    *,
    decision: str = "answer",
) -> dict[str, str]:
    return {
        "candidate_id": candidate_id,
        "pass2_item_id": item_id,
        "origin_pool": "original",
        "pass1_outcome": "accepted",
        "pass1_final_question": f"{item_id} 的最终问题",
        "pass1_expected_decision": decision,
        "pass1_current_kb_support": "supported_by_current_kb",
        "pass1_gold_evidence_status": "page_span_located",
        "evidence_anchor_group_id": f"EAG-{candidate_id}",
        "provisional_fact_cluster_id": f"FC-{candidate_id}",
        "independence_unit_id": f"IU-{candidate_id}",
        "evidence_anchor_ids": '["ANCH-001"]',
        "source_row_sha256": f"row-{candidate_id}",
    }


def test_build_selectable_pool_keeps_only_consistent_old_rows():
    module = _load_module()
    pass2_rows = [_pass2_row("P2-001"), _pass2_row("P2-002")]
    linkage_rows = [
        _linkage("CAND-001", "P2-001"),
        _linkage("CAND-002", "P2-002", decision="review_required"),
    ]

    pool, quarantine = module.build_selectable_pool(
        pass2_rows=pass2_rows,
        linkage_rows=linkage_rows,
        resolution_rows=[],
        new_reviewed_rows=[],
        config={"corrupt_reason_pattern": r"^\?+$"},
    )

    assert [row["candidate_id"] for row in pool] == ["CAND-001"]
    assert pool[0]["annotation_pass_count"] == 2
    assert pool[0]["requires_second_pass"] is False
    assert quarantine == [
        {
            "candidate_id": "CAND-002",
            "reason": "pass1_pass2_disagreement",
        }
    ]


def test_build_selectable_pool_marks_new_author_reviewed_rows_for_second_pass():
    module = _load_module()
    new_row = {
        "candidate_id": "NEW-001",
        "question": "证据只支持复评时，能否补写个体处方？",
        "reviewed_expected_decision": "review_required",
        "reviewed_risk_labels": ["review_required"],
        "reviewed_allowed_answer_scope": "仅说明需要复评。",
        "reviewed_forbidden_claims": ["不得补写个体处方"],
        "source_id": "SRC-002",
        "source_title": "测试指南二",
        "source_filename": "test2.pdf",
        "source_sha256": "source-hash-2",
        "page_number": 20,
        "anchor_text_span": "治疗后应按时复评。",
        "evidence_scope": "仅支持复评。",
        "age_scope": "儿童",
        "applicability_conditions": "治疗后未改善。",
        "supported_claim_types": ["monitoring"],
        "policy_rule_ids": [],
        "evidence_anchor_ids": ["ANCH-002"],
        "evidence_anchor_group_id": "EAG-NEW-001",
        "provisional_fact_cluster_id": "FC-NEW-001",
        "independence_unit_id": "IU-NEW-001",
        "author_outcome": "accepted",
        "candidate_status": "author_reviewed_candidate",
    }

    pool, quarantine = module.build_selectable_pool(
        pass2_rows=[],
        linkage_rows=[],
        resolution_rows=[],
        new_reviewed_rows=[new_row],
        config={"corrupt_reason_pattern": r"^\?+$"},
    )

    assert quarantine == []
    assert len(pool) == 1
    assert pool[0]["candidate_id"] == "NEW-001"
    assert pool[0]["expected_decision"] == "review_required"
    assert pool[0]["annotation_pass_count"] == 1
    assert pool[0]["requires_second_pass"] is True
    assert pool[0]["freeze_status"] == "draft"


def test_build_selectable_pool_quarantines_corrupt_resolution_reason():
    module = _load_module()
    pass2_rows = [_pass2_row("P2-001", decision="review_required")]
    linkage_rows = [
        _linkage("CAND-001", "P2-001", decision="review_required")
    ]
    resolution_rows = [
        {
            "candidate_id": "CAND-001",
            "resolution_reason": "NG224?11????????????????",
        }
    ]

    pool, quarantine = module.build_selectable_pool(
        pass2_rows=pass2_rows,
        linkage_rows=linkage_rows,
        resolution_rows=resolution_rows,
        new_reviewed_rows=[],
        config={"corrupt_reason_pattern": r"\?{3,}"},
    )

    assert pool == []
    assert quarantine == [
        {
            "candidate_id": "CAND-001",
            "reason": "corrupt_resolution_reason",
        }
    ]


def _selection_row(decision: str, index: int) -> dict:
    shared_group = decision == "answer" and index in {1, 2}
    group_suffix = "ANSWER-SHARED" if shared_group else f"{decision}-{index:03d}"
    pass_count = 2 if index % 4 else 1
    return {
        "candidate_id": f"SEL-{decision}-{index:03d}",
        "question": f"{decision} 第 {index} 条独立审核问题？",
        "expected_decision": decision,
        "source_id": f"SRC-{group_suffix}",
        "page_number": index if not shared_group else 1,
        "fact_cluster_id": f"FC-{group_suffix}",
        "evidence_anchor_group_id": f"EAG-{group_suffix}",
        "independence_unit_id": f"IU-{group_suffix}",
        "annotation_pass_count": pass_count,
        "requires_second_pass": pass_count == 1,
        "candidate_status": "selectable_test_candidate",
        "freeze_status": "draft",
    }


def test_selection_and_grouped_split_meet_exact_protocol_quotas():
    module = _load_module()
    totals = {
        "answer": 4,
        "review_required": 4,
        "insufficient_evidence": 2,
        "boundary_refusal": 2,
    }
    rows = [
        _selection_row(decision, index)
        for decision, count in totals.items()
        for index in range(1, count + 1)
    ]

    selected_first, selection_meta = module.select_benchmark120(
        rows,
        target_distribution=totals,
        seed=20260814,
        ngram_size=3,
        similarity_threshold=1.0,
    )
    selected_second, _ = module.select_benchmark120(
        list(reversed(rows)),
        target_distribution=totals,
        seed=20260814,
        ngram_size=3,
        similarity_threshold=1.0,
    )
    split_rows, split_meta = module.propose_grouped_split(
        selected_first,
        split_targets={
            "validation": {
                "answer": 1,
                "review_required": 1,
                "insufficient_evidence": 1,
                "boundary_refusal": 1,
            },
            "pilot_test": {
                "answer": 3,
                "review_required": 3,
                "insufficient_evidence": 1,
                "boundary_refusal": 1,
            },
        },
        desired_validation_two_pass=3,
        seed=20260814,
        ngram_size=3,
        similarity_threshold=1.0,
    )

    assert [row["candidate_id"] for row in selected_first] == [
        row["candidate_id"] for row in selected_second
    ]
    assert selection_meta["selected_count"] == 12
    assert Counter(row["expected_decision"] for row in selected_first) == totals
    assert Counter(row["dataset_split"] for row in split_rows) == {
        "validation": 4,
        "pilot_test": 8,
    }
    for split_name, expected in split_meta["decision_distribution"].items():
        assert expected == Counter(
            row["expected_decision"]
            for row in split_rows
            if row["dataset_split"] == split_name
        )
    shared_splits = {
        row["dataset_split"]
        for row in split_rows
        if row["fact_cluster_id"] == "FC-ANSWER-SHARED"
    }
    assert len(shared_splits) == 1
    assert all(row["freeze_status"] == "draft" for row in split_rows)
    assert all(row.get("gold_status") is None for row in split_rows)


def test_audit_fails_closed_when_an_independence_group_crosses_splits():
    module = _load_module()
    pool = [_selection_row("answer", 1), _selection_row("answer", 2)]
    split_rows = [dict(pool[0]), dict(pool[1])]
    split_rows[0].update(
        {
            "dataset_split": "validation",
            "split_status": "proposal",
        }
    )
    split_rows[1].update(
        {
            "dataset_split": "pilot_test",
            "split_status": "proposal",
        }
    )

    with pytest.raises(ValueError, match="cross-split evidence leakage"):
        module.audit_selection_and_split(
            selectable_pool=pool,
            split_rows=split_rows,
            quarantine_rows=[],
            dev50_rows=[],
            config={
                "target_decision_distribution": {"answer": 2},
                "split_targets": {
                    "validation": {"answer": 1},
                    "pilot_test": {"answer": 1},
                },
                "expected_selectable_pool_count": 2,
                "expected_selected_count": 2,
                "expected_quarantine_count": 0,
                "expected_selected_two_pass_count": 2,
                "expected_selected_single_pass_count": 0,
                "ngram_size": 3,
                "jaccard_threshold": 1.0,
            },
        )


def test_verify_input_hashes_fails_closed_on_parent_asset_drift(tmp_path):
    module = _load_module()
    asset = tmp_path / "parent.jsonl"
    asset.write_text('{"candidate_id":"CAND-001"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="parent asset hash mismatch"):
        module.verify_input_hashes(
            {str(asset): "0" * 64},
            base_dir=Path("."),
        )


def test_formal_config_locks_benchmark120_draft_protocol():
    assert FORMAL_CONFIG_PATH.exists(), "Formal Benchmark120 config is missing"
    config = json.loads(FORMAL_CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["expected_selectable_pool_count"] == 146
    assert config["expected_quarantine_count"] == 36
    assert config["expected_selected_count"] == 120
    assert config["expected_selected_two_pass_count"] == 89
    assert config["expected_selected_single_pass_count"] == 31
    assert config["target_decision_distribution"] == {
        "answer": 40,
        "review_required": 40,
        "insufficient_evidence": 24,
        "boundary_refusal": 16,
    }
    assert config["split_targets"] == {
        "validation": {
            "answer": 13,
            "review_required": 13,
            "insufficient_evidence": 8,
            "boundary_refusal": 6,
        },
        "pilot_test": {
            "answer": 27,
            "review_required": 27,
            "insufficient_evidence": 16,
            "boundary_refusal": 10,
        },
    }
    assert config["desired_validation_two_pass_count"] == 30
    assert config["gold_promotion_allowed"] is False
    assert config["freeze_allowed"] is False
    assert len(config["input_sha256"]) == 8


def test_audit_reports_draft_status_and_zero_usage_for_valid_split():
    module = _load_module()
    pool = [
        _selection_row("answer", 3),
        _selection_row("answer", 4),
        _selection_row("review_required", 1),
        _selection_row("review_required", 2),
    ]
    split_rows = []
    for index, row in enumerate(pool):
        output = dict(row)
        output.update(
            {
                "dataset_split": "validation" if index % 2 == 0 else "pilot_test",
                "split_status": "proposal",
            }
        )
        split_rows.append(output)
    two_pass_count = sum(row["annotation_pass_count"] == 2 for row in pool)

    audit = module.audit_selection_and_split(
        selectable_pool=pool,
        split_rows=split_rows,
        quarantine_rows=[],
        dev50_rows=[],
        config={
            "target_decision_distribution": {
                "answer": 2,
                "review_required": 2,
            },
            "split_targets": {
                "validation": {"answer": 1, "review_required": 1},
                "pilot_test": {"answer": 1, "review_required": 1},
            },
            "expected_selectable_pool_count": 4,
            "expected_selected_count": 4,
            "expected_quarantine_count": 0,
            "expected_selected_two_pass_count": two_pass_count,
            "expected_selected_single_pass_count": 4 - two_pass_count,
            "ngram_size": 3,
            "jaccard_threshold": 1.0,
        },
    )

    assert audit["status"] == "draft_selection_ready_for_second_pass"
    assert audit["gold_promotion_performed"] is False
    assert audit["freeze_performed"] is False
    assert audit["cross_split_structural_overlap_count"] == 0
    assert audit["usage"] == {
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
    }
