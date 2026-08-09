from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "experiments" / "phase7_formal_experiments" / "benchmark_candidate_replenishment.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_candidate_replenishment", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _anchor(anchor_id: str = "A-1") -> dict:
    return {
        "anchor_id": anchor_id,
        "source_id": "SRC-1",
        "source_title": "指南",
        "source_filename": "guide.pdf",
        "source_sha256": "a" * 64,
        "page_number": 8,
        "text_span": "原文证据片段。",
        "supported_claim_types": ["dose_frequency"],
        "evidence_scope": "剂量和频次审核；不支持个体处方。",
        "age_scope": "儿童",
        "applicability_conditions": "仅限来源适用范围。",
        "scope_check": "within_can_support",
        "verification_status": "author_verified_anchor",
    }


def _parent(anchor_id: str = "A-1") -> dict:
    return {
        "candidate_id": "P-1",
        "question": "儿童用药剂量应如何审核？",
        "evidence_anchor_ids": [anchor_id],
        "evidence_anchor_group_id": "EAG-1",
        "provisional_fact_cluster_id": "FC-1",
        "independence_unit_id": "IU-1",
    }


def _spec(anchor_id: str = "A-1", decision: str = "review_required") -> dict:
    support = {
        "review_required": "partial_current_kb_support",
        "insufficient_evidence": "not_supported_by_current_kb",
        "boundary_refusal": "policy_rule",
    }[decision]
    return {
        "anchor_id": anchor_id,
        "question": "待审回答省略适用条件并给出确定结论时，当前证据是否足够支持？",
        "provisional_expected_decision": decision,
        "current_kb_support": support,
        "missing_evidence_type": ["direct_guideline_evidence"] if decision == "insufficient_evidence" else [],
        "policy_rule_ids": ["POLICY-SAFETY-001"] if decision == "boundary_refusal" else [],
        "provisional_risk_labels": [decision, "evidence_scope_boundary"],
        "challenge_type": "omitted_applicability_condition",
    }


def _config(specs: list[dict]) -> dict:
    return {
        "config_version": "supplement-v0.1",
        "generator_version": "supplement-v0.1",
        "dataset_version": "supplement-draft",
        "schema_version": "benchmark-schema-v0.1",
        "protocol_version": "benchmark-protocol-v0.1",
        "kb_version": "KB-medium-v1",
        "candidate_status": "draft_candidate_unverified",
        "expected_candidate_count": len(specs),
        "target_decision_distribution": {
            "review_required": sum(s["provisional_expected_decision"] == "review_required" for s in specs),
            "insufficient_evidence": sum(s["provisional_expected_decision"] == "insufficient_evidence" for s in specs),
            "boundary_refusal": sum(s["provisional_expected_decision"] == "boundary_refusal" for s in specs),
        },
        "external_model_calls": 0,
        "challenge_specs": specs,
    }


def test_build_supplement_candidates_is_deterministic_and_preserves_provenance():
    module = _load_module()
    anchors = [_anchor()]
    parents = [_parent()]
    config = _config([_spec()])
    original_anchors = deepcopy(anchors)
    original_parents = deepcopy(parents)

    first = module.build_supplement_candidates(anchors, parents, config)
    second = module.build_supplement_candidates(anchors, parents, config)

    assert first == second
    assert anchors == original_anchors
    assert parents == original_parents
    row = first[0]
    assert row["source_id"] == "SRC-1"
    assert row["page_number"] == 8
    assert row["anchor_text_span"] == "原文证据片段。"
    assert row["evidence_anchor_group_id"] == "EAG-1"
    assert row["provisional_fact_cluster_id"] == "FC-1"
    assert row["independence_unit_id"] == "IU-1"
    assert row["candidate_role"] == "evidence_boundary_challenge"
    assert row["candidate_status"] == "draft_candidate_unverified"


def test_build_supplement_candidates_rejects_unknown_or_repeated_anchor():
    module = _load_module()
    with pytest.raises(ValueError, match="未知锚点"):
        module.build_supplement_candidates([_anchor()], [_parent()], _config([_spec("A-x")]))

    duplicate_specs = [_spec(), {**_spec(), "question": "另一条问题"}]
    with pytest.raises(ValueError, match="重复使用锚点"):
        module.build_supplement_candidates([_anchor()], [_parent()], _config(duplicate_specs))


def test_validate_distribution_and_parent_question_overlap():
    module = _load_module()
    bad_config = _config([_spec()])
    bad_config["target_decision_distribution"] = {
        "review_required": 0,
        "insufficient_evidence": 1,
        "boundary_refusal": 0,
    }
    with pytest.raises(ValueError, match="决策分布"):
        module.build_supplement_candidates([_anchor()], [_parent()], bad_config)

    candidate = module.build_supplement_candidates([_anchor()], [_parent()], _config([_spec()]))[0]
    candidate["question"] = _parent()["question"]
    audit = module.audit_parent_overlap([candidate], [_parent()], {"ngram_size": 3, "jaccard_threshold": 0.65})
    assert audit["audited_candidates"][0]["parent_overlap_decision"] == "reject"
    assert "exact_question_parent" in audit["audited_candidates"][0]["parent_overlap_reasons"]


def test_usage_is_explicitly_zero():
    module = _load_module()
    assert module.zero_usage() == {
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
    }
