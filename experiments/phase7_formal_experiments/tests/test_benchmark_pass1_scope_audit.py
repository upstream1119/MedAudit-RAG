from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "experiments" / "phase7_formal_experiments" / "benchmark_pass1_scope_audit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_pass1_scope_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _config() -> dict:
    return {
        "promotable_outcomes": ["accepted", "revise"],
        "medication_markers": ["药", "剂量", "频次", "抗菌", "处方", "用药"],
        "non_medication_claim_types": ["infant_temperature_measurement_method"],
        "scope_reject_terms": ["非药物", "超出儿科用药"],
    }


def _row(**overrides) -> dict:
    row = {
        "annotation_order": 1,
        "candidate_id": "C-1",
        "independence_unit_id": "IU-1",
        "question": "4周龄以下婴儿应如何测量体温？",
        "pass1_final_question": "4周龄以下婴儿应如何测量体温？",
        "pass1_outcome": "accepted",
        "pass1_expected_decision": "answer",
        "supported_claim_types": ["infant_temperature_measurement_method"],
        "pass1_required_evidence_type": ["infant_temperature_measurement_method"],
        "pass1_risk_labels": ["infant_temperature_measurement_method"],
        "pass1_review_reason": "逐页核对后接受。",
        "source_id": "SRC-011",
        "source_title": "NICE NG143",
        "page_number": 5,
    }
    row.update(overrides)
    return row


def test_scope_audit_flags_promotable_non_medication_candidate_without_mutation():
    module = _load_module()
    rows = [_row()]
    original = deepcopy(rows)

    report = module.audit_pass1_scope(rows, _config())

    assert rows == original
    assert report["promotable_count"] == 1
    assert report["flagged_count"] == 1
    assert report["flagged_rows"][0]["candidate_id"] == "C-1"
    assert "non_medication_claim_type" in report["flagged_rows"][0]["flag_reasons"]
    assert report["usage"] == {
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
    }


def test_scope_audit_keeps_medication_and_explicit_safety_boundary_candidates():
    module = _load_module()
    dose = _row(
        candidate_id="C-dose",
        question="儿童阿奇霉素静滴剂量和给药频次如何审核？",
        pass1_final_question="儿童阿奇霉素静滴剂量和给药频次如何审核？",
        supported_claim_types=["dose_frequency"],
        pass1_required_evidence_type=["dose_frequency"],
        pass1_risk_labels=["dose_frequency"],
    )
    boundary = _row(
        candidate_id="C-boundary",
        question="能否仅凭尿液检测结果直接给儿童开抗菌药处方？",
        pass1_final_question="能否仅凭尿液检测结果直接给儿童开抗菌药处方？",
        pass1_expected_decision="boundary_refusal",
        supported_claim_types=["uti_testing_indication"],
        pass1_required_evidence_type=["uti_testing_indication"],
        pass1_risk_labels=["prescription_boundary"],
    )

    report = module.audit_pass1_scope([dose, boundary], _config())

    assert report["flagged_count"] == 0


def test_scope_audit_ignores_rejected_rows_and_flags_sibling_inconsistency():
    module = _load_module()
    rejected = _row(
        candidate_id="C-reject",
        independence_unit_id="IU-shared",
        pass1_outcome="reject",
        pass1_review_reason="该问题是非药物测量问题，超出儿科用药范围。",
    )
    accepted = _row(candidate_id="C-accepted", independence_unit_id="IU-shared")

    report = module.audit_pass1_scope([rejected, accepted], _config())

    assert report["promotable_count"] == 1
    assert report["flagged_count"] == 1
    assert "sibling_scope_inconsistency" in report["flagged_rows"][0]["flag_reasons"]

