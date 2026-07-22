import copy
from importlib import import_module

import pytest


def _load_module():
    try:
        return import_module(
            "experiments.phase6_evidence_graph.claim_evidence_aligner"
        )
    except ModuleNotFoundError:
        pytest.fail("claim_evidence_aligner module has not been implemented")


def _constraint(constraint_type: str, normalized_value: str) -> dict:
    return {
        "constraint_type": constraint_type,
        "normalized_value": normalized_value,
        "surface_forms": [normalized_value],
        "ruleset_version": "phase6b-runtime-constraint-rules-v0.3",
    }


def _evidence(
    *,
    evidence_id: str,
    content: str,
    constraints: list[dict],
    source_id: str = "SRC-002",
    page_number: int = 26,
) -> dict:
    return {
        "evidence_id": evidence_id,
        "source_id": source_id,
        "source_file": f"{source_id}.pdf",
        "page_number": page_number,
        "content": content,
        "rank_before": 1,
        "rank_after": 1,
        "runtime_constraints": constraints,
        "matched_constraint_types": [],
        "graph_score": 0.9,
    }


def _artifact(
    *,
    sample_id: str,
    evidence: list[dict],
    status: str = "success",
) -> dict:
    return {
        "artifact_schema_version": "phase6b-reranking-artifact-v0.2",
        "artifact_type": "phase6b_reranking_artifact",
        "artifact_status": status,
        "artifact_sha256": f"artifact-sha::{sample_id}",
        "sample_id": sample_id,
        "parent_inference_graph_id": f"inference_graph::{sample_id}",
        "parent_inference_graph_sha256": f"graph-sha::{sample_id}",
        "versions": {
            "schema_version": "phase6a-pedimedkg-schema-v0.1",
            "dataset_version": "dev50-v1.0",
            "kb_version": "KB-medium-v1",
        },
        "method_id": "runtime_constraint_graph_reranking",
        "method_version": "phase6b-runtime-constraint-v0.4",
        "ruleset_version": "phase6b-runtime-constraint-rules-v0.3",
        "ranked_evidence": evidence,
        "external_model_calls": 0,
        "estimated_cost": 0,
    }


def _config() -> dict:
    return {
        "config_version": "phase6b-claim-alignment-test-v0.1",
        "method_id": "constraint_grounded_claim_evidence_alignment",
        "method_version": "phase6b-claim-alignment-v0.1",
        "evidence_budget": 4,
    }


def test_marks_monitoring_claim_as_supported():
    aligner = _load_module()
    artifact = _artifact(
        sample_id="PMSQA_DEV_001",
        evidence=[
            _evidence(
                evidence_id="evidence::cap::p26",
                content="治疗48-72小时症状无改善，应再次进行评估。",
                constraints=[
                    _constraint("monitoring_window", "48-72h"),
                    _constraint("monitoring_trigger", "nonresponse"),
                    _constraint("monitoring_action", "reassess"),
                ],
            )
        ],
    )

    result = aligner.align_claims(
        artifact,
        "治疗48-72小时症状无改善，应再次评估。",
        _config(),
    )

    assert result["overall_support_state"] == "supported"
    assert result["claim_summary"] == {
        "claim_count": 1,
        "supported": 1,
        "contradicted": 0,
        "unsupported": 0,
        "insufficient_evidence": 0,
        "requires_human_review": False,
    }
    assert result["claims"][0]["supporting_evidence_ids"] == [
        "evidence::cap::p26"
    ]


def test_marks_asserted_bid_as_contradicted_by_qd_evidence():
    aligner = _load_module()
    artifact = _artifact(
        sample_id="PMSQA_DEV_002",
        evidence=[
            _evidence(
                evidence_id="evidence::mpp::p14",
                content="阿奇霉素静脉滴注10mg/(kg.d)，qd。",
                constraints=[
                    _constraint("dose", "10mg/kg/day"),
                    _constraint("frequency", "qd"),
                    _constraint("route", "iv_infusion"),
                ],
                source_id="SRC-003",
                page_number=14,
            )
        ],
    )

    result = aligner.align_claims(
        artifact,
        "阿奇霉素静脉滴注10mg/kg，一天两次可以。",
        _config(),
    )

    claim = result["claims"][0]
    assert claim["stance"] == "assert"
    assert claim["support_state"] == "contradicted"
    assert claim["contradicting_evidence_ids"] == ["evidence::mpp::p14"]
    assert "exclusive_constraint_conflict" in claim["reason_codes"]


def test_does_not_misclassify_rejected_bid_as_contradicted():
    aligner = _load_module()
    artifact = _artifact(
        sample_id="PMSQA_DEV_002",
        evidence=[
            _evidence(
                evidence_id="evidence::mpp::p14",
                content="推荐每日一次，qd。",
                constraints=[_constraint("frequency", "qd")],
                source_id="SRC-003",
                page_number=14,
            )
        ],
    )

    result = aligner.align_claims(
        artifact,
        "一天两次不可以；证据支持每日一次。",
        _config(),
    )

    assert [claim["stance"] for claim in result["claims"]] == [
        "reject",
        "assert",
    ]
    assert [claim["support_state"] for claim in result["claims"]] == [
        "supported",
        "supported",
    ]


def test_detects_evidence_insufficiency_wording_as_rejection():
    aligner = _load_module()

    assert (
        aligner.detect_claim_stance(
            "现有证据不足以支持阿奇霉素、头孢、激素和雾化同时使用"
        )
        == "reject"
    )


def test_matches_rejected_combination_to_negative_english_evidence():
    aligner = _load_module()
    artifact = _artifact(
        sample_id="PMSQA_DEV_011",
        evidence=[
            _evidence(
                evidence_id="evidence::nice::p28",
                content=(
                    "When using paracetamol or ibuprofen in children with fever, "
                    "do not give both agents simultaneously."
                ),
                constraints=[
                    _constraint(
                        "combination_antipyretic", "coadministration"
                    )
                ],
                source_id="SRC-011",
                page_number=28,
            )
        ],
    )

    result = aligner.align_claims(
        artifact,
        "布洛芬和对乙酰氨基酚不应同时服用。",
        _config(),
    )

    assert result["claims"][0]["support_state"] == "supported"


def test_partial_positive_evidence_does_not_contradict_compound_rejection():
    aligner = _load_module()
    artifact = _artifact(
        sample_id="PMSQA_DEV_007",
        evidence=[
            _evidence(
                evidence_id="evidence::antimicrobial::p7",
                content="特定感染可联合使用两种抗菌药物。",
                constraints=[
                    _constraint(
                        "combination_antimicrobial", "coadministration"
                    )
                ],
                source_id="SRC-005",
                page_number=7,
            )
        ],
    )

    result = aligner.align_claims(
        artifact,
        "现有证据不足以支持阿奇霉素、头孢、激素和雾化同时使用。",
        _config(),
    )

    claim = result["claims"][0]
    assert claim["stance"] == "reject"
    assert claim["support_state"] == "unsupported"
    assert "compound_rejection_not_fully_evidenced" in claim["reason_codes"]


def test_marks_scope_mismatch_as_unsupported_not_contradicted():
    aligner = _load_module()
    artifact = _artifact(
        sample_id="PMSQA_DEV_019",
        evidence=[
            _evidence(
                evidence_id="evidence::essential-list::p20",
                content="阿奇霉素剂型、规格包括片剂、胶囊和颗粒剂。",
                constraints=[
                    _constraint("evidence_scope", "formulation_listing")
                ],
                source_id="SRC-004",
                page_number=20,
            )
        ],
    )

    result = aligner.align_claims(
        artifact,
        "国家基本药物目录可以作为阿奇霉素剂量依据。",
        _config(),
    )

    claim = result["claims"][0]
    assert claim["support_state"] == "unsupported"
    assert "evidence_scope_mismatch" in claim["reason_codes"]
    assert claim["contradicting_evidence_ids"] == []


def test_does_not_borrow_dose_scope_from_a_different_named_source():
    aligner = _load_module()
    directory_evidence = _evidence(
        evidence_id="evidence::essential-list::p20",
        content="阿奇霉素剂型、规格包括片剂、胶囊和颗粒剂。",
        constraints=[_constraint("evidence_scope", "formulation_listing")],
        source_id="SRC-004",
        page_number=20,
    )
    directory_evidence["source_file"] = "国家基本药物目录（2018年版）.pdf"
    guideline_evidence = _evidence(
        evidence_id="evidence::guideline::p14",
        content="阿奇霉素10mg/(kg.d)，qd。",
        constraints=[_constraint("evidence_scope", "dose_guidance")],
        source_id="SRC-003",
        page_number=14,
    )
    guideline_evidence["source_file"] = "儿童肺炎支原体肺炎诊疗指南（2023年版）.pdf"
    artifact = _artifact(
        sample_id="PMSQA_DEV_019",
        evidence=[directory_evidence, guideline_evidence],
    )

    result = aligner.align_claims(
        artifact,
        "国家基本药物目录可以作为阿奇霉素剂量依据。",
        _config(),
    )

    claim = result["claims"][0]
    assert claim["support_state"] == "unsupported"
    assert claim["source_binding"]["binding_status"] == "bound"
    assert claim["source_binding"]["admitted_source_ids"] == ["SRC-004"]
    assert claim["supporting_evidence_ids"] == []


def test_supports_source_bound_rejection_of_an_unavailable_scope():
    aligner = _load_module()
    directory_evidence = _evidence(
        evidence_id="evidence::essential-list::p20",
        content="国家基本药物目录列出阿奇霉素的剂型、规格。",
        constraints=[_constraint("evidence_scope", "formulation_listing")],
        source_id="SRC-004",
        page_number=20,
    )
    directory_evidence["source_file"] = "国家基本药物目录（2018年版）.pdf"
    artifact = _artifact(
        sample_id="PMSQA_DEV_019",
        evidence=[directory_evidence],
    )

    result = aligner.align_claims(
        artifact,
        "国家基本药物目录不能作为阿奇霉素剂量依据。",
        _config(),
    )

    claim = result["claims"][0]
    assert claim["source_binding"]["binding_status"] == "bound"
    assert claim["support_state"] == "supported"


def test_reextracts_new_caution_constraint_from_parent_evidence_content():
    aligner = _load_module()
    artifact = _artifact(
        sample_id="PMSQA_DEV_005",
        evidence=[
            _evidence(
                evidence_id="evidence::mpp::p14",
                content="对婴幼儿，阿奇霉素静脉制剂的使用尤其要慎重。",
                constraints=[_constraint("age_group", "infant")],
                source_id="SRC-003",
                page_number=14,
            )
        ],
    )

    result = aligner.align_claims(
        artifact,
        "婴幼儿使用阿奇霉素静脉制剂需慎重。",
        _config(),
    )

    claim = result["claims"][0]
    assert claim["support_state"] == "supported"
    assert {
        check["constraint_type"] for check in claim["constraint_checks"]
    } >= {"age_group", "contraindication_action"}


def test_treats_contraindication_actions_as_normative_values():
    aligner = _load_module()
    artifact = _artifact(
        sample_id="PMSQA_DEV_043",
        evidence=[
            _evidence(
                evidence_id="evidence::antimicrobial::p27",
                content="对相关药物过敏者禁用，其他过敏史患者慎用。",
                constraints=[
                    _constraint("contraindication_check", "allergy_history"),
                    _constraint("contraindication_action", "avoid"),
                    _constraint("contraindication_action", "caution"),
                ],
                source_id="SRC-005",
                page_number=27,
            )
        ],
    )

    result = aligner.align_claims(
        artifact,
        "对相关药物过敏时应禁用或慎用。",
        _config(),
    )

    claim = result["claims"][0]
    assert claim["support_state"] == "supported"
    assert {
        check["claim_value"]
        for check in claim["constraint_checks"]
        if check["constraint_type"] == "contraindication_action"
    } == {"avoid", "caution"}
    assert {
        check["claim_stance"] for check in claim["constraint_checks"]
    } == {"assert"}


def test_marks_claim_as_insufficient_when_parent_has_no_evidence():
    aligner = _load_module()
    artifact = _artifact(
        sample_id="PMSQA_DEV_006",
        evidence=[],
        status="insufficient_graph_evidence",
    )

    result = aligner.align_claims(
        artifact,
        "小儿支气管肺炎可以静脉使用氨溴索。",
        _config(),
    )

    assert result["overall_support_state"] == "insufficient_evidence"
    assert result["claims"][0]["support_state"] == "insufficient_evidence"
    assert result["claims"][0]["reason_codes"] == [
        "parent_has_no_admitted_evidence"
    ]


def test_preserves_boundary_refusal_without_claim_inference():
    aligner = _load_module()
    artifact = _artifact(
        sample_id="PMSQA_DEV_003",
        evidence=[],
        status="boundary_refusal",
    )

    result = aligner.align_claims(
        artifact,
        "无法根据现有信息开具个体化处方，请由医生线下评估。",
        _config(),
    )

    assert result["artifact_status"] == "boundary_refusal"
    assert result["overall_support_state"] == "not_applicable"
    assert result["claims"] == []
    assert result["claim_summary"]["claim_count"] == 0


def test_rejects_gold_content_and_is_deterministic_without_parent_mutation():
    aligner = _load_module()
    artifact = _artifact(
        sample_id="PMSQA_DEV_001",
        evidence=[
            _evidence(
                evidence_id="evidence::cap::p26",
                content="48-72小时无改善应再次评估。",
                constraints=[
                    _constraint("monitoring_window", "48-72h"),
                    _constraint("monitoring_trigger", "nonresponse"),
                    _constraint("monitoring_action", "reassess"),
                ],
            )
        ],
    )
    before = copy.deepcopy(artifact)

    first = aligner.align_claims(
        artifact,
        "48-72小时无改善应再次评估。",
        _config(),
    )
    second = aligner.align_claims(
        artifact,
        "48-72小时无改善应再次评估。",
        _config(),
    )

    assert first == second
    assert artifact == before
    assert first["external_model_calls"] == 0
    assert first["estimated_cost"] == 0

    contaminated = copy.deepcopy(artifact)
    contaminated["expected_decision"] = "answer"
    with pytest.raises(ValueError, match="gold-only key"):
        aligner.align_claims(
            contaminated,
            "48-72小时无改善应再次评估。",
            _config(),
        )
