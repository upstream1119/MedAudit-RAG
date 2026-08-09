import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "benchmark_candidate_builder.py"
)
ANCHOR_POOL_PATH = (
    REPO_ROOT
    / "revision"
    / "benchmark"
    / "benchmark_v1"
    / "evidence_anchor_pool_v0_1.jsonl"
)
CONFIG_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "configs"
    / "benchmark_candidate_builder_v0_1.json"
)


def _load_module():
    assert MODULE_PATH.exists(), "Benchmark candidate builder is not implemented"
    spec = importlib.util.spec_from_file_location(
        "benchmark_candidate_builder",
        MODULE_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config() -> dict:
    return {
        "config_version": "benchmark-candidate-builder-test-v0.1",
        "generator_version": "benchmark-candidate-builder-test-v0.1",
        "dataset_version": "benchmark-v1.0-draft",
        "schema_version": "benchmark-schema-v0.1",
        "protocol_version": "benchmark-protocol-v0.1",
        "kb_version": "KB-medium-v1",
        "expected_anchor_count": 1,
        "variants_per_anchor": 2,
        "minimum_candidate_count": 2,
        "maximum_candidate_count": 2,
        "candidate_status": "draft_candidate_unverified",
        "minimum_question_chars": 12,
        "maximum_question_chars": 320,
        "require_all_decision_types": False,
        "policy_rule": {
            "rule_id": "POLICY-SAFETY-001",
            "rule_version": "2026-v1",
            "source_path": "docs/medical_safety_policy.md",
            "required_text": "不能替代医生诊断或处方",
        },
        "insufficient_claim_types": [
            "pediatric_evidence_gap",
            "adult_evidence_extrapolation_risk",
        ],
        "insufficient_scope_terms": ["儿童证据缺失"],
        "boundary_refusal_terms": ["个体化处方", "个体处方", "直接开药"],
        "missing_evidence_term_map": {
            "剂量": "dose",
            "频次": "frequency",
            "儿童证据": "pediatric_direct_evidence",
        },
    }


def _anchor(
    *,
    anchor_id: str = "ANCH-test-001",
    status: str = "author_verified_anchor",
    scope: str = (
        "儿童长期用药时的肝功能监测原则；"
        "不支持脱离患者信息给出个体化处方或具体剂量。"
    ),
    age_scope: str = "儿童",
    claim_types: list[str] | None = None,
) -> dict:
    return {
        "anchor_id": anchor_id,
        "source_id": "SRC-101",
        "source_title": "测试儿科指南",
        "source_filename": "test.pdf",
        "source_sha256": "sha256-test",
        "page_number": 8,
        "text_span": "儿童长期用药时应结合肝功能进行监测。",
        "supported_claim_types": claim_types or ["monitoring", "dose_context"],
        "evidence_scope": scope,
        "age_scope": age_scope,
        "applicability_conditions": "仅限指南明确覆盖的儿童人群",
        "scope_check": "within_can_support",
        "verification_status": status,
        "reviewer_id": "author-01",
        "author_reviewed_at": "2026-07-29",
    }


def test_unverified_anchor_is_rejected():
    module = _load_module()

    with pytest.raises(ValueError, match="author_verified_anchor"):
        module.build_candidate_pool(
            [_anchor(status="candidate_unverified")],
            _config(),
        )


def test_two_distinct_candidates_are_built_for_each_anchor():
    module = _load_module()
    candidates = module.build_candidate_pool([_anchor()], _config())

    assert len(candidates) == 2
    assert {row["candidate_role"] for row in candidates} == {
        "direct_support",
        "scope_boundary",
    }
    assert len({row["question"] for row in candidates}) == 2
    assert all(
        row["candidate_status"] == "draft_candidate_unverified"
        for row in candidates
    )
    assert all(row["evidence_anchor_ids"] == ["ANCH-test-001"] for row in candidates)


def test_candidate_ids_and_order_are_deterministic():
    module = _load_module()
    anchors = [
        _anchor(
            anchor_id="ANCH-test-002",
            scope=(
                "儿童长期用药时的肾功能监测原则；"
                "不支持脱离患者信息直接换药。"
            ),
        ),
        _anchor(anchor_id="ANCH-test-001"),
    ]
    config = {**_config(), "expected_anchor_count": 2}
    config["minimum_candidate_count"] = 4
    config["maximum_candidate_count"] = 4

    first = module.build_candidate_pool(anchors, config)
    second = module.build_candidate_pool(list(reversed(anchors)), config)

    assert first == second
    assert len({row["candidate_id"] for row in first}) == 4


def test_boundary_refusal_candidate_binds_project_policy_rule():
    module = _load_module()
    candidates = module.build_candidate_pool([_anchor()], _config())
    boundary = next(
        row
        for row in candidates
        if row["provisional_expected_decision"] == "boundary_refusal"
    )

    assert boundary["policy_rule_ids"] == ["POLICY-SAFETY-001"]
    assert boundary["current_kb_support"] == "policy_rule"


def test_insufficient_evidence_candidate_declares_missing_evidence_type():
    module = _load_module()
    anchor = _anchor(
        scope=(
            "急性鼻窦炎延迟抗菌药策略的现有研究范围；"
            "儿童证据缺失，直接研究对象为成人。"
        ),
        age_scope="儿童证据缺失，直接研究对象为成人",
        claim_types=[
            "pediatric_evidence_gap",
            "adult_evidence_extrapolation_risk",
        ],
    )
    candidates = module.build_candidate_pool([anchor], _config())
    insufficient = next(
        row
        for row in candidates
        if row["provisional_expected_decision"] == "insufficient_evidence"
    )

    assert insufficient["missing_evidence_type"]
    assert "pediatric_direct_evidence" in insufficient["missing_evidence_type"]


def test_non_pediatric_scope_is_not_rewritten_as_pediatric_recommendation():
    module = _load_module()
    anchor = _anchor(
        scope=(
            "一般人群的抗菌药复核原则；"
            "不支持改写为儿童专属剂量或个体处方。"
        ),
        age_scope="一般人群，包含儿童但非儿科专属",
    )
    candidates = module.build_candidate_pool([anchor], _config())

    assert all("一般人群，包含儿童但非儿科专属" in row["question"] for row in candidates)
    assert all("儿童通用推荐" not in row["question"] for row in candidates)


def test_model_input_export_contains_no_evaluation_or_provenance_labels():
    module = _load_module()
    candidate = module.build_candidate_pool([_anchor()], _config())[0]

    assert module.to_model_input_record(candidate) == {
        "sample_id": candidate["candidate_id"],
        "question": candidate["question"],
    }


def test_embedded_boundary_marker_is_extracted_from_metadata_clause():
    module = _load_module()

    supported, boundary = module._split_evidence_scope(
        "发热儿童紧急评估条件；属于委员会理由，不等同确诊标准或治疗建议。"
    )

    assert supported == "发热儿童紧急评估条件"
    assert boundary == "确诊标准或治疗建议"


def test_metadata_only_tail_is_not_turned_into_boundary_question():
    module = _load_module()

    supported, boundary = module._split_evidence_scope(
        "发热儿童紧急肠外抗菌药适应条件；国际补充证据。"
    )

    assert supported == "发热儿童紧急肠外抗菌药适应条件"
    assert boundary == "超出上述已核验范围的个体化诊断或处方"


def test_default_diagnosis_or_prescription_boundary_binds_policy_rule():
    module = _load_module()
    config = _config()
    config["boundary_refusal_terms"].append("个体化诊断或处方")
    candidates = module.build_candidate_pool(
        [_anchor(scope="儿童长期用药监测原则；国际补充证据。")],
        config,
    )
    boundary = next(
        row for row in candidates if row["candidate_role"] == "scope_boundary"
    )

    assert boundary["provisional_expected_decision"] == "boundary_refusal"
    assert boundary["policy_rule_ids"] == ["POLICY-SAFETY-001"]
    assert boundary["current_kb_support"] == "policy_rule"


def test_second_supported_clause_is_preserved_when_no_boundary_marker_exists():
    module = _load_module()

    supported, boundary = module._split_evidence_scope(
        "急性肾盂肾炎口服优先原则；静脉治疗应在48小时复核并在可行时序贯口服。"
    )

    assert supported == (
        "急性肾盂肾炎口服优先原则；"
        "静脉治疗应在48小时复核并在可行时序贯口服"
    )
    assert boundary == "超出上述已核验范围的个体化诊断或处方"


def test_real_anchor_pool_builds_auditable_140_to_160_candidate_drafts():
    module = _load_module()
    anchors = module.load_jsonl(ANCHOR_POOL_PATH)
    config = module.load_config(CONFIG_PATH)
    candidates = module.build_candidate_pool(anchors, config)
    summary = module.validate_candidate_pool(candidates, anchors, config)

    assert 140 <= len(candidates) <= 160
    assert len(candidates) == 144
    assert len({row["candidate_id"] for row in candidates}) == len(candidates)
    assert len({row["question"] for row in candidates}) == len(candidates)
    assert summary["candidate_count"] == 144
    assert set(summary["decision_distribution"]) == {
        "answer",
        "review_required",
        "insufficient_evidence",
        "boundary_refusal",
    }
    assert all(summary["decision_distribution"].values())
    assert summary["external_model_calls"] == 0
