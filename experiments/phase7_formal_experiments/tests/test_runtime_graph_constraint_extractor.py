from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "runtime_graph_constraint_extractor.py"
)
REAL_LEXICON_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "configs"
    / "runtime_graph_entity_lexicon_v0_2.json"
)


def _load_module():
    assert MODULE_PATH.exists(), "Phase 7 runtime graph extractor is missing"
    spec = importlib.util.spec_from_file_location(
        "runtime_graph_constraint_extractor",
        MODULE_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lexicon() -> dict:
    return {
        "lexicon_version": "test-entity-lexicon-v0.2",
        "entries": [
            {
                "constraint_type": "clinical_condition",
                "normalized_value": "mycoplasma_pneumoniae_pneumonia",
                "aliases": ["MPP", "肺炎支原体肺炎"],
                "strong_anchor": True,
            },
            {
                "constraint_type": "clinical_condition",
                "normalized_value": "bronchiolitis",
                "aliases": ["毛细支气管炎", "bronchiolitis"],
                "strong_anchor": True,
            },
            {
                "constraint_type": "medication",
                "normalized_value": "azithromycin",
                "aliases": ["阿奇霉素", "azithromycin"],
                "strong_anchor": True,
            },
            {
                "constraint_type": "evidence_topic",
                "normalized_value": "neurologic_complication",
                "aliases": ["神经系统并发症"],
                "strong_anchor": False,
            },
            {
                "constraint_type": "medication_class",
                "normalized_value": "antimicrobial",
                "aliases": ["抗菌药", "antimicrobial"],
                "strong_anchor": True,
            },
            {
                "constraint_type": "medication_class",
                "normalized_value": "corticosteroid",
                "aliases": ["糖皮质激素", "corticosteroid"],
                "strong_anchor": True,
            },
        ],
    }


def test_entity_constraints_extend_phase6_runtime_constraints():
    module = _load_module()

    constraints = module.extract_graph_runtime_constraints(
        "MPP神经系统并发症中的阿奇霉素疗程应如何理解？",
        lexicon=_lexicon(),
    )

    pairs = {
        (row["constraint_type"], row["normalized_value"])
        for row in constraints
    }
    assert pairs == {
        ("clinical_condition", "mycoplasma_pneumoniae_pneumonia"),
        ("evidence_topic", "neurologic_complication"),
        ("medication", "azithromycin"),
    }
    assert all(row["ruleset_version"] for row in constraints)


def test_phase6_constraint_is_preserved_and_deduplicated():
    module = _load_module()

    constraints = module.extract_graph_runtime_constraints(
        "肺炎支原体肺炎患儿静脉滴注阿奇霉素，每日一次。",
        lexicon=_lexicon(),
    )

    pairs = [
        (row["constraint_type"], row["normalized_value"])
        for row in constraints
    ]
    assert pairs.count(("frequency", "qd")) == 1
    assert pairs.count(("route", "iv_infusion")) == 1
    assert pairs.count(("medication", "azithromycin")) == 1


def test_condition_conflict_is_fail_closed_for_disjoint_diseases():
    module = _load_module()
    query_constraints = module.extract_graph_runtime_constraints(
        "MPP患儿如何用药？",
        lexicon=_lexicon(),
    )
    candidate_constraints = module.extract_graph_runtime_constraints(
        "毛细支气管炎患儿的支持治疗。",
        lexicon=_lexicon(),
    )

    assert module.has_condition_conflict(
        query_constraints,
        candidate_constraints,
    )


def test_shared_condition_does_not_trigger_conflict():
    module = _load_module()
    query_constraints = module.extract_graph_runtime_constraints(
        "MPP患儿如何用药？",
        lexicon=_lexicon(),
    )
    candidate_constraints = module.extract_graph_runtime_constraints(
        "肺炎支原体肺炎的阿奇霉素治疗。",
        lexicon=_lexicon(),
    )

    assert not module.has_condition_conflict(
        query_constraints,
        candidate_constraints,
    )


def test_strong_anchor_requires_condition_medication_or_class():
    module = _load_module()
    only_topic = module.extract_graph_runtime_constraints(
        "神经系统并发症的适用边界。",
        lexicon=_lexicon(),
    )
    with_medication = module.extract_graph_runtime_constraints(
        "阿奇霉素的适用边界。",
        lexicon=_lexicon(),
    )

    assert not module.has_strong_anchor(only_topic)
    assert module.has_strong_anchor(with_medication)


def test_reliable_path_requires_discriminative_content_match():
    module = _load_module()
    query = module.extract_graph_runtime_constraints(
        "MPP神经系统并发症中的阿奇霉素疗程如何理解？",
        lexicon=_lexicon(),
    )
    candidate_content = module.extract_graph_runtime_constraints(
        "神经系统并发症可涉及阿奇霉素疗程。",
        lexicon=_lexicon(),
    )
    candidate_context = module.extract_graph_runtime_constraints(
        "神经系统并发症可涉及阿奇霉素疗程。",
        "MPP指南.pdf",
        lexicon=_lexicon(),
    )

    assessment = module.assess_constraint_path(
        query,
        candidate_context,
        candidate_content,
        minimum_matched_constraint_types=2,
    )

    assert assessment["qualified"] is True
    assert assessment["reason"] == "qualified"
    assert assessment["matched_constraint_types"] == [
        "clinical_condition",
        "evidence_topic",
        "medication",
    ]


def test_metadata_only_match_is_not_a_reliable_path():
    module = _load_module()
    query = module.extract_graph_runtime_constraints(
        "MPP患儿使用阿奇霉素是否有依据？",
        lexicon=_lexicon(),
    )
    candidate_content = module.extract_graph_runtime_constraints(
        "本页仅包含一般注意事项。",
        lexicon=_lexicon(),
    )
    candidate_context = module.extract_graph_runtime_constraints(
        "本页仅包含一般注意事项。",
        "MPP阿奇霉素指南.pdf",
        lexicon=_lexicon(),
    )

    assessment = module.assess_constraint_path(
        query,
        candidate_context,
        candidate_content,
        minimum_matched_constraint_types=2,
    )

    assert assessment["qualified"] is False
    assert assessment["reason"] == "no_content_supported_match"


def test_generic_medication_class_alone_is_not_discriminative_content():
    module = _load_module()
    query = module.extract_graph_runtime_constraints(
        "MPP抗菌药治疗是否有依据？",
        lexicon=_lexicon(),
    )
    candidate_content = module.extract_graph_runtime_constraints(
        "应合理使用抗菌药。",
        lexicon=_lexicon(),
    )
    candidate_context = module.extract_graph_runtime_constraints(
        "应合理使用抗菌药。",
        "MPP指南.pdf",
        lexicon=_lexicon(),
    )

    assessment = module.assess_constraint_path(
        query,
        candidate_context,
        candidate_content,
        minimum_matched_constraint_types=2,
    )

    assert assessment["qualified"] is False
    assert assessment["reason"] == "broad_content_only"


def test_condition_plus_medication_class_requires_another_relation():
    module = _load_module()
    query = module.extract_graph_runtime_constraints(
        "MPP抗菌药治疗是否有依据？",
        lexicon=_lexicon(),
    )
    candidate = module.extract_graph_runtime_constraints(
        "MPP应合理使用抗菌药。",
        lexicon=_lexicon(),
    )

    assessment = module.assess_constraint_path(
        query,
        candidate,
        candidate,
        minimum_matched_constraint_types=2,
    )

    assert assessment["qualified"] is False
    assert assessment["reason"] == "condition_class_only"


def test_specific_condition_class_path_requires_explicit_opt_in():
    module = _load_module()
    query = module.extract_graph_runtime_constraints(
        "MPP糖皮质激素治疗是否有依据？",
        lexicon=_lexicon(),
    )
    candidate = module.extract_graph_runtime_constraints(
        "MPP可在限定情况下考虑糖皮质激素。",
        lexicon=_lexicon(),
    )

    default_assessment = module.assess_constraint_path(
        query,
        candidate,
        candidate,
        minimum_matched_constraint_types=2,
    )
    enabled_assessment = module.assess_constraint_path(
        query,
        candidate,
        candidate,
        minimum_matched_constraint_types=2,
        allow_specific_condition_class_path=True,
    )

    assert default_assessment["qualified"] is False
    assert default_assessment["reason"] == "condition_class_only"
    assert enabled_assessment["qualified"] is True
    assert enabled_assessment["reason"] == "qualified"


def test_generic_medication_class_and_topic_need_specific_anchor():
    module = _load_module()
    query = module.extract_graph_runtime_constraints(
        "抗菌药静脉给药适用边界是什么？",
        lexicon=_lexicon(),
    )
    candidate = module.extract_graph_runtime_constraints(
        "抗菌药静脉给药适用边界。",
        lexicon=_lexicon(),
    )

    assessment = module.assess_constraint_path(
        query,
        candidate,
        candidate,
        minimum_matched_constraint_types=2,
    )

    assert assessment["qualified"] is False
    assert assessment["reason"] == "missing_specific_anchor"


def test_real_lexicon_covers_verified_runtime_and_kb_aliases():
    import json

    module = _load_module()
    lexicon = json.loads(REAL_LEXICON_PATH.read_text(encoding="utf-8"))
    texts_and_expected = [
        (
            "年龄<1月龄的化脓性脑膜炎经验治疗。",
            {
                ("clinical_condition", "bacterial_meningitis"),
                ("population_specific", "under_1_month"),
            },
        ),
        (
            "8至17岁儿童使用红霉素。",
            {
                ("medication", "erythromycin"),
                ("population_specific", "eight_to_seventeen_years"),
            },
        ),
        (
            "3至11月龄可见co-amoxiclav剂量表。",
            {
                ("medication", "co_amoxiclav"),
                ("population_specific", "three_to_eleven_months"),
            },
        ),
        (
            "Phenazone with lidocaine for a maximum of 7 days.",
            {
                ("evidence_topic", "duration_restriction"),
                ("medication", "phenazone_lidocaine"),
            },
        ),
        (
            "2岁以下毛细支气管炎的典型年龄分布和诊断特征。",
            {
                ("clinical_condition", "bronchiolitis"),
                ("evidence_topic", "diagnostic_features"),
                ("population_specific", "under_2_years"),
            },
        ),
        (
            "儿科超说明书用药的治理原则。",
            {
                ("audit_domain", "pediatric_off_label_use"),
                ("evidence_topic", "governance_principle"),
            },
        ),
        (
            "第二代口服头孢菌素用于敏感菌轻症病例的一般范围。",
            {
                (
                    "medication_class",
                    "second_generation_oral_cephalosporin",
                ),
                ("evidence_topic", "mild_sensitive_infection_scope"),
            },
        ),
    ]

    for text, expected in texts_and_expected:
        constraints = module.extract_graph_runtime_constraints(
            text,
            lexicon=lexicon,
        )
        pairs = {
            (row["constraint_type"], row["normalized_value"])
            for row in constraints
        }
        assert expected <= pairs
