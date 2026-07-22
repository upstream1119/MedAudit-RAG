from importlib import import_module

import pytest


def _load_extractor_module():
    try:
        return import_module(
            "experiments.phase6_evidence_graph.runtime_constraint_extractor"
        )
    except ModuleNotFoundError:
        pytest.fail("runtime_constraint_extractor module has not been implemented")


def _values_by_type(constraints: list[dict]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    for constraint in constraints:
        values.setdefault(constraint["constraint_type"], set()).add(
            constraint["normalized_value"]
        )
    return values


def test_extracts_frequency_route_and_dose_deterministically():
    extractor = _load_extractor_module()
    text = "儿童支原体肺炎阿奇霉素静滴 10mg/kg，一天两次可以吗？"

    first = extractor.extract_runtime_constraints(text)
    second = extractor.extract_runtime_constraints(text)

    assert first == second
    values = _values_by_type(first)
    assert values["dose"] == {"10mg/kg"}
    assert values["frequency"] == {"bid"}
    assert values["route"] == {"iv_infusion"}


def test_normalizes_guideline_frequency_route_and_daily_dose():
    extractor = _load_extractor_module()

    constraints = extractor.extract_runtime_constraints(
        "重症推荐阿奇霉素静点，10 mg/(kg.d)，qd。"
    )

    values = _values_by_type(constraints)
    assert values["dose"] == {"10mg/kg/day"}
    assert values["frequency"] == {"qd"}
    assert values["route"] == {"iv_infusion"}


def test_extracts_monitoring_window_trigger_and_action():
    extractor = _load_extractor_module()

    constraints = extractor.extract_runtime_constraints(
        "所有患者经48～72小时治疗症状无改善，应再次进行临床或实验室评估。"
    )

    values = _values_by_type(constraints)
    assert values["monitoring_window"] == {"48-72h"}
    assert values["monitoring_trigger"] == {"nonresponse"}
    assert values["monitoring_action"] == {"reassess"}


def test_preserves_distinct_frequency_values():
    extractor = _load_extractor_module()

    once = _values_by_type(
        extractor.extract_runtime_constraints("每日一次，qd。")
    )
    twice = _values_by_type(
        extractor.extract_runtime_constraints("一天两次，bid。")
    )

    assert once["frequency"] == {"qd"}
    assert twice["frequency"] == {"bid"}
    assert once["frequency"] != twice["frequency"]


def test_extracts_family_specific_combination_constraints():
    extractor = _load_extractor_module()

    antimicrobial = _values_by_type(
        extractor.extract_runtime_constraints(
            "儿童肺炎能否同时使用阿奇霉素和头孢？"
        )
    )
    antipyretic = _values_by_type(
        extractor.extract_runtime_constraints(
            "儿童发热可以同时吃布洛芬和对乙酰氨基酚吗？"
        )
    )
    cough_medicine = _values_by_type(
        extractor.extract_runtime_constraints(
            "儿童咳嗽有痰，能否同时口服止咳药和化痰药？"
        )
    )

    assert antimicrobial["combination_antimicrobial"] == {"coadministration"}
    assert antipyretic["combination_antipyretic"] == {"coadministration"}
    assert cough_medicine["combination_cough_medicine"] == {"coadministration"}
    assert "combination_antimicrobial" not in cough_medicine


def test_extracts_age_weight_allergy_route_and_monitoring_targets():
    extractor = _load_extractor_module()

    values = _values_by_type(
        extractor.extract_runtime_constraints(
            "1岁婴儿体重10kg，静脉使用药物前需要关注过敏史和肝肾功能。"
        )
    )

    assert values["age_group"] == {"infant"}
    assert values["patient_weight"] == {"10kg"}
    assert values["route"] == {"iv_unspecified"}
    assert values["contraindication_check"] == {"allergy_history"}
    assert values["monitoring_target"] == {"hepatic_renal_function"}


def test_extracts_single_monitoring_window_and_effect_assessment():
    extractor = _load_extractor_module()

    values = _values_by_type(
        extractor.extract_runtime_constraints(
            "阿奇霉素治疗72小时后体温仍高，需要评价药物疗效。"
        )
    )

    assert values["monitoring_window"] == {"72h"}
    assert values["monitoring_trigger"] == {"persistent_fever"}
    assert values["monitoring_action"] == {"assess_effect"}


def test_extracts_dosage_scope_population_and_adjustment_constraints():
    extractor = _load_extractor_module()

    question_values = _values_by_type(
        extractor.extract_runtime_constraints(
            "免疫缺陷儿童可以把基本药物目录作为阿奇霉素剂量依据吗？"
        )
    )
    evidence_values = _values_by_type(
        extractor.extract_runtime_constraints(
            "推荐剂量为10mg/(kg.d)，不应只因治疗无效而加大剂量。"
        )
    )

    assert question_values["population_context"] == {"immunodeficiency"}
    assert question_values["evidence_scope"] == {"dose_guidance"}
    assert evidence_values["evidence_scope"] == {"dose_guidance"}
    assert evidence_values["dose_adjustment"] == {"increase"}


def test_extracts_formulation_listing_as_a_distinct_evidence_scope():
    extractor = _load_extractor_module()

    values = _values_by_type(
        extractor.extract_runtime_constraints(
            "品种名称为阿奇霉素，剂型、规格包括片剂、胶囊。"
        )
    )

    assert values["evidence_scope"] == {"formulation_listing"}


def test_extracts_caution_body_temperature_and_pathogen_coverage():
    extractor = _load_extractor_module()

    caution = _values_by_type(
        extractor.extract_runtime_constraints(
            "对婴幼儿，静脉制剂的使用尤其需要慎重。"
        )
    )
    temperature = _values_by_type(
        extractor.extract_runtime_constraints(
            "治疗72小时后根据体温情况评价药物疗效。"
        )
    )
    coverage = _values_by_type(
        extractor.extract_runtime_constraints(
            "免疫缺陷患者应广覆 盖可能病原体。"
        )
    )

    assert caution["contraindication_action"] == {"caution"}
    assert temperature["monitoring_target"] == {"body_temperature"}
    assert coverage["coverage_action"] == {"broaden_pathogen_coverage"}

    contraindication_actions = _values_by_type(
        extractor.extract_runtime_constraints(
            "对相关药物过敏时应禁用或慎用。"
        )
    )
    assert contraindication_actions["contraindication_action"] == {
        "avoid",
        "caution",
    }


def test_extracts_each_family_from_a_multimodal_combination_claim():
    extractor = _load_extractor_module()

    values = _values_by_type(
        extractor.extract_runtime_constraints(
            "现有证据不足以支持阿奇霉素、头孢、激素和雾化同时使用。"
        )
    )

    assert values["combination_antimicrobial"] == {"coadministration"}
    assert values["combination_steroid"] == {"coadministration"}
    assert values["combination_nebulization"] == {"coadministration"}


def test_does_not_join_distant_combination_families_across_a_long_chunk():
    extractor = _load_extractor_module()

    values = _values_by_type(
        extractor.extract_runtime_constraints(
            "特定感染可以联合使用两种抗菌药物。"
            + "其他说明" * 60
            + "应用肾上腺皮质激素的患者需要另行评估。"
        )
    )

    assert values["combination_antimicrobial"] == {"coadministration"}
    assert "combination_steroid" not in values
