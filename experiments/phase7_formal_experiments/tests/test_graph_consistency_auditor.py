import importlib


def _module():
    return importlib.import_module(
        "experiments.phase7_formal_experiments.graph_consistency_auditor"
    )


def _lexicon() -> dict:
    return {
        "lexicon_version": "fixture-v0.1",
        "entries": [
            {
                "constraint_type": "clinical_condition",
                "normalized_value": "mycoplasma_pneumonia",
                "aliases": ["支原体肺炎"],
                "strong_anchor": True,
            },
            {
                "constraint_type": "medication",
                "normalized_value": "azithromycin",
                "aliases": ["阿奇霉素"],
                "strong_anchor": True,
            },
            {
                "constraint_type": "medication",
                "normalized_value": "ceftriaxone",
                "aliases": ["头孢曲松"],
                "strong_anchor": True,
            },
            {
                "constraint_type": "population_context",
                "normalized_value": "pediatric",
                "aliases": ["儿童"],
                "strong_anchor": False,
            },
            {
                "constraint_type": "population_context",
                "normalized_value": "adult",
                "aliases": ["成人"],
                "strong_anchor": False,
            },
        ],
    }


def _evidence(key: str, content: str) -> dict:
    return {
        "candidate_key": key,
        "source_file": "guideline.pdf",
        "page_number": 10,
        "chapter_title": "支原体肺炎治疗",
        "content": content,
    }


def _contract() -> dict:
    return {
        "exclusive_constraint_types": [
            "dose",
            "frequency",
            "route",
            "monitoring_window",
            "monitoring_action",
            "contraindication_action",
            "evidence_scope",
        ],
        "high_risk_coverage_types": [
            "dose",
            "frequency",
            "route",
            "contraindication_check",
            "contraindication_action",
            "drug_interaction",
            "patient_weight",
        ],
        "scope_compatibility": {
            "shared_strong_anchor_required": True,
            "strong_anchor_types": ["medication", "medication_class"],
            "compatible_context_types": [
                "clinical_condition",
                "population_context",
                "evidence_scope",
            ],
            "explicitly_disjoint_context_is_conflict": True,
            "missing_scope_is_not_comparable": True,
        },
        "route_precedence": [
            "boundary_refusal_passthrough",
            "review_required",
            "insufficient_evidence",
            "allow_corrective_answer",
            "allow_supported_answer",
        ],
    }


def test_supported_constraints_allow_a_supported_answer() -> None:
    audit = _module().audit_graph_consistency(
        "儿童支原体肺炎阿奇霉素静脉滴注10mg/kg，一天一次可以吗？",
        evidence_top4=[
            _evidence(
                "e1",
                "儿童支原体肺炎可使用阿奇霉素静脉滴注10mg/kg，每日一次。",
            )
        ],
        lexicon=_lexicon(),
        contract=_contract(),
    )

    assert "supported_match" in audit["summary_labels"]
    assert "corrective_value_mismatch" not in audit["summary_labels"]
    assert audit["route_action"] == "allow_supported_answer"


def test_query_value_mismatch_is_a_correction_not_evidence_conflict() -> None:
    audit = _module().audit_graph_consistency(
        "儿童支原体肺炎阿奇霉素静脉滴注10mg/kg，一天两次可以吗？",
        evidence_top4=[
            _evidence(
                "e1",
                "儿童支原体肺炎可使用阿奇霉素静脉滴注10mg/kg，每日一次。",
            )
        ],
        lexicon=_lexicon(),
        contract=_contract(),
    )

    assert "corrective_value_mismatch" in audit["summary_labels"]
    assert "evidence_evidence_conflict" not in audit["summary_labels"]
    assert audit["route_action"] == "allow_corrective_answer"


def test_comparable_evidence_value_conflict_requires_review() -> None:
    audit = _module().audit_graph_consistency(
        "儿童支原体肺炎阿奇霉素的给药频次是什么？",
        evidence_top4=[
            _evidence("e1", "儿童支原体肺炎阿奇霉素每日一次。"),
            _evidence("e2", "儿童支原体肺炎阿奇霉素每日两次。"),
        ],
        lexicon=_lexicon(),
        contract=_contract(),
    )

    assert "evidence_evidence_conflict" in audit["summary_labels"]
    assert audit["route_action"] == "review_required"
    assert audit["pairwise_comparisons"][0]["labels"] == [
        "evidence_evidence_conflict"
    ]


def test_disjoint_population_scope_requires_review() -> None:
    audit = _module().audit_graph_consistency(
        "儿童支原体肺炎阿奇霉素每天一次可以吗？",
        evidence_top4=[
            _evidence("e1", "成人支原体肺炎阿奇霉素每日一次。")
        ],
        lexicon=_lexicon(),
        contract=_contract(),
    )

    assert "scope_mismatch" in audit["summary_labels"]
    assert audit["route_action"] == "review_required"


def test_missing_requested_high_risk_constraint_is_a_coverage_gap() -> None:
    audit = _module().audit_graph_consistency(
        "体重20kg，儿童支原体肺炎使用阿奇霉素时如何考虑体重？",
        evidence_top4=[
            _evidence("e1", "儿童支原体肺炎可考虑使用阿奇霉素。")
        ],
        lexicon=_lexicon(),
        contract=_contract(),
    )

    assert "coverage_gap" in audit["summary_labels"]
    assert audit["route_action"] == "insufficient_evidence"


def test_different_medications_are_not_forced_into_a_conflict() -> None:
    audit = _module().audit_graph_consistency(
        "儿童支原体肺炎阿奇霉素每天一次可以吗？",
        evidence_top4=[
            _evidence("e1", "儿童支原体肺炎阿奇霉素每日一次。"),
            _evidence("e2", "儿童支原体肺炎头孢曲松每日两次。"),
        ],
        lexicon=_lexicon(),
        contract=_contract(),
    )

    assert "not_comparable" in audit["summary_labels"]
    assert "evidence_evidence_conflict" not in audit["summary_labels"]
    assert audit["route_action"] == "allow_supported_answer"


def test_existing_boundary_refusal_has_highest_route_precedence() -> None:
    action, reasons = _module().resolve_route_action(
        ["evidence_evidence_conflict", "coverage_gap"],
        upstream_boundary_refusal=True,
    )

    assert action == "boundary_refusal_passthrough"
    assert reasons == ["existing_upstream_boundary_refusal"]
