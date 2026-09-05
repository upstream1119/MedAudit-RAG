import json
from pathlib import Path


PHASE7_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PHASE7_DIR
    / "configs"
    / "validation40_graph_consistency_preregistration_v0_1.json"
)


def _load_contract() -> dict:
    assert CONFIG_PATH.exists(), "G3 preregistration contract is missing"
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_preregistration_locks_the_frozen_g2_inputs() -> None:
    contract = _load_contract()

    assert contract["phase"] == "Phase 7-C1c-4e-3c-0"
    assert contract["dataset_version"] == (
        "benchmark-v1.0-guideline-grounded-author-adjudicated"
    )
    assert contract["kb_version"] == "KB-medium-v1"
    assert contract["input_method"] == (
        "g2_v0_1_symmetric_graph_path_reranker_dedup"
    )
    assert contract["expected_count"] == 40
    assert contract["candidate_budget"] == 24
    assert contract["final_evidence_k"] == 4
    assert contract["expected_input_results_sha256"] == (
        "02a0d92877fbf34db431886e090700e02f18e05d40403d361d4c8652fcea939f"
    )
    assert contract["expected_input_manifest_sha256"] == (
        "85e802b2e609083778ead8565f2f68e2978cb7b7e90017d3be390067888027ad"
    )
    assert contract["expected_g2_config_sha256"] == (
        "9cbbf9b458da42a45d25ee606d4f848812e072745147c56b14bedbf5ce332312"
    )
    assert contract["expected_lexicon_sha256"] == (
        "6e251c97a131b2304bc237effbb7f7c8e0e4ed3548d9e7d8882c1b7c7deef08a"
    )


def test_preregistration_keeps_g3_audit_only() -> None:
    contract = _load_contract()
    policy = contract["mutation_policy"]

    assert contract["audit_mode"] == "annotate_only"
    assert contract["audit_scope"] == "final_evidence_top4"
    assert policy == {
        "allow_candidate_membership_change": False,
        "allow_candidate_order_change": False,
        "allow_final_evidence_membership_change": False,
        "allow_final_evidence_order_change": False,
    }
    assert contract["freeze_rule"]["retrieval_metrics_must_remain_identical"] is True


def test_preregistration_separates_correction_from_evidence_conflict() -> None:
    contract = _load_contract()
    labels = contract["audit_labels"]

    assert set(labels) == {
        "supported_match",
        "corrective_value_mismatch",
        "evidence_evidence_conflict",
        "coverage_gap",
        "scope_mismatch",
        "not_comparable",
    }
    assert labels["corrective_value_mismatch"]["category"] == "query_evidence"
    assert labels["corrective_value_mismatch"]["material_conflict"] is False
    assert labels["evidence_evidence_conflict"]["category"] == (
        "evidence_evidence"
    )
    assert labels["evidence_evidence_conflict"]["material_conflict"] is True
    assert labels["not_comparable"]["material_conflict"] is False
    assert contract["scope_compatibility"]["shared_strong_anchor_required"] is True


def test_preregistration_freezes_actions_guards_and_acceptance_boundary() -> None:
    contract = _load_contract()

    assert contract["route_actions"] == [
        "allow_supported_answer",
        "allow_corrective_answer",
        "review_required",
        "insufficient_evidence",
        "boundary_refusal_passthrough",
    ]
    assert contract["route_precedence"] == [
        "boundary_refusal_passthrough",
        "review_required",
        "insufficient_evidence",
        "allow_corrective_answer",
        "allow_supported_answer",
    ]
    assert contract["freeze_rule"] == {
        "expected_sample_count": 40,
        "input_identity_must_match": True,
        "retrieval_metrics_must_remain_identical": True,
        "audit_trace_completeness_must_equal_one": True,
        "independent_run_core_hashes_must_match": True,
        "all_non_allow_samples_require_manual_adjudication": True,
        "retrieval_gain_claimed": False,
        "safety_gain_claimed": False,
    }
    assert contract["execution_guards"] == {
        "validation40_only": True,
        "gold_access": False,
        "pilot_test_content_access": False,
        "external_model_calls": False,
        "clinical_validation_claimed": False,
    }
    assert contract["expected_pilot_test_sha256"] == (
        "14afa2988d9ff579471f2d082f9803ce3bfc9e030eb439e7cf2d4c6fa55d5da9"
    )
