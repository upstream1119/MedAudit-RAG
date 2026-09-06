import hashlib
import importlib
import json
from copy import deepcopy
from pathlib import Path

import pytest


PHASE7_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PHASE7_DIR
    / "configs"
    / "validation40_graph_consistency_audit_v0_1.json"
)


def _module():
    return importlib.import_module(
        "experiments.phase7_formal_experiments.validation40_graph_consistency_audit"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[dict, Path]:
    candidate = {
        "candidate_key": "context_512::fixture",
        "source_file": "guideline.pdf",
        "page_number": 10,
        "chapter_title": "支原体肺炎治疗",
        "content": "儿童支原体肺炎可考虑使用阿奇霉素。",
    }
    g2_method = "g2"
    results_path = tmp_path / "input" / "results.jsonl"
    _write_jsonl(
        results_path,
        [
            {
                "sample_id": "S1",
                "question": "体重20kg，儿童支原体肺炎使用阿奇霉素时如何考虑体重？",
                "methods": {
                    g2_method: {
                        "candidates_top24": [candidate],
                        "evidence_top4": [candidate],
                        "dedup_audit": {},
                    }
                },
            }
        ],
    )
    input_manifest_path = results_path.parent / "manifest.json"
    _write_json(
        input_manifest_path,
        {
            "ready": True,
            "files": {
                "results": {
                    "path": results_path.name,
                    "sha256": _sha256(results_path),
                }
            },
        },
    )
    g2_config_path = tmp_path / "input" / "g2_config.json"
    _write_json(g2_config_path, {"config_version": "g2-fixture"})
    lexicon_path = tmp_path / "input" / "lexicon.json"
    _write_json(
        lexicon_path,
        {
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
                    "constraint_type": "population_context",
                    "normalized_value": "pediatric",
                    "aliases": ["儿童"],
                    "strong_anchor": False,
                },
            ],
        },
    )
    pilot_path = tmp_path / "input" / "pilot.bin"
    pilot_path.write_bytes(b"\xff\xfePILOT_HASH_ONLY")
    prereg_path = tmp_path / "prereg.json"
    prereg = {
        "config_version": "fixture-prereg-v0.1",
        "phase": "Phase 7-C1c-4e-3c-0",
        "dataset_version": "fixture-dataset",
        "kb_version": "fixture-kb",
        "input_results_path": str(results_path.relative_to(tmp_path)),
        "input_manifest_path": str(input_manifest_path.relative_to(tmp_path)),
        "g2_config_path": str(g2_config_path.relative_to(tmp_path)),
        "lexicon_path": str(lexicon_path.relative_to(tmp_path)),
        "pilot_test_path": str(pilot_path.relative_to(tmp_path)),
        "expected_input_results_sha256": _sha256(results_path),
        "expected_input_manifest_sha256": _sha256(input_manifest_path),
        "expected_g2_config_sha256": _sha256(g2_config_path),
        "expected_lexicon_sha256": _sha256(lexicon_path),
        "expected_pilot_test_sha256": _sha256(pilot_path),
        "expected_count": 1,
        "candidate_budget": 24,
        "candidate_output_field": "candidates_top24",
        "final_evidence_k": 4,
        "final_evidence_field": "evidence_top4",
        "input_method": g2_method,
        "planned_output_method": "g3",
        "audit_mode": "annotate_only",
        "audit_scope": "final_evidence_top4",
        "mutation_policy": {
            "allow_candidate_membership_change": False,
            "allow_candidate_order_change": False,
            "allow_final_evidence_membership_change": False,
            "allow_final_evidence_order_change": False,
        },
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
        "required_trace_fields": [
            "sample_id",
            "input_method",
            "query_constraints",
            "evidence_constraints",
            "pairwise_comparisons",
            "summary_labels",
            "route_action",
            "route_reasons",
            "candidate_keys",
            "source_pages",
            "input_identity_sha256",
        ],
        "freeze_rule": {
            "expected_sample_count": 1,
            "input_identity_must_match": True,
            "retrieval_metrics_must_remain_identical": True,
            "audit_trace_completeness_must_equal_one": True,
            "independent_run_core_hashes_must_match": True,
            "all_non_allow_samples_require_manual_adjudication": True,
            "retrieval_gain_claimed": False,
            "safety_gain_claimed": False,
        },
        "execution_guards": {
            "validation40_only": True,
            "gold_access": False,
            "pilot_test_content_access": False,
            "external_model_calls": False,
            "clinical_validation_claimed": False,
        },
    }
    _write_json(prereg_path, prereg)
    execution = {
        "audit_version": "fixture-audit-v0.1",
        "config_version": "fixture-execution-v0.1",
        "manifest_version": "fixture-manifest-v0.1",
        "phase": "Phase 7-C1c-4e-3c-1",
        "preregistration_path": str(prereg_path.relative_to(tmp_path)),
        "expected_preregistration_sha256": _sha256(prereg_path),
        "output_dir": "output",
        "results_filename": "results.jsonl",
        "audit_filename": "audit.json",
        "manual_queue_filename": "manual_queue.jsonl",
        "manifest_filename": "manifest.json",
    }
    return execution, prereg_path


def test_runner_preserves_g2_evidence_and_queues_non_allow_rows(tmp_path) -> None:
    execution, _ = _fixture(tmp_path)
    output_dir = tmp_path / "run_a"

    audit = _module().run_graph_consistency_audit(
        repo_root=tmp_path,
        output_dir=output_dir,
        config=execution,
    )

    row = json.loads((output_dir / "results.jsonl").read_text(encoding="utf-8"))
    g2 = row["methods"]["g2"]
    g3 = row["methods"]["g3"]
    assert g3["candidates_top24"] == g2["candidates_top24"]
    assert g3["evidence_top4"] == g2["evidence_top4"]
    assert g3["graph_consistency_audit"]["route_action"] == (
        "insufficient_evidence"
    )
    assert audit["trace_complete_count"] == 1
    assert audit["manual_adjudication_count"] == 1
    assert len((output_dir / "manual_queue.jsonl").read_text().splitlines()) == 1


def test_runner_is_byte_deterministic_across_two_empty_output_dirs(tmp_path) -> None:
    execution, _ = _fixture(tmp_path)
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"

    _module().run_graph_consistency_audit(
        repo_root=tmp_path,
        output_dir=run_a,
        config=execution,
    )
    _module().run_graph_consistency_audit(
        repo_root=tmp_path,
        output_dir=run_b,
        config=execution,
    )

    for filename in (
        "results.jsonl",
        "audit.json",
        "manual_queue.jsonl",
        "manifest.json",
    ):
        assert (run_a / filename).read_bytes() == (run_b / filename).read_bytes()


def test_runner_fails_closed_when_a_frozen_input_hash_changes(tmp_path) -> None:
    execution, prereg_path = _fixture(tmp_path)
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    input_path = tmp_path / prereg["input_results_path"]
    input_path.write_text(input_path.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="input results SHA-256 mismatch"):
        _module().run_graph_consistency_audit(
            repo_root=tmp_path,
            output_dir=tmp_path / "output",
            config=execution,
        )


def test_runner_rejects_gold_only_content_even_when_hashes_match(tmp_path) -> None:
    execution, prereg_path = _fixture(tmp_path)
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    input_path = tmp_path / prereg["input_results_path"]
    row = json.loads(input_path.read_text(encoding="utf-8"))
    row["forbidden_claims"] = ["unsafe claim"]
    _write_jsonl(input_path, [row])
    manifest_path = tmp_path / prereg["input_manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["results"]["sha256"] = _sha256(input_path)
    _write_json(manifest_path, manifest)
    prereg["expected_input_results_sha256"] = _sha256(input_path)
    prereg["expected_input_manifest_sha256"] = _sha256(manifest_path)
    _write_json(prereg_path, prereg)
    execution["expected_preregistration_sha256"] = _sha256(prereg_path)

    with pytest.raises(ValueError, match="gold-only key"):
        _module().run_graph_consistency_audit(
            repo_root=tmp_path,
            output_dir=tmp_path / "output",
            config=execution,
        )


def test_runner_fails_closed_if_auditor_mutates_frozen_evidence(
    tmp_path, monkeypatch
) -> None:
    execution, _ = _fixture(tmp_path)
    module = _module()
    original_auditor = module.audit_graph_consistency

    def mutating_auditor(question, *, evidence_top4, **kwargs):
        evidence_top4[0]["content"] = "mutated evidence"
        return original_auditor(
            question,
            evidence_top4=evidence_top4,
            **kwargs,
        )

    monkeypatch.setattr(module, "audit_graph_consistency", mutating_auditor)

    with pytest.raises(RuntimeError, match="mutated frozen G2 evidence"):
        module.run_graph_consistency_audit(
            repo_root=tmp_path,
            output_dir=tmp_path / "output",
            config=execution,
        )


def test_repository_execution_config_binds_the_preregistered_contract() -> None:
    assert CONFIG_PATH.exists(), "G3 execution configuration is missing"
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["phase"] == "Phase 7-C1c-4e-3c-1"
    assert config["preregistration_path"].endswith(
        "validation40_graph_consistency_preregistration_v0_1.json"
    )
    assert config["expected_preregistration_sha256"] == (
        "c969327c4d7df1d9932e13617c847bffca37c6ffe8528d8d1b8b50d5a595da61"
    )
    assert config["output_dir"].endswith(
        "validation40_g3_graph_consistency_v0_1/run_a"
    )
