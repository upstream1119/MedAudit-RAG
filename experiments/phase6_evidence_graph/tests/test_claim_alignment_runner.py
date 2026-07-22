import json
from importlib import import_module
from pathlib import Path

import pytest


def _load_module():
    try:
        return import_module(
            "experiments.phase6_evidence_graph.claim_alignment_runner"
        )
    except ModuleNotFoundError:
        pytest.fail("claim_alignment_runner module has not been implemented")


def _artifact(sample_id: str, *, with_evidence: bool) -> dict:
    evidence = []
    status = "insufficient_graph_evidence"
    if with_evidence:
        status = "success"
        evidence = [
            {
                "evidence_id": "evidence::cap::p26",
                "source_id": "SRC-002",
                "source_file": "儿童社区获得性肺炎诊疗规范（2019年版）.pdf",
                "page_number": 26,
                "content": "治疗48-72小时症状无改善，应再次评估。",
                "runtime_constraints": [
                    {
                        "constraint_type": "monitoring_window",
                        "normalized_value": "48-72h",
                        "surface_forms": ["48-72小时"],
                        "ruleset_version": "phase6b-runtime-constraint-rules-v0.3",
                    },
                    {
                        "constraint_type": "monitoring_trigger",
                        "normalized_value": "nonresponse",
                        "surface_forms": ["症状无改善"],
                        "ruleset_version": "phase6b-runtime-constraint-rules-v0.3",
                    },
                    {
                        "constraint_type": "monitoring_action",
                        "normalized_value": "reassess",
                        "surface_forms": ["再次评估"],
                        "ruleset_version": "phase6b-runtime-constraint-rules-v0.3",
                    },
                ],
            }
        ]
    return {
        "artifact_schema_version": "phase6b-reranking-artifact-v0.2",
        "artifact_type": "phase6b_reranking_artifact",
        "artifact_status": status,
        "artifact_sha256": f"artifact::{sample_id}",
        "sample_id": sample_id,
        "parent_inference_graph_id": f"inference_graph::{sample_id}",
        "parent_inference_graph_sha256": f"graph::{sample_id}",
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


def test_runner_writes_versioned_artifacts_summary_and_validations(
    tmp_path: Path,
):
    runner = _load_module()
    input_dir = tmp_path / "reranking_run"
    artifact_dir = input_dir / "method_artifacts"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "PMSQA_DEV_001.json").write_text(
        json.dumps(
            _artifact("PMSQA_DEV_001", with_evidence=True),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "PMSQA_DEV_006.json").write_text(
        json.dumps(
            _artifact("PMSQA_DEV_006", with_evidence=False),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "claim_alignment_run"
    config = {
        "config_version": "phase6b-claim-alignment-test-v0.1",
        "method_id": "constraint_grounded_claim_evidence_alignment",
        "method_version": "phase6b-claim-alignment-v0.1",
        "input_run_dir": str(input_dir),
        "output_root": str(tmp_path),
        "sample_ids": ["PMSQA_DEV_001", "PMSQA_DEV_006"],
        "evidence_budget": 4,
        "candidate_output_origin": "deterministic_development_fixture",
        "candidate_outputs": {
            "PMSQA_DEV_001": "治疗48-72小时症状无改善，应再次评估。",
            "PMSQA_DEV_006": "小儿支气管肺炎可以静脉使用氨溴索。",
        },
    }

    summary = runner.run_batch(config, output_dir=output_dir)

    assert summary["total_samples"] == 2
    assert summary["supported_count"] == 1
    assert summary["insufficient_evidence_count"] == 1
    assert summary["failed_count"] == 0
    assert summary["external_model_calls"] == 0
    assert summary["estimated_cost"] == 0
    assert len(list((output_dir / "claim_alignment_artifacts").glob("*.json"))) == 2

    manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["config_version"] == config["config_version"]
    assert manifest["candidate_output_origin"] == (
        "deterministic_development_fixture"
    )
    assert manifest["sample_ids"] == config["sample_ids"]

    validations = [
        json.loads(line)
        for line in (output_dir / "validations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert len(validations) == 2
    assert all(row["deterministic"] for row in validations)
    assert all(row["parent_artifact_unchanged"] for row in validations)
    assert all(row["gold_leakage_check"] == "passed" for row in validations)


def test_runner_writes_claim_level_failure_analysis(tmp_path: Path):
    runner = _load_module()
    input_dir = tmp_path / "reranking_run"
    artifact_dir = input_dir / "method_artifacts"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "PMSQA_DEV_001.json").write_text(
        json.dumps(
            _artifact("PMSQA_DEV_001", with_evidence=True),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "PMSQA_DEV_006.json").write_text(
        json.dumps(
            _artifact("PMSQA_DEV_006", with_evidence=False),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "claim_alignment_run"
    config = {
        "config_version": "phase6b-claim-alignment-test-v0.1",
        "method_id": "constraint_grounded_claim_evidence_alignment",
        "method_version": "phase6b-claim-alignment-v0.1",
        "input_run_dir": str(input_dir),
        "output_root": str(tmp_path),
        "sample_ids": ["PMSQA_DEV_001", "PMSQA_DEV_006"],
        "evidence_budget": 4,
        "candidate_output_origin": "deterministic_development_fixture",
        "candidate_outputs": {
            "PMSQA_DEV_001": "治疗48-72小时症状无改善，应再次评估。",
            "PMSQA_DEV_006": "小儿支气管肺炎可以静脉使用氨溴索。",
        },
    }

    summary = runner.run_batch(config, output_dir=output_dir)

    assert summary["total_claim_count"] == 2
    assert summary["claim_state_counts"] == {
        "supported": 1,
        "contradicted": 0,
        "unsupported": 0,
        "insufficient_evidence": 1,
    }
    assert summary["reason_code_counts"] == {
        "all_runtime_constraints_supported": 1,
        "parent_has_no_admitted_evidence": 1,
    }
    assert summary["source_binding_status_counts"] == {
        "not_explicitly_named": 2
    }

    claim_rows = [
        json.loads(line)
        for line in (output_dir / "claim_audit_rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert len(claim_rows) == 2
    assert claim_rows[0]["sample_id"] == "PMSQA_DEV_001"
    assert claim_rows[0]["claim_index"] == 1
    assert claim_rows[0]["support_state"] == "supported"
    assert claim_rows[0]["supporting_evidence_count"] == 1
    assert claim_rows[1]["sample_id"] == "PMSQA_DEV_006"
    assert claim_rows[1]["support_state"] == "insufficient_evidence"
    assert claim_rows[1]["reason_codes"] == [
        "parent_has_no_admitted_evidence"
    ]
