import json
from importlib import import_module
from pathlib import Path

import pytest


def _load_runner_module():
    try:
        return import_module(
            "experiments.phase6_evidence_graph.phase6b_smoke_runner"
        )
    except ModuleNotFoundError:
        pytest.fail("phase6b_smoke_runner module has not been implemented")


def test_smoke_runner_writes_versioned_artifacts_and_summary(tmp_path: Path):
    runner = _load_runner_module()
    repo_root = Path(__file__).resolve().parents[3]
    source_run = (
        repo_root
        / "revision/phase6/graph_runs/phase6a_batch_v0_1_20260719_002943"
    )
    output_dir = tmp_path / "phase6b_smoke"
    config = {
        "config_version": "phase6b-rerank-smoke3-v0.1",
        "method_id": "runtime_constraint_graph_reranking",
        "method_version": "phase6b-runtime-constraint-v0.1",
        "input_run_dir": str(source_run),
        "output_root": str(tmp_path),
        "sample_ids": [
            "PMSQA_DEV_001",
            "PMSQA_DEV_002",
            "PMSQA_DEV_003",
        ],
        "top_k": 4,
        "score_weights": {
            "relevance": 0.65,
            "authority": 0.20,
            "constraint_type_coverage": 0.15,
        },
        "constraint_type_weights": {
            "dose": 1.0,
            "frequency": 1.0,
            "route": 1.0,
            "monitoring_window": 1.0,
            "monitoring_trigger": 1.0,
            "monitoring_action": 1.0,
        },
    }

    summary = runner.run_smoke(config, output_dir=output_dir)

    assert summary["total_samples"] == 3
    assert summary["success_count"] == 2
    assert summary["boundary_refusal_count"] == 1
    assert summary["failed_count"] == 0
    assert summary["ranking_applicable_count"] == 2
    assert summary["ranking_not_applicable_count"] == 1
    assert (
        summary["top1_changed_count"] + summary["top1_unchanged_count"]
        == summary["ranking_applicable_count"]
    )
    assert summary["external_model_calls"] == 0
    assert summary["estimated_cost"] == 0
    assert len(list((output_dir / "method_artifacts").glob("*.json"))) == 3

    run_manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["config_version"] == config["config_version"]
    assert run_manifest["sample_ids"] == config["sample_ids"]
    assert run_manifest["input_run_dir"] == str(source_run.resolve())

    validations = [
        json.loads(line)
        for line in (output_dir / "validations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert len(validations) == 3
    assert all(row["deterministic"] for row in validations)
    assert all(row["parent_graph_unchanged"] for row in validations)
    assert all(row["gold_leakage_check"] == "passed" for row in validations)

    ranking_diagnostics = [
        json.loads(line)
        for line in (output_dir / "ranking_diagnostics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert len(ranking_diagnostics) == 3
    assert {
        row["sample_id"]: row["ranking_status"] for row in ranking_diagnostics
    }["PMSQA_DEV_003"] == "not_applicable"
    assert all("gold" not in json.dumps(row) for row in ranking_diagnostics)
