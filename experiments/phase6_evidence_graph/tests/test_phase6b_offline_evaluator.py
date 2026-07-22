import json
from importlib import import_module
from pathlib import Path

import pytest


def _load_evaluator_module():
    try:
        return import_module(
            "experiments.phase6_evidence_graph.phase6b_offline_evaluator"
        )
    except ModuleNotFoundError:
        pytest.fail("phase6b_offline_evaluator module has not been implemented")


def _artifact(
    *,
    sample_id: str = "PMSQA_DEV_001",
    vector_ids: list[str],
    graph_ids: list[str],
    parent_sha: str = "parent-sha",
    artifact_status: str = "success",
    rerank_applied: bool = True,
) -> dict:
    return {
        "artifact_type": "phase6b_reranking_artifact",
        "artifact_schema_version": "phase6b-reranking-artifact-v0.2",
        "artifact_status": artifact_status,
        "sample_id": sample_id,
        "parent_inference_graph_id": f"inference_graph::{sample_id}",
        "parent_inference_graph_sha256": parent_sha,
        "artifact_sha256": f"artifact-sha::{sample_id}",
        "rerank_applied": rerank_applied,
        "rerank_skip_reason": (
            None if rerank_applied else "no_runtime_query_constraints"
        ),
        "ranking_baseline": {
            "vector_top_k_evidence_ids": vector_ids,
            "graph_top_k_evidence_ids": graph_ids,
            "top1_changed": bool(vector_ids and graph_ids and vector_ids[0] != graph_ids[0]),
            "moved_in_evidence_ids": sorted(set(graph_ids) - set(vector_ids)),
            "moved_out_evidence_ids": sorted(set(vector_ids) - set(graph_ids)),
        },
        "ranked_evidence": [],
    }


def _evaluation_graph(
    *,
    sample_id: str = "PMSQA_DEV_001",
    parent_sha: str = "parent-sha",
    expected_decision: str = "answer",
    gold_status: str = "page_span_located",
) -> dict:
    nodes = [
        {
            "id": "runtime-wrong",
            "type": "EvidenceSpan",
            "properties": {"source_id": "SRC-002", "page_number": 12},
            "provenance": "runtime_retrieval",
        },
        {
            "id": "runtime-gold",
            "type": "EvidenceSpan",
            "properties": {"source_id": "SRC-002", "page_number": 26},
            "provenance": "runtime_retrieval",
        },
        {
            "id": f"decision::{sample_id}::expected::{expected_decision}",
            "type": "Decision",
            "properties": {"decision_type": expected_decision},
            "provenance": "benchmark_annotation",
        },
    ]
    if gold_status == "page_span_located":
        nodes.append(
            {
                "id": "gold-evidence",
                "type": "EvidenceSpan",
                "properties": {"source_id": "SRC-002", "page_number": 26},
                "provenance": "gold_evidence",
            }
        )
    return {
        "graph_type": "evaluation_graph",
        "sample_id": sample_id,
        "inference_graph_id": f"inference_graph::{sample_id}",
        "inference_graph_sha256": parent_sha,
        "gold_evidence_status": gold_status,
        "nodes": nodes,
    }


@pytest.mark.parametrize(
    ("vector_ids", "graph_ids", "expected_outcome"),
    [
        (["runtime-wrong", "runtime-gold"], ["runtime-gold", "runtime-wrong"], "improved"),
        (["runtime-gold", "runtime-wrong"], ["runtime-gold", "runtime-wrong"], "unchanged"),
        (["runtime-gold", "runtime-wrong"], ["runtime-wrong", "runtime-gold"], "worsened"),
    ],
)
def test_evaluate_sample_compares_gold_page_rank_after_runtime_is_frozen(
    vector_ids,
    graph_ids,
    expected_outcome,
):
    evaluator = _load_evaluator_module()

    row = evaluator.evaluate_sample(
        _artifact(vector_ids=vector_ids, graph_ids=graph_ids),
        _evaluation_graph(),
    )

    assert row["gold_page_rank_outcome"] == expected_outcome
    assert row["vector_best_gold_page_rank"] == vector_ids.index("runtime-gold") + 1
    assert row["graph_best_gold_page_rank"] == graph_ids.index("runtime-gold") + 1


def test_evaluate_sample_marks_policy_rule_as_not_applicable():
    evaluator = _load_evaluator_module()
    sample_id = "PMSQA_DEV_003"

    row = evaluator.evaluate_sample(
        _artifact(
            sample_id=sample_id,
            vector_ids=[],
            graph_ids=[],
            artifact_status="boundary_refusal",
            rerank_applied=False,
        ),
        _evaluation_graph(
            sample_id=sample_id,
            expected_decision="boundary_refusal",
            gold_status="policy_rule",
        ),
    )

    assert row["gold_page_rank_outcome"] == "not_applicable"
    assert row["boundary_status_alignment"] == "aligned"


def test_evaluate_sample_excludes_page_rank_when_rerank_was_not_applied():
    evaluator = _load_evaluator_module()

    row = evaluator.evaluate_sample(
        _artifact(
            vector_ids=["runtime-gold", "runtime-wrong"],
            graph_ids=["runtime-gold", "runtime-wrong"],
            rerank_applied=False,
        ),
        _evaluation_graph(),
    )

    assert row["gold_page_rank_outcome"] == "not_applicable"
    assert row["vector_best_gold_page_rank"] is None
    assert row["graph_best_gold_page_rank"] is None


def test_evaluate_sample_rejects_parent_fingerprint_mismatch():
    evaluator = _load_evaluator_module()

    with pytest.raises(ValueError, match="fingerprint"):
        evaluator.evaluate_sample(
            _artifact(
                vector_ids=["runtime-wrong"],
                graph_ids=["runtime-wrong"],
                parent_sha="artifact-parent",
            ),
            _evaluation_graph(parent_sha="evaluation-parent"),
        )


def test_evaluate_method_run_writes_rows_summary_and_failures(tmp_path: Path):
    evaluator = _load_evaluator_module()
    method_run = tmp_path / "method_run"
    artifact_dir = method_run / "method_artifacts"
    artifact_dir.mkdir(parents=True)
    evaluation_run = tmp_path / "evaluation_run"
    evaluation_dir = evaluation_run / "evaluation_graphs"
    evaluation_dir.mkdir(parents=True)
    output_dir = tmp_path / "offline_evaluation"

    artifact = _artifact(
        vector_ids=["runtime-wrong", "runtime-gold"],
        graph_ids=["runtime-gold", "runtime-wrong"],
    )
    evaluation_graph = _evaluation_graph()
    (artifact_dir / "PMSQA_DEV_001.json").write_text(
        json.dumps(artifact, ensure_ascii=False),
        encoding="utf-8",
    )
    (evaluation_dir / "PMSQA_DEV_001.json").write_text(
        json.dumps(evaluation_graph, ensure_ascii=False),
        encoding="utf-8",
    )
    (method_run / "run_manifest.json").write_text(
        json.dumps({"sample_ids": ["PMSQA_DEV_001"]}),
        encoding="utf-8",
    )

    summary = evaluator.evaluate_method_run(
        method_run_dir=method_run,
        evaluation_run_dir=evaluation_run,
        output_dir=output_dir,
    )

    assert summary["total_samples"] == 1
    assert summary["improved_count"] == 1
    assert summary["failed_count"] == 0
    assert summary["external_model_calls"] == 0
    assert summary["estimated_cost"] == 0
    assert (output_dir / "comparison_rows.jsonl").is_file()
    assert (output_dir / "failed_cases.jsonl").is_file()
