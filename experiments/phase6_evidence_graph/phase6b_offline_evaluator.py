"""Offline gold-page evaluation for frozen Phase 6-B runtime artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVALUATOR_VERSION = "phase6b-offline-gold-page-evaluator-v0.2"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            f"{json.dumps(row, ensure_ascii=False, sort_keys=True)}\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _decision_type(evaluation_graph: dict[str, Any]) -> str:
    decisions = [
        node
        for node in evaluation_graph.get("nodes", [])
        if node.get("type") == "Decision"
        and node.get("provenance") == "benchmark_annotation"
    ]
    if len(decisions) != 1:
        raise ValueError("evaluation graph must contain exactly one gold Decision")
    decision_type = decisions[0].get("properties", {}).get("decision_type")
    if not isinstance(decision_type, str) or not decision_type:
        raise ValueError("evaluation decision_type must be a non-empty string")
    return decision_type


def _runtime_evidence_by_id(
    evaluation_graph: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        node["id"]: node
        for node in evaluation_graph.get("nodes", [])
        if node.get("type") == "EvidenceSpan"
        and node.get("provenance") == "runtime_retrieval"
    }


def _gold_source_pages(
    evaluation_graph: dict[str, Any],
) -> set[tuple[str, int]]:
    source_pages = set()
    for node in evaluation_graph.get("nodes", []):
        if (
            node.get("type") != "EvidenceSpan"
            or node.get("provenance") != "gold_evidence"
        ):
            continue
        properties = node.get("properties", {})
        source_id = properties.get("source_id")
        page_number = properties.get("page_number")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("gold evidence source_id must be a non-empty string")
        if isinstance(page_number, bool) or not isinstance(page_number, int):
            raise ValueError("gold evidence page_number must be an integer")
        source_pages.add((source_id, page_number))
    return source_pages


def _best_gold_page_rank(
    evidence_ids: list[str],
    runtime_evidence: dict[str, dict[str, Any]],
    gold_source_pages: set[tuple[str, int]],
) -> int | None:
    for rank, evidence_id in enumerate(evidence_ids, 1):
        node = runtime_evidence.get(evidence_id)
        if node is None:
            raise ValueError(
                f"ranking references unknown runtime evidence: {evidence_id}"
            )
        properties = node.get("properties", {})
        key = (properties.get("source_id"), properties.get("page_number"))
        if key in gold_source_pages:
            return rank
    return None


def _rank_outcome(
    vector_rank: int | None,
    graph_rank: int | None,
) -> str:
    if vector_rank == graph_rank:
        return "unchanged"
    if vector_rank is None:
        return "improved"
    if graph_rank is None:
        return "worsened"
    return "improved" if graph_rank < vector_rank else "worsened"


def _boundary_status_alignment(
    expected_decision: str,
    artifact_status: str,
) -> str:
    if expected_decision == "boundary_refusal":
        return (
            "aligned"
            if artifact_status == "boundary_refusal"
            else "misaligned"
        )
    if expected_decision == "insufficient_evidence":
        return (
            "aligned"
            if artifact_status == "insufficient_graph_evidence"
            else "not_applicable"
        )
    return "not_applicable"


def evaluate_sample(
    artifact: dict[str, Any],
    evaluation_graph: dict[str, Any],
) -> dict[str, Any]:
    """Compare frozen runtime rankings against isolated gold source/page pairs."""
    if artifact.get("artifact_type") != "phase6b_reranking_artifact":
        raise ValueError("unsupported method artifact type")
    if evaluation_graph.get("graph_type") != "evaluation_graph":
        raise ValueError("offline evaluator accepts evaluation_graph only")
    sample_id = artifact.get("sample_id")
    if sample_id != evaluation_graph.get("sample_id"):
        raise ValueError("sample_id mismatch")
    if artifact.get("parent_inference_graph_id") != evaluation_graph.get(
        "inference_graph_id"
    ):
        raise ValueError("parent inference graph ID mismatch")
    if artifact.get("parent_inference_graph_sha256") != evaluation_graph.get(
        "inference_graph_sha256"
    ):
        raise ValueError("parent inference graph fingerprint mismatch")

    baseline = artifact.get("ranking_baseline")
    if not isinstance(baseline, dict):
        raise ValueError("method artifact lacks ranking_baseline")
    vector_ids = baseline.get("vector_top_k_evidence_ids")
    graph_ids = baseline.get("graph_top_k_evidence_ids")
    if not isinstance(vector_ids, list) or not isinstance(graph_ids, list):
        raise ValueError("ranking baseline IDs must be lists")

    expected_decision = _decision_type(evaluation_graph)
    gold_status = evaluation_graph.get("gold_evidence_status")
    rerank_applied = artifact.get("rerank_applied")
    if not isinstance(rerank_applied, bool):
        raise ValueError("method artifact rerank_applied must be a boolean")
    vector_rank = None
    graph_rank = None
    outcome = "not_applicable"
    if gold_status == "page_span_located" and rerank_applied:
        runtime_evidence = _runtime_evidence_by_id(evaluation_graph)
        gold_source_pages = _gold_source_pages(evaluation_graph)
        if not gold_source_pages:
            raise ValueError("page_span_located graph has no gold evidence pages")
        vector_rank = _best_gold_page_rank(
            vector_ids,
            runtime_evidence,
            gold_source_pages,
        )
        graph_rank = _best_gold_page_rank(
            graph_ids,
            runtime_evidence,
            gold_source_pages,
        )
        outcome = _rank_outcome(vector_rank, graph_rank)

    return {
        "sample_id": sample_id,
        "artifact_sha256": artifact.get("artifact_sha256"),
        "artifact_status": artifact.get("artifact_status"),
        "rerank_applied": rerank_applied,
        "rerank_skip_reason": artifact.get("rerank_skip_reason"),
        "gold_evidence_status": gold_status,
        "expected_decision": expected_decision,
        "vector_best_gold_page_rank": vector_rank,
        "graph_best_gold_page_rank": graph_rank,
        "gold_page_rank_outcome": outcome,
        "top1_changed": baseline.get("top1_changed"),
        "boundary_status_alignment": _boundary_status_alignment(
            expected_decision,
            str(artifact.get("artifact_status") or ""),
        ),
    }


def _summary_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Phase 6-B offline gold-page evaluation",
            "",
            f"- Total samples: {summary['total_samples']}",
            f"- Rerank applied: {summary['rerank_applied_count']}",
            f"- Improved: {summary['improved_count']}",
            f"- Unchanged: {summary['unchanged_count']}",
            f"- Worsened: {summary['worsened_count']}",
            f"- Not applicable: {summary['not_applicable_count']}",
            f"- Failed: {summary['failed_count']}",
            f"- Boundary aligned: {summary['boundary_aligned_count']}",
            f"- External model calls: {summary['external_model_calls']}",
            f"- Estimated cost: {summary['estimated_cost']}",
            "",
            "Outcomes are exact source/page development diagnostics, not clinical effectiveness claims.",
            "",
        ]
    )


def evaluate_method_run(
    *,
    method_run_dir: Path,
    evaluation_run_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate all frozen runtime artifacts without modifying either parent run."""
    method_run_dir = Path(method_run_dir).resolve()
    evaluation_run_dir = Path(evaluation_run_dir).resolve()
    output_dir = Path(output_dir)
    artifact_dir = method_run_dir / "method_artifacts"
    evaluation_dir = evaluation_run_dir / "evaluation_graphs"
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"method artifact directory not found: {artifact_dir}")
    if not evaluation_dir.is_dir():
        raise FileNotFoundError(
            f"evaluation graph directory not found: {evaluation_dir}"
        )
    manifest = _read_json(method_run_dir / "run_manifest.json")
    sample_ids = manifest.get("sample_ids")
    if not isinstance(sample_ids, list) or not sample_ids:
        raise ValueError("method run manifest sample_ids must be a non-empty list")

    output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    failed_cases: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        try:
            artifact = _read_json(artifact_dir / f"{sample_id}.json")
            evaluation_graph = _read_json(
                evaluation_dir / f"{sample_id}.json"
            )
            rows.append(evaluate_sample(artifact, evaluation_graph))
        except Exception as exc:  # pragma: no cover - real-run failure capture
            failed_cases.append(
                {
                    "sample_id": sample_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    outcome_counts = Counter(row["gold_page_rank_outcome"] for row in rows)
    summary = {
        "evaluator_version": EVALUATOR_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_run_dir": str(method_run_dir),
        "evaluation_run_dir": str(evaluation_run_dir),
        "total_samples": len(sample_ids),
        "evaluated_count": len(rows),
        "rerank_applied_count": sum(
            row["rerank_applied"] for row in rows
        ),
        "improved_count": outcome_counts.get("improved", 0),
        "unchanged_count": outcome_counts.get("unchanged", 0),
        "worsened_count": outcome_counts.get("worsened", 0),
        "not_applicable_count": outcome_counts.get("not_applicable", 0),
        "boundary_aligned_count": sum(
            row["boundary_status_alignment"] == "aligned" for row in rows
        ),
        "failed_count": len(failed_cases),
        "external_model_calls": 0,
        "estimated_cost": 0,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_jsonl(output_dir / "comparison_rows.jsonl", rows)
    _write_jsonl(output_dir / "failed_cases.jsonl", failed_cases)
    (output_dir / "summary.md").write_text(
        _summary_markdown(summary),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-run-dir", type=Path, required=True)
    parser.add_argument("--evaluation-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = evaluate_method_run(
        method_run_dir=args.method_run_dir,
        evaluation_run_dir=args.evaluation_run_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
