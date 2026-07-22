"""Reproducible Phase 6-B same-candidate reranking batch runner."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .graph_contract import assert_no_gold_only_content
from .graph_reranker import build_reranking_artifact, canonical_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _sample_ids(config: dict[str, Any]) -> list[str]:
    raw_ids = config.get("sample_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError("sample_ids must be a non-empty list")
    sample_ids = []
    seen = set()
    for raw_id in raw_ids:
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise ValueError("sample_ids must contain non-empty strings")
        sample_id = raw_id.strip()
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        sample_ids.append(sample_id)
    return sample_ids


def _summary_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Phase 6-B reranking batch summary",
            "",
            f"- Total samples: {summary['total_samples']}",
            f"- Success: {summary['success_count']}",
            f"- Boundary refusal: {summary['boundary_refusal_count']}",
            f"- Insufficient graph evidence: {summary['insufficient_graph_evidence_count']}",
            f"- Failed: {summary['failed_count']}",
            f"- Ranking applicable: {summary['ranking_applicable_count']}",
            f"- Ranking not applicable: {summary['ranking_not_applicable_count']}",
            f"- Top-1 changed: {summary['top1_changed_count']}",
            f"- Top-1 unchanged: {summary['top1_unchanged_count']}",
            f"- Constraint conflicts: {summary['constraint_conflict_count']}",
            f"- Unsupported constraints: {summary['constraint_unsupported_count']}",
            f"- Deterministic: {summary['deterministic_count']}/{summary['total_samples']}",
            f"- Parent graph unchanged: {summary['parent_graph_unchanged_count']}/{summary['total_samples']}",
            f"- Gold leakage checks passed: {summary['gold_leakage_passed_count']}/{summary['total_samples']}",
            f"- External model calls: {summary['external_model_calls']}",
            f"- Estimated cost: {summary['estimated_cost']}",
            "",
            "This is a development-set ranking diagnostic, not a paper-level effectiveness result.",
            "",
        ]
    )


def _ranking_diagnostic(artifact: dict[str, Any]) -> dict[str, Any]:
    baseline = artifact["ranking_baseline"]
    vector_ids = baseline["vector_top_k_evidence_ids"]
    graph_ids = baseline["graph_top_k_evidence_ids"]
    ranking_status = (
        "applicable" if artifact["rerank_applied"] else "not_applicable"
    )
    return {
        "sample_id": artifact["sample_id"],
        "artifact_status": artifact["artifact_status"],
        "ranking_status": ranking_status,
        "rerank_applied": artifact["rerank_applied"],
        "rerank_skip_reason": artifact["rerank_skip_reason"],
        "vector_top_k_evidence_ids": vector_ids,
        "graph_top_k_evidence_ids": graph_ids,
        "top1_changed": (
            baseline["top1_changed"] if ranking_status == "applicable" else None
        ),
        "moved_in_evidence_ids": baseline["moved_in_evidence_ids"],
        "moved_out_evidence_ids": baseline["moved_out_evidence_ids"],
        "audit_summary": artifact["audit_summary"],
    }


def run_batch(
    config: dict[str, Any],
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the frozen sample list without reading any evaluation graph."""
    sample_ids = _sample_ids(config)
    input_run_dir = _resolve_path(config.get("input_run_dir", ""))
    inference_dir = input_run_dir / "inference_graphs"
    if not inference_dir.is_dir():
        raise FileNotFoundError(f"inference graph directory not found: {inference_dir}")

    if output_dir is None:
        output_root = _resolve_path(config.get("output_root", ""))
        run_id_prefix = str(
            config.get("run_id_prefix") or "phase6b_rerank_batch"
        ).strip()
        run_id = (
            f"{run_id_prefix}_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
        output_dir = output_root / run_id
    else:
        output_dir = Path(output_dir)
        run_id = output_dir.name

    output_dir.mkdir(parents=True, exist_ok=False)
    artifact_dir = output_dir / "method_artifacts"
    artifact_dir.mkdir()

    validations: list[dict[str, Any]] = []
    failed_cases: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    ranking_diagnostics: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        graph_path = inference_dir / f"{sample_id}.json"
        try:
            before_bytes = graph_path.read_bytes()
            inference_graph = json.loads(before_bytes.decode("utf-8"))
            before_sha = canonical_sha256(inference_graph)
            first = build_reranking_artifact(inference_graph, config)
            second = build_reranking_artifact(inference_graph, config)
            deterministic = first == second
            after_bytes = graph_path.read_bytes()
            parent_graph_unchanged = before_bytes == after_bytes
            assert_no_gold_only_content(first)

            _write_json(artifact_dir / f"{sample_id}.json", first)
            artifacts.append(first)
            ranking_diagnostics.append(_ranking_diagnostic(first))
            validations.append(
                {
                    "sample_id": sample_id,
                    "artifact_status": first["artifact_status"],
                    "deterministic": deterministic,
                    "parent_graph_unchanged": parent_graph_unchanged,
                    "parent_graph_sha256": before_sha,
                    "artifact_parent_sha256": first[
                        "parent_inference_graph_sha256"
                    ],
                    "gold_leakage_check": "passed",
                    "evidence_count": len(first["ranked_evidence"]),
                    "constraint_audit_count": len(first["constraint_audit"]),
                }
            )
        except Exception as exc:  # pragma: no cover - exercised by failed real inputs
            failed_cases.append(
                {
                    "sample_id": sample_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    status_counts: dict[str, int] = {}
    for artifact in artifacts:
        status = artifact["artifact_status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    applicable_diagnostics = [
        row
        for row in ranking_diagnostics
        if row["ranking_status"] == "applicable"
    ]

    summary = {
        "run_id": run_id,
        "config_version": config.get("config_version"),
        "method_id": config.get("method_id"),
        "method_version": config.get("method_version"),
        "total_samples": len(sample_ids),
        "success_count": status_counts.get("success", 0),
        "boundary_refusal_count": status_counts.get("boundary_refusal", 0),
        "insufficient_graph_evidence_count": status_counts.get(
            "insufficient_graph_evidence",
            0,
        ),
        "failed_count": len(failed_cases),
        "ranking_applicable_count": len(applicable_diagnostics),
        "ranking_not_applicable_count": (
            len(ranking_diagnostics) - len(applicable_diagnostics)
        ),
        "top1_changed_count": sum(
            row["top1_changed"] is True for row in applicable_diagnostics
        ),
        "top1_unchanged_count": sum(
            row["top1_changed"] is False for row in applicable_diagnostics
        ),
        "constraint_conflict_count": sum(
            int(row["audit_summary"].get("conflict", 0))
            for row in ranking_diagnostics
        ),
        "constraint_unsupported_count": sum(
            int(row["audit_summary"].get("unsupported", 0))
            for row in ranking_diagnostics
        ),
        "deterministic_count": sum(
            bool(row["deterministic"]) for row in validations
        ),
        "parent_graph_unchanged_count": sum(
            bool(row["parent_graph_unchanged"]) for row in validations
        ),
        "gold_leakage_passed_count": sum(
            row["gold_leakage_check"] == "passed" for row in validations
        ),
        "external_model_calls": 0,
        "estimated_cost": 0,
    }
    run_manifest = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_version": config.get("config_version"),
        "method_id": config.get("method_id"),
        "method_version": config.get("method_version"),
        "input_run_dir": str(input_run_dir),
        "sample_ids": sample_ids,
        "top_k": config.get("top_k"),
        "score_weights": config.get("score_weights"),
        "constraint_type_weights": config.get("constraint_type_weights"),
        "external_model_calls": 0,
        "estimated_cost": 0,
    }

    _write_json(output_dir / "run_manifest.json", run_manifest)
    _write_json(output_dir / "summary.json", summary)
    _write_jsonl(output_dir / "validations.jsonl", validations)
    _write_jsonl(
        output_dir / "ranking_diagnostics.jsonl",
        ranking_diagnostics,
    )
    _write_jsonl(output_dir / "failed_cases.jsonl", failed_cases)
    (output_dir / "summary.md").write_text(
        _summary_markdown(summary),
        encoding="utf-8",
    )
    return summary


def run_smoke(
    config: dict[str, Any],
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Backward-compatible alias for the original Phase 6-B1 entry point."""
    return run_batch(config, output_dir=output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = _read_json(args.config.resolve())
    summary = run_batch(config, output_dir=args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
