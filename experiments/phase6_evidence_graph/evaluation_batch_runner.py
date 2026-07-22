"""Build isolated Phase 6-A evaluation graphs from a frozen inference run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .evaluation_graph_builder import build_evaluation_graph
from .graph_contract import validate_evaluation_graph


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


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


def _canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _indexed_records(
    rows: list[dict[str, Any]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"{label} record is missing sample_id")
        if sample_id in indexed:
            raise ValueError(f"duplicate {label} sample_id: {sample_id}")
        indexed[sample_id] = row
    return indexed


def _required_path(config: dict[str, Any], field: str) -> Path:
    raw_path = config.get(field)
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"config.{field} must be a non-empty path")
    return Path(raw_path)


def _validate_versions(
    actual: object,
    expected: dict[str, Any],
    *,
    label: str,
    fields: tuple[str, ...] = (
        "schema_version",
        "dataset_version",
        "kb_version",
    ),
) -> None:
    if not isinstance(actual, dict):
        raise ValueError(f"{label} versions must be a dictionary")
    for field in fields:
        if actual.get(field) != expected.get(field):
            raise ValueError(
                f"{label} version mismatch for {field}: "
                f"expected={expected.get(field)!r}, actual={actual.get(field)!r}"
            )


def _render_summary(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Phase 6-A evaluation graph batch summary",
            "",
            f"- Run ID: `{manifest['run_id']}`",
            f"- Parent inference run: `{manifest['parent_inference_run_id']}`",
            f"- Total samples: {summary['total_samples']}",
            f"- Evaluation graphs: {summary['evaluation_graph_count']}",
            f"- Deterministic graphs: {summary['deterministic_count']}",
            f"- Parent files unchanged: {summary['parent_immutability_count']}",
            f"- Page-span cases: {summary['page_span_count']}",
            f"- Policy-rule cases: {summary['policy_rule_count']}",
            f"- Missing-source cases: {summary['missing_source_count']}",
            f"- Method outputs pending: {summary['method_output_pending_count']}",
            f"- Failed cases: {summary['failed_count']}",
            f"- External model calls: {summary['external_model_calls']}",
            f"- Estimated cost: {summary['estimated_cost']}",
            "",
            "This run attaches frozen benchmark annotations to physically "
            "isolated evaluation graphs. It does not call a model, rerun "
            "retrieval, score method outputs, or modify parent inference graphs.",
            "",
        ]
    )


def run_evaluation_batch(
    config: dict[str, Any],
    *,
    sample_limit: int | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Build deterministic evaluation scaffolds without touching runtime graphs."""
    inference_run_dir = _required_path(config, "inference_run_dir")
    selection_path = _required_path(config, "selection_manifest")
    dev50_path = _required_path(config, "dev50_path")
    inference_graph_dir = inference_run_dir / "inference_graphs"
    inference_manifest_path = inference_run_dir / "run_manifest.json"

    versions = config.get("versions")
    if not isinstance(versions, dict):
        raise ValueError("config.versions must be a dictionary")
    _validate_versions(versions, versions, label="config")

    selection_rows = _read_jsonl(selection_path)
    dev50_records = _read_jsonl(dev50_path)
    inference_manifest = _read_json(inference_manifest_path)
    _validate_versions(
        inference_manifest.get("versions"),
        versions,
        label="inference run",
    )

    expected_sample_count = config.get("expected_sample_count")
    if (
        expected_sample_count is not None
        and len(selection_rows) != expected_sample_count
    ):
        raise ValueError(
            "selection sample count mismatch: "
            f"expected={expected_sample_count}, actual={len(selection_rows)}"
        )
    if sample_limit is not None and sample_limit < 1:
        raise ValueError("sample_limit must be a positive integer")

    dev50_by_id = _indexed_records(dev50_records, label="Dev50")
    selected_rows = sorted(
        selection_rows,
        key=lambda row: row.get("selection_rank", 0),
    )
    selected_sample_ids = [row.get("sample_id") for row in selected_rows]
    if any(
        not isinstance(sample_id, str) or not sample_id.strip()
        for sample_id in selected_sample_ids
    ):
        raise ValueError("selection record is missing sample_id")
    if len(set(selected_sample_ids)) != len(selected_sample_ids):
        raise ValueError("selection manifest contains duplicate sample_id")
    if sample_limit is not None:
        selected_sample_ids = selected_sample_ids[:sample_limit]

    missing_dev50 = [
        sample_id
        for sample_id in selected_sample_ids
        if sample_id not in dev50_by_id
    ]
    if missing_dev50:
        raise ValueError(
            "missing Dev50 record for selected sample(s): "
            + ", ".join(missing_dev50)
        )

    preflight: dict[str, dict[str, Any]] = {}
    for sample_id in selected_sample_ids:
        benchmark_record = dev50_by_id[sample_id]
        _validate_versions(
            benchmark_record,
            versions,
            label=sample_id,
            fields=("dataset_version", "kb_version"),
        )
        if benchmark_record.get("freeze_status") != "frozen":
            raise ValueError(f"benchmark record is not frozen: {sample_id}")

        graph_path = inference_graph_dir / f"{sample_id}.json"
        if not graph_path.is_file():
            raise ValueError(f"missing inference graph: {sample_id}")
        inference_graph = _read_json(graph_path)
        _validate_versions(
            inference_graph.get("versions"),
            versions,
            label=f"inference graph {sample_id}",
        )
        if inference_graph.get("sample_id") != sample_id:
            raise ValueError(f"inference graph sample_id mismatch: {sample_id}")
        preflight[sample_id] = {
            "benchmark_record": benchmark_record,
            "graph_path": graph_path,
            "inference_graph": inference_graph,
            "file_sha256": _file_sha256(graph_path),
        }

    started_at = datetime.now().astimezone()
    run_id = config.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        run_id = (
            output_dir.name
            if output_dir is not None
            else f"phase6a_evaluation_v0_1_{started_at:%Y%m%d_%H%M%S}"
        )
    if output_dir is None:
        output_root = Path(
            config.get("output_root", "revision/phase6/graph_runs")
        )
        output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    graph_output_dir = output_dir / "evaluation_graphs"
    validation_dir = output_dir / "validation"
    graph_output_dir.mkdir()
    validation_dir.mkdir()

    validations: list[dict[str, Any]] = []
    failed_cases: list[dict[str, Any]] = []
    for sample_id in selected_sample_ids:
        item = preflight[sample_id]
        try:
            inference_graph = item["inference_graph"]
            benchmark_record = item["benchmark_record"]
            evaluation_graph = build_evaluation_graph(
                inference_graph=inference_graph,
                benchmark_record=benchmark_record,
            )
            repeated_graph = build_evaluation_graph(
                inference_graph=inference_graph,
                benchmark_record=benchmark_record,
            )
            deterministic = evaluation_graph == repeated_graph
            if not deterministic:
                raise ValueError("evaluation graph changed on repeated build")
            validate_evaluation_graph(evaluation_graph)

            expected_parent_hash = _canonical_sha256(inference_graph)
            if evaluation_graph.get("inference_graph_sha256") != expected_parent_hash:
                raise ValueError("evaluation graph parent hash mismatch")
            parent_unchanged = (
                _file_sha256(item["graph_path"]) == item["file_sha256"]
            )
            if not parent_unchanged:
                raise ValueError("parent inference graph file was modified")

            validation = {
                "sample_id": sample_id,
                "build_status": "success",
                "failure_reason": None,
                "deterministic": True,
                "parent_unchanged": True,
                "parent_file_sha256": item["file_sha256"],
                "parent_graph_sha256": expected_parent_hash,
                "evaluation_graph_sha256": _canonical_sha256(evaluation_graph),
                "gold_evidence_status": evaluation_graph.get(
                    "gold_evidence_status"
                ),
                "method_output_status": evaluation_graph.get(
                    "method_output_status"
                ),
                "evaluation_status": evaluation_graph.get("evaluation_status"),
                "node_count": len(evaluation_graph["nodes"]),
                "edge_count": len(evaluation_graph["edges"]),
            }
            _write_json(
                graph_output_dir / f"{sample_id}.json",
                evaluation_graph,
            )
        except Exception as exc:
            validation = {
                "sample_id": sample_id,
                "build_status": "failed",
                "failure_reason": str(exc),
                "deterministic": False,
                "parent_unchanged": (
                    _file_sha256(item["graph_path"]) == item["file_sha256"]
                ),
                "parent_file_sha256": item["file_sha256"],
                "parent_graph_sha256": None,
                "evaluation_graph_sha256": None,
                "gold_evidence_status": item["benchmark_record"].get(
                    "gold_evidence_status"
                ),
                "method_output_status": None,
                "evaluation_status": None,
                "node_count": 0,
                "edge_count": 0,
            }
            failed_cases.append(
                {
                    "sample_id": sample_id,
                    "failure_reason": str(exc),
                }
            )
        validations.append(validation)
        _write_json(validation_dir / f"{sample_id}.json", validation)

    summary: dict[str, Any] = {
        "total_samples": len(selected_sample_ids),
        "evaluation_graph_count": sum(
            row["build_status"] == "success" for row in validations
        ),
        "deterministic_count": sum(
            row["deterministic"] is True for row in validations
        ),
        "parent_immutability_count": sum(
            row["parent_unchanged"] is True for row in validations
        ),
        "page_span_count": sum(
            row["gold_evidence_status"] == "page_span_located"
            for row in validations
        ),
        "policy_rule_count": sum(
            row["gold_evidence_status"] == "policy_rule"
            for row in validations
        ),
        "missing_source_count": sum(
            row["gold_evidence_status"] == "missing_source"
            for row in validations
        ),
        "method_output_pending_count": sum(
            row["method_output_status"] == "not_attached"
            for row in validations
        ),
        "failed_count": len(failed_cases),
        "external_model_calls": 0,
        "estimated_cost": 0,
        "output_dir": str(output_dir.resolve()),
    }
    completed_at = datetime.now().astimezone()
    run_manifest = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "python_version": sys.version.split()[0],
        "versions": versions,
        "parent_inference_run_id": inference_manifest.get("run_id"),
        "parent_inference_run_dir": str(inference_run_dir),
        "input_artifacts": {
            "selection_manifest": str(selection_path),
            "dev50_path": str(dev50_path),
            "inference_run_manifest": str(inference_manifest_path),
        },
        "selected_sample_ids": selected_sample_ids,
        "sample_limit": sample_limit,
        "gold_isolation": "evaluation_graph_only",
        "parent_graph_access": "read_only_verified",
        "method_output_status": "not_attached",
        "evaluation_status": "awaiting_method_output",
        "summary": summary,
    }
    _write_json(output_dir / "run_manifest.json", run_manifest)
    _write_jsonl(output_dir / "failed_cases.jsonl", failed_cases)
    (output_dir / "summary.md").write_text(
        _render_summary(summary, run_manifest),
        encoding="utf-8",
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = _read_json(args.config)
    summary = run_evaluation_batch(
        config,
        sample_limit=args.sample_limit,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
