"""Reproducible Phase 6-A batch construction and audit runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.app.services.safety_boundary import is_direct_prescription_request

from .graph_contract import assert_no_gold_only_content, validate_selection_manifest
from .inference_graph_builder import build_inference_graph


RUNTIME_RETRIEVAL_FIELDS = (
    "content",
    "granularity",
    "distance",
    "relevance_score",
    "authority_weight",
    "final_score",
    "source_file",
    "page_number",
    "chapter_title",
    "block_type",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
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


def build_runtime_source_registry(
    source_manifest: dict[str, Any],
    index_status: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Intersect approved manifest entries with the actual ready index."""
    if index_status.get("ready") is not True:
        raise ValueError("index_status.ready must be true")

    sources = source_manifest.get("sources")
    indexed_sources = index_status.get("indexed_sources")
    if not isinstance(sources, list):
        raise ValueError("source_manifest.sources must be a list")
    if not isinstance(indexed_sources, list):
        raise ValueError("index_status.indexed_sources must be a list")

    indexed_filenames = {
        filename.strip()
        for filename in indexed_sources
        if isinstance(filename, str) and filename.strip()
    }
    eligible_by_filename: dict[str, dict[str, Any]] = {}
    seen_source_ids: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("every source manifest entry must be a dictionary")
        if (
            source.get("included_in_kb") is not True
            or source.get("status") not in {"approved", "indexed"}
        ):
            continue

        source_id = source.get("source_id")
        filename = source.get("filename")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("eligible source is missing source_id")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(f"eligible source is missing filename: {source_id}")
        if source_id in seen_source_ids:
            raise ValueError(f"duplicate eligible source_id: {source_id}")
        if filename in eligible_by_filename:
            raise ValueError(f"duplicate eligible source filename: {filename}")
        seen_source_ids.add(source_id)
        eligible_by_filename[filename] = source

    eligible_filenames = set(eligible_by_filename)
    admitted_filenames = sorted(eligible_filenames & indexed_filenames)
    runtime_sources = []
    for filename in admitted_filenames:
        runtime_source = dict(eligible_by_filename[filename])
        runtime_source["status"] = "indexed"
        runtime_sources.append(runtime_source)

    audit = {
        "index_ready": True,
        "manifest_eligible_count": len(eligible_filenames),
        "indexed_source_count": len(indexed_filenames),
        "admitted_source_count": len(runtime_sources),
        "manifest_eligible_not_indexed": sorted(
            eligible_filenames - indexed_filenames
        ),
        "indexed_not_manifest_eligible": sorted(
            indexed_filenames - eligible_filenames
        ),
    }
    return {
        "schema_version": source_manifest.get("schema_version"),
        "sources": runtime_sources,
    }, audit


def _project_retrieved_evidence(raw_results: object) -> list[dict[str, Any]]:
    if not isinstance(raw_results, list):
        raise ValueError("retrieval results must be a list")
    projected = []
    for result in raw_results:
        if not isinstance(result, dict):
            raise ValueError("every retrieval result must be a dictionary")
        projected.append(
            {
                field: result.get(field)
                for field in RUNTIME_RETRIEVAL_FIELDS
            }
        )
    return projected


def _canonical_graph_bytes(graph: dict[str, Any]) -> bytes:
    return json.dumps(
        graph,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _graph_validation(
    graph: dict[str, Any],
    *,
    deterministic: bool,
) -> dict[str, Any]:
    assert_no_gold_only_content(graph)
    source_nodes = [
        node for node in graph["nodes"] if node["type"] == "SourceDocument"
    ]
    evidence_nodes = [
        node for node in graph["nodes"] if node["type"] == "EvidenceSpan"
    ]
    return {
        "sample_id": graph["sample_id"],
        "build_status": graph["build_status"],
        "failure_reason": graph["failure_reason"],
        "deterministic": deterministic,
        "gold_leakage_check": "passed",
        "source_admission_check": "passed",
        "node_count": len(graph["nodes"]),
        "edge_count": len(graph["edges"]),
        "source_count": len(source_nodes),
        "evidence_count": len(evidence_nodes),
        "graph_sha256": hashlib.sha256(_canonical_graph_bytes(graph)).hexdigest(),
        "source_ids": sorted(
            node["properties"]["source_id"] for node in source_nodes
        ),
    }


def _render_summary(summary: dict[str, Any], run_manifest: dict[str, Any]) -> str:
    source_audit = run_manifest["source_registry_audit"]
    return "\n".join(
        [
            "# Phase 6-A inference graph batch summary",
            "",
            f"- Run ID: `{run_manifest['run_id']}`",
            f"- Total samples: {summary['total_samples']}",
            f"- Success graphs: {summary['success_count']}",
            f"- Empty-evidence graphs: {summary['empty_evidence_count']}",
            f"- Failed samples: {summary['failed_count']}",
            f"- Admitted sources: {source_audit['admitted_source_count']}",
            f"- External model calls: {summary['external_model_calls']}",
            f"- Estimated cost: {summary['estimated_cost']}",
            "",
            "## Runtime adapter limitation",
            "",
            "The normalized query is taken from the cached retrieval query and the "
            "intent is set to `CONTEXT`. This is a transparent cross-granularity "
            "passthrough adapter, not a production Router output.",
            "",
        ]
    )


def run_phase6a_batch(
    config: dict[str, Any],
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Build and audit frozen Phase 6-A inference graphs without model calls."""
    required_paths = (
        "selection_manifest",
        "dev50_path",
        "retrieval_outputs_path",
        "source_manifest_path",
        "index_status_path",
    )
    paths: dict[str, Path] = {}
    for field in required_paths:
        raw_path = config.get(field)
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"config.{field} must be a non-empty path")
        paths[field] = Path(raw_path)

    versions = config.get("versions")
    if not isinstance(versions, dict):
        raise ValueError("config.versions must be a dictionary")
    if config.get("router_input_mode") != "retrieval_query_passthrough":
        raise ValueError(
            "config.router_input_mode must be retrieval_query_passthrough"
        )

    selection_rows = _read_jsonl(paths["selection_manifest"])
    dev50_records = _read_jsonl(paths["dev50_path"])
    retrieval_records = _read_jsonl(paths["retrieval_outputs_path"])
    source_manifest = _read_json(paths["source_manifest_path"])
    index_status = _read_json(paths["index_status_path"])

    validate_selection_manifest(selection_rows, dev50_records)
    expected_sample_count = config.get("expected_sample_count")
    if expected_sample_count is not None and len(selection_rows) != expected_sample_count:
        raise ValueError(
            f"selection sample count mismatch: "
            f"expected={expected_sample_count}, actual={len(selection_rows)}"
        )

    dev50_by_id = _indexed_records(dev50_records, label="Dev50")
    retrieval_by_id = _indexed_records(retrieval_records, label="retrieval")
    selected_rows = sorted(selection_rows, key=lambda row: row["selection_rank"])
    selected_sample_ids = [row["sample_id"] for row in selected_rows]
    missing_retrieval = [
        sample_id
        for sample_id in selected_sample_ids
        if sample_id not in retrieval_by_id
    ]
    if missing_retrieval:
        raise ValueError(
            "missing retrieval record for selected sample(s): "
            + ", ".join(missing_retrieval)
        )

    runtime_questions = {
        sample_id: {
            "sample_id": sample_id,
            "question": dev50_by_id[sample_id]["question"],
        }
        for sample_id in selected_sample_ids
    }
    runtime_source_registry, source_audit = build_runtime_source_registry(
        source_manifest,
        index_status,
    )

    started_at = datetime.now().astimezone()
    run_id = config.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        run_id = f"phase6a_batch_v0_1_{started_at:%Y%m%d_%H%M%S}"
    if output_dir is None:
        output_root = Path(config.get("output_root", "revision/phase6/graph_runs"))
        output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    graph_dir = output_dir / "inference_graphs"
    validation_dir = output_dir / "validation"
    graph_dir.mkdir()
    validation_dir.mkdir()

    validations: list[dict[str, Any]] = []
    failed_cases: list[dict[str, Any]] = []
    for sample_id in selected_sample_ids:
        question = runtime_questions[sample_id]["question"]
        raw_retrieval = retrieval_by_id[sample_id]
        try:
            raw_query = raw_retrieval.get("query")
            if not isinstance(raw_query, str) or not raw_query.strip():
                raise ValueError("retrieval record query must be a non-empty string")
            router_output = {
                "normalized_query": raw_query,
                "intent": "CONTEXT",
            }
            evidence = _project_retrieved_evidence(raw_retrieval.get("results"))
            if is_direct_prescription_request(question):
                evidence = []
                empty_reason = "prescription_boundary_detected"
            else:
                empty_reason = "retrieval_returned_no_admitted_evidence"

            build_kwargs = {
                "sample_id": sample_id,
                "question": question,
                "router_output": router_output,
                "source_registry": runtime_source_registry,
                "versions": versions,
                "empty_evidence_reason": empty_reason,
            }
            graph = build_inference_graph(
                retrieved_evidence=evidence,
                **build_kwargs,
            )
            reordered_graph = build_inference_graph(
                retrieved_evidence=list(reversed(evidence)),
                **build_kwargs,
            )
            deterministic = graph == reordered_graph
            if not deterministic:
                raise ValueError("graph changed after retrieval evidence reordering")

            validation = _graph_validation(
                graph,
                deterministic=deterministic,
            )
            _write_json(graph_dir / f"{sample_id}.json", graph)
        except Exception as exc:
            validation = {
                "sample_id": sample_id,
                "build_status": "failed",
                "failure_reason": str(exc),
                "deterministic": False,
                "gold_leakage_check": "not_completed",
                "source_admission_check": "not_completed",
                "node_count": 0,
                "edge_count": 0,
                "source_count": 0,
                "evidence_count": 0,
                "graph_sha256": None,
                "source_ids": [],
            }
            failed_cases.append(
                {
                    "sample_id": sample_id,
                    "failure_reason": str(exc),
                }
            )
        validations.append(validation)
        _write_json(validation_dir / f"{sample_id}.json", validation)

    summary = {
        "total_samples": len(selected_sample_ids),
        "success_count": sum(
            validation["build_status"] == "success"
            for validation in validations
        ),
        "empty_evidence_count": sum(
            validation["build_status"] == "empty_evidence"
            for validation in validations
        ),
        "failed_count": len(failed_cases),
        "external_model_calls": 0,
        "estimated_cost": 0,
    }
    completed_at = datetime.now().astimezone()
    run_manifest = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "python_version": sys.version.split()[0],
        "versions": versions,
        "input_artifacts": {
            field: str(path) for field, path in paths.items()
        },
        "selected_sample_ids": selected_sample_ids,
        "router_input_mode": "retrieval_query_passthrough",
        "router_input_limitation": (
            "Cached retrieval queries are used as normalized queries with a "
            "CONTEXT intent; these are not production Router outputs."
        ),
        "runtime_input_gold_isolation": "projected_and_verified",
        "runtime_question_fields": ["sample_id", "question"],
        "runtime_retrieval_fields": list(RUNTIME_RETRIEVAL_FIELDS),
        "source_registry_audit": source_audit,
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
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = _read_json(args.config)
    summary = run_phase6a_batch(config, output_dir=args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
