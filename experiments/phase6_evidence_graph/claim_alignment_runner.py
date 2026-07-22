"""Reproducible batch runner for Phase 6-B claim alignment."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .claim_evidence_aligner import align_claims
from .graph_contract import assert_no_gold_only_content
from .graph_reranker import canonical_sha256


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
    sample_ids: list[str] = []
    seen: set[str] = set()
    for raw_id in raw_ids:
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise ValueError("sample_ids must contain non-empty strings")
        sample_id = raw_id.strip()
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        sample_ids.append(sample_id)
    return sample_ids


def _candidate_outputs(
    config: dict[str, Any], sample_ids: list[str]
) -> dict[str, str]:
    raw_outputs = config.get("candidate_outputs")
    if not isinstance(raw_outputs, dict):
        raise ValueError("candidate_outputs must be a dictionary")
    outputs: dict[str, str] = {}
    for sample_id in sample_ids:
        value = raw_outputs.get(sample_id)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"missing candidate output for {sample_id}")
        outputs[sample_id] = value.strip()
    return outputs


def _claim_audit_rows(
    artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        for claim_index, claim in enumerate(artifact["claims"], start=1):
            source_binding = claim.get("source_binding") or {}
            runtime_constraints = claim.get("runtime_constraints") or []
            rows.append(
                {
                    "sample_id": artifact["sample_id"],
                    "claim_index": claim_index,
                    "claim_id": claim["claim_id"],
                    "claim_text": claim["claim_text"],
                    "stance": claim["stance"],
                    "support_state": claim["support_state"],
                    "reason_codes": list(claim.get("reason_codes") or []),
                    "source_binding_status": source_binding.get(
                        "binding_status", "unbound"
                    ),
                    "bound_source_ids": list(
                        source_binding.get("admitted_source_ids") or []
                    ),
                    "runtime_constraint_types": sorted(
                        {
                            constraint["constraint_type"]
                            for constraint in runtime_constraints
                        }
                    ),
                    "supporting_evidence_count": len(
                        claim.get("supporting_evidence_ids") or []
                    ),
                    "contradicting_evidence_count": len(
                        claim.get("contradicting_evidence_ids") or []
                    ),
                }
            )
    return rows


def _count_claim_values(
    rows: list[dict[str, Any]], key: str
) -> dict[str, int]:
    return dict(sorted(Counter(row[key] for row in rows).items()))


def _claim_state_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row["support_state"] for row in rows)
    return {
        state: counts.get(state, 0)
        for state in (
            "supported",
            "contradicted",
            "unsupported",
            "insufficient_evidence",
        )
    }


def _count_reason_codes(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        reason_code
        for row in rows
        for reason_code in row["reason_codes"]
    )
    return dict(sorted(counts.items()))


def _summary_markdown(summary: dict[str, Any]) -> str:
    claim_state_counts = json.dumps(
        summary["claim_state_counts"], ensure_ascii=False, sort_keys=True
    )
    reason_code_counts = json.dumps(
        summary["reason_code_counts"], ensure_ascii=False, sort_keys=True
    )
    source_binding_counts = json.dumps(
        summary["source_binding_status_counts"],
        ensure_ascii=False,
        sort_keys=True,
    )
    return "\n".join(
        [
            "# Phase 6-B claim alignment summary",
            "",
            f"- Total samples: {summary['total_samples']}",
            f"- Supported: {summary['supported_count']}",
            f"- Contradicted: {summary['contradicted_count']}",
            f"- Unsupported: {summary['unsupported_count']}",
            f"- Insufficient evidence: {summary['insufficient_evidence_count']}",
            f"- Boundary refusal: {summary['boundary_refusal_count']}",
            f"- Failed: {summary['failed_count']}",
            f"- Deterministic: {summary['deterministic_count']}/{summary['total_samples']}",
            f"- Parent artifact unchanged: {summary['parent_artifact_unchanged_count']}/{summary['total_samples']}",
            f"- Gold leakage checks passed: {summary['gold_leakage_passed_count']}/{summary['total_samples']}",
            f"- Total atomic claims: {summary['total_claim_count']}",
            f"- Claim states: {claim_state_counts}",
            f"- Reason codes: {reason_code_counts}",
            f"- Source binding states: {source_binding_counts}",
            f"- External model calls: {summary['external_model_calls']}",
            f"- Estimated cost: {summary['estimated_cost']}",
            "",
            "This is a deterministic development diagnostic, not a paper-level effectiveness result.",
            "",
        ]
    )


def run_batch(
    config: dict[str, Any],
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Align frozen candidate outputs without reading evaluation graphs."""
    sample_ids = _sample_ids(config)
    outputs = _candidate_outputs(config, sample_ids)
    input_run_dir = _resolve_path(config.get("input_run_dir", ""))
    parent_dir = input_run_dir / "method_artifacts"
    if not parent_dir.is_dir():
        raise FileNotFoundError(
            f"reranking artifact directory not found: {parent_dir}"
        )

    if output_dir is None:
        output_root = _resolve_path(config.get("output_root", ""))
        prefix = str(
            config.get("run_id_prefix") or "phase6b_claim_alignment"
        ).strip()
        run_id = (
            f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
        output_dir = output_root / run_id
    else:
        output_dir = Path(output_dir)
        run_id = output_dir.name

    output_dir.mkdir(parents=True, exist_ok=False)
    artifact_dir = output_dir / "claim_alignment_artifacts"
    artifact_dir.mkdir()

    validations: list[dict[str, Any]] = []
    failed_cases: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    cached_outputs: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        parent_path = parent_dir / f"{sample_id}.json"
        answer_text = outputs[sample_id]
        cached_outputs.append(
            {
                "sample_id": sample_id,
                "candidate_output_origin": config.get(
                    "candidate_output_origin"
                ),
                "answer_text": answer_text,
                "dataset_version": config.get("dataset_version"),
                "kb_version": config.get("kb_version"),
                "prompt_version": config.get("prompt_version"),
            }
        )
        try:
            before_bytes = parent_path.read_bytes()
            parent = json.loads(before_bytes.decode("utf-8"))
            assert_no_gold_only_content(parent)
            parent_sha = canonical_sha256(parent)
            first = align_claims(parent, answer_text, config)
            second = align_claims(parent, answer_text, config)
            deterministic = first == second
            parent_unchanged = before_bytes == parent_path.read_bytes()
            assert_no_gold_only_content(first)

            _write_json(artifact_dir / f"{sample_id}.json", first)
            artifacts.append(first)
            validations.append(
                {
                    "sample_id": sample_id,
                    "overall_support_state": first[
                        "overall_support_state"
                    ],
                    "deterministic": deterministic,
                    "parent_artifact_unchanged": parent_unchanged,
                    "parent_artifact_sha256": parent_sha,
                    "alignment_parent_sha256": first[
                        "parent_reranking_artifact_sha256"
                    ],
                    "gold_leakage_check": "passed",
                    "claim_count": first["claim_summary"]["claim_count"],
                }
            )
        except Exception as exc:  # pragma: no cover - real-input guard
            failed_cases.append(
                {
                    "sample_id": sample_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    state_counts: dict[str, int] = {}
    for artifact in artifacts:
        state = artifact["overall_support_state"]
        state_counts[state] = state_counts.get(state, 0) + 1
    claim_rows = _claim_audit_rows(artifacts)
    summary = {
        "run_id": run_id,
        "config_version": config.get("config_version"),
        "method_id": config.get("method_id"),
        "method_version": config.get("method_version"),
        "total_samples": len(sample_ids),
        "supported_count": state_counts.get("supported", 0),
        "contradicted_count": state_counts.get("contradicted", 0),
        "unsupported_count": state_counts.get("unsupported", 0),
        "insufficient_evidence_count": state_counts.get(
            "insufficient_evidence", 0
        ),
        "boundary_refusal_count": state_counts.get("not_applicable", 0),
        "failed_count": len(failed_cases),
        "deterministic_count": sum(
            bool(row["deterministic"]) for row in validations
        ),
        "parent_artifact_unchanged_count": sum(
            bool(row["parent_artifact_unchanged"])
            for row in validations
        ),
        "gold_leakage_passed_count": sum(
            row["gold_leakage_check"] == "passed" for row in validations
        ),
        "total_claim_count": len(claim_rows),
        "claim_state_counts": _claim_state_counts(claim_rows),
        "reason_code_counts": _count_reason_codes(claim_rows),
        "source_binding_status_counts": _count_claim_values(
            claim_rows, "source_binding_status"
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
        "candidate_output_origin": config.get("candidate_output_origin"),
        "dataset_version": config.get("dataset_version"),
        "kb_version": config.get("kb_version"),
        "prompt_version": config.get("prompt_version"),
        "external_model_calls": 0,
        "estimated_cost": 0,
    }

    _write_json(output_dir / "run_manifest.json", run_manifest)
    _write_json(output_dir / "summary.json", summary)
    _write_jsonl(output_dir / "candidate_outputs.jsonl", cached_outputs)
    _write_jsonl(output_dir / "claim_audit_rows.jsonl", claim_rows)
    _write_jsonl(output_dir / "validations.jsonl", validations)
    _write_jsonl(output_dir / "failed_cases.jsonl", failed_cases)
    (output_dir / "summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic Phase 6-B claim alignment."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config = _read_json(_resolve_path(args.config))
    output_dir = _resolve_path(args.output_dir) if args.output_dir else None
    summary = run_batch(config, output_dir=output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
