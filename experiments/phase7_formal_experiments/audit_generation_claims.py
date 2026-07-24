"""Audit Phase 7 generated answers against their admitted runtime evidence."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.phase6_evidence_graph.claim_evidence_aligner import align_claims
from experiments.phase6_evidence_graph.graph_contract import (
    assert_no_gold_only_content,
)
from experiments.phase6_evidence_graph.graph_reranker import canonical_sha256


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_VERSION = "phase7-generation-claim-audit-v0.1"
AUDIT_METHOD_ID = "phase7_generation_claim_evidence_alignment"


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
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


def _index_unique(
    rows: list[dict[str, Any]], artifact_name: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        cache_key = str(row.get("cache_key") or "").strip()
        if not cache_key:
            raise ValueError(f"missing cache_key in {artifact_name}")
        if cache_key in indexed:
            raise ValueError(f"duplicate cache_key in {artifact_name}: {cache_key}")
        indexed[cache_key] = row
    return indexed


def _method_evidence_view(
    parent: dict[str, Any], method_id: str
) -> dict[str, Any]:
    rank_field_by_method = {
        "vector_only_rag": "rank_before",
        "graph_enhanced_rag": "rank_after",
    }
    rank_field = rank_field_by_method.get(method_id)
    if rank_field is None:
        raise ValueError(f"unsupported generation method: {method_id}")

    view = copy.deepcopy(parent)
    evidence_rows = view.get("ranked_evidence")
    if not isinstance(evidence_rows, list):
        raise ValueError("parent ranked_evidence must be a list")
    if any(rank_field not in row for row in evidence_rows):
        raise ValueError(f"missing {rank_field} in parent ranked_evidence")
    view["ranked_evidence"] = sorted(
        evidence_rows, key=lambda row: int(row[rank_field])
    )
    return view


def _claim_rows(
    artifact: dict[str, Any],
    generation_row: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for claim_index, claim in enumerate(artifact["claims"], start=1):
        rows.append(
            {
                "sample_id": generation_row["sample_id"],
                "method_id": generation_row["method_id"],
                "model_provider": generation_row.get("model_provider"),
                "model_name": generation_row.get("model_name"),
                "inference_profile": generation_row.get("inference_profile"),
                "prompt_version": generation_row.get("prompt_version"),
                "dataset_version": generation_row.get("dataset_version"),
                "kb_version": generation_row.get("kb_version"),
                "cache_key": generation_row["cache_key"],
                "claim_index": claim_index,
                "claim_id": claim["claim_id"],
                "claim_text": claim["claim_text"],
                "stance": claim["stance"],
                "support_state": claim["support_state"],
                "reason_codes": list(claim.get("reason_codes") or []),
                "supporting_evidence_ids": list(
                    claim.get("supporting_evidence_ids") or []
                ),
                "contradicting_evidence_ids": list(
                    claim.get("contradicting_evidence_ids") or []
                ),
            }
        )
    return rows


def _summary_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Phase 7 generation claim audit",
            "",
            f"- Total outputs: `{summary['total_outputs']}`",
            f"- Audited outputs: `{summary['audited_outputs']}`",
            f"- Supported: `{summary['supported_count']}`",
            f"- Contradicted: `{summary['contradicted_count']}`",
            f"- Unsupported: `{summary['unsupported_count']}`",
            (
                "- Insufficient evidence: "
                f"`{summary['insufficient_evidence_count']}`"
            ),
            f"- Boundary refusal: `{summary['boundary_refusal_count']}`",
            f"- Failed: `{summary['failed_count']}`",
            f"- Total atomic claims: `{summary['total_claim_count']}`",
            f"- Claim states: `{json.dumps(summary['claim_state_counts'], ensure_ascii=False, sort_keys=True)}`",
            "",
            "This deterministic development audit is not a clinical expert evaluation.",
            "",
        ]
    )


def audit_generation_run(
    generation_run_dir: str | Path,
    *,
    output_dir: str | Path,
    evidence_budget: int = 4,
) -> dict[str, Any]:
    """Audit one Phase 7 generation run without external model calls."""

    if evidence_budget < 1:
        raise ValueError("evidence_budget must be a positive integer")
    generation_dir = _resolve_path(generation_run_dir)
    resolved_output_dir = _resolve_path(output_dir)
    raw_path = generation_dir / "raw_model_outputs.jsonl"
    metadata_path = generation_dir / "evaluation_metadata.jsonl"
    if not raw_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "generation run must contain raw_model_outputs.jsonl "
            "and evaluation_metadata.jsonl"
        )

    generation_rows = _load_jsonl(raw_path)
    generation_by_key = _index_unique(generation_rows, raw_path.name)
    metadata_by_key = _index_unique(_load_jsonl(metadata_path), metadata_path.name)
    if set(generation_by_key) != set(metadata_by_key):
        raise ValueError(
            "cache_key sets differ across generation outputs and evaluation metadata"
        )
    resolved_output_dir.mkdir(parents=True, exist_ok=False)
    artifact_dir = resolved_output_dir / "claim_alignment_artifacts"
    artifact_dir.mkdir()

    artifacts: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    failed_cases: list[dict[str, Any]] = []
    for cache_key, generation_row in generation_by_key.items():
        metadata = metadata_by_key[cache_key]
        try:
            for field in ("sample_id", "method_id"):
                if generation_row.get(field) != metadata.get(field):
                    raise ValueError(f"{field} mismatch for cache_key {cache_key}")
            if generation_row.get("status") not in {"success", "cache_hit"}:
                raise ValueError(
                    f"generation status is not auditable: "
                    f"{generation_row.get('status')}"
                )
            answer_text = str(generation_row.get("raw_output") or "").strip()
            if not answer_text:
                raise ValueError("raw_output must be non-empty")

            parent_artifact_value = str(
                metadata.get("parent_reranking_artifact") or ""
            ).strip()
            if not parent_artifact_value:
                raise ValueError("missing parent_reranking_artifact")
            parent_path = _resolve_path(parent_artifact_value)
            if not parent_path.is_file():
                raise FileNotFoundError(
                    f"parent reranking artifact not found: {parent_path}"
                )
            before_bytes = parent_path.read_bytes()
            parent = json.loads(before_bytes.decode("utf-8"))
            assert_no_gold_only_content(parent)
            method_view = _method_evidence_view(
                parent, str(generation_row["method_id"])
            )
            aligner_config = {
                "config_version": AUDIT_VERSION,
                "method_id": AUDIT_METHOD_ID,
                "method_version": AUDIT_VERSION,
                "candidate_output_origin": str(generation_dir),
                "evidence_budget": evidence_budget,
            }
            first = align_claims(method_view, answer_text, aligner_config)
            second = align_claims(method_view, answer_text, aligner_config)
            deterministic = first == second
            parent_unchanged = before_bytes == parent_path.read_bytes()
            if not deterministic:
                raise RuntimeError("claim audit is not deterministic")
            if not parent_unchanged:
                raise RuntimeError("parent reranking artifact changed during audit")
            assert_no_gold_only_content(first)

            generation_context = {
                key: generation_row.get(key)
                for key in (
                    "sample_id",
                    "method_id",
                    "model_provider",
                    "model_name",
                    "inference_profile",
                    "prompt_version",
                    "dataset_version",
                    "kb_version",
                    "cache_key",
                )
            }
            first["generation_context"] = generation_context
            first.pop("artifact_sha256", None)
            assert_no_gold_only_content(first)
            first["artifact_sha256"] = canonical_sha256(first)
            _write_json(artifact_dir / f"{cache_key}.json", first)
            artifacts.append(first)
            claim_rows.extend(_claim_rows(first, generation_row))
            validations.append(
                {
                    **generation_context,
                    "overall_support_state": first["overall_support_state"],
                    "deterministic": deterministic,
                    "parent_artifact_unchanged": parent_unchanged,
                    "gold_leakage_check": "passed",
                    "parent_artifact_path": str(parent_path),
                    "parent_artifact_sha256": canonical_sha256(parent),
                    "method_evidence_order": (
                        "rank_before"
                        if generation_row["method_id"] == "vector_only_rag"
                        else "rank_after"
                    ),
                }
            )
        except Exception as exc:  # noqa: BLE001 - preserve all failed cases.
            failed_cases.append(
                {
                    "cache_key": cache_key,
                    "sample_id": generation_row.get("sample_id"),
                    "method_id": generation_row.get("method_id"),
                    "model_name": generation_row.get("model_name"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    state_counts = Counter(
        artifact["overall_support_state"] for artifact in artifacts
    )
    claim_state_counts = Counter(row["support_state"] for row in claim_rows)
    summary = {
        "audit_version": AUDIT_VERSION,
        "generation_run_dir": str(generation_dir),
        "output_dir": str(resolved_output_dir),
        "evidence_budget": evidence_budget,
        "total_outputs": len(generation_rows),
        "audited_outputs": len(artifacts),
        "supported_count": state_counts.get("supported", 0),
        "contradicted_count": state_counts.get("contradicted", 0),
        "unsupported_count": state_counts.get("unsupported", 0),
        "insufficient_evidence_count": state_counts.get(
            "insufficient_evidence", 0
        ),
        "boundary_refusal_count": state_counts.get("not_applicable", 0),
        "failed_count": len(failed_cases),
        "total_claim_count": len(claim_rows),
        "claim_state_counts": {
            state: claim_state_counts.get(state, 0)
            for state in (
                "supported",
                "contradicted",
                "unsupported",
                "insufficient_evidence",
            )
        },
        "external_model_calls": 0,
        "estimated_cost_cny": 0.0,
    }
    manifest = {
        "audit_version": AUDIT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation_run_dir": str(generation_dir),
        "output_dir": str(resolved_output_dir),
        "evidence_budget": evidence_budget,
        "external_model_calls": 0,
        "estimated_cost_cny": 0.0,
    }
    _write_json(resolved_output_dir / "run_manifest.json", manifest)
    _write_json(resolved_output_dir / "summary.json", summary)
    _write_jsonl(resolved_output_dir / "claim_audit_rows.jsonl", claim_rows)
    _write_jsonl(resolved_output_dir / "validations.jsonl", validations)
    _write_jsonl(resolved_output_dir / "failed_cases.jsonl", failed_cases)
    (resolved_output_dir / "summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase 7 generated answers against runtime evidence."
    )
    parser.add_argument("--generation-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evidence-budget", type=int, default=4)
    args = parser.parse_args()
    summary = audit_generation_run(
        args.generation_run_dir,
        output_dir=args.output_dir,
        evidence_budget=args.evidence_budget,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
