"""Run preregistered Gold-free G3 consistency auditing on Validation40."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from experiments.phase6_evidence_graph.graph_contract import (
    assert_no_gold_only_content,
)
from experiments.phase7_formal_experiments.graph_consistency_auditor import (
    audit_graph_consistency,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL row {line_number} must be an object: {path}")
        rows.append(payload)
    return rows


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", delete=False, dir=path.parent, prefix=f".{path.name}."
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def _require_empty_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")


def _verify_hash(path: Path, expected: object, label: str) -> str:
    actual = sha256_file(path)
    if not expected or actual != str(expected).strip().lower():
        raise ValueError(f"{label} SHA-256 mismatch")
    return actual


def _candidate_keys(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("candidate_key", "")) for row in rows]


def _source_pages(rows: list[dict[str, Any]]) -> list[dict[str, object]]:
    return [
        {
            "candidate_key": str(row.get("candidate_key", "")),
            "source_file": str(row.get("source_file", "")),
            "page_number": int(row.get("page_number", 0) or 0),
        }
        for row in rows
    ]


def _validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("audit_mode") != "annotate_only":
        raise ValueError("G3 audit mode must remain annotate_only")
    if contract.get("audit_scope") != "final_evidence_top4":
        raise ValueError("G3 audit scope must remain final_evidence_top4")
    mutation_policy = contract.get("mutation_policy")
    if not isinstance(mutation_policy, dict) or any(mutation_policy.values()):
        raise ValueError("G3 mutation policy must prohibit retrieval changes")
    guards = contract.get("execution_guards")
    expected_guards = {
        "validation40_only": True,
        "gold_access": False,
        "pilot_test_content_access": False,
        "external_model_calls": False,
        "clinical_validation_claimed": False,
    }
    if guards != expected_guards:
        raise ValueError("G3 Gold-free execution guards are not frozen")


def _existing_boundary_refusal(input_row: dict[str, Any]) -> bool:
    if input_row.get("upstream_boundary_refusal") is True:
        return True
    return str(input_row.get("upstream_route_action", "")) in {
        "boundary_refusal",
        "boundary_refusal_passthrough",
    }


def run_graph_consistency_audit(
    *,
    repo_root: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Annotate G2 Top-4 evidence without reading Gold or changing retrieval."""
    _require_empty_output_dir(output_dir)
    prereg_path = repo_root / str(config["preregistration_path"])
    prereg_hash = _verify_hash(
        prereg_path,
        config.get("expected_preregistration_sha256"),
        "preregistration",
    )
    contract = _read_json(prereg_path)
    _validate_contract(contract)

    input_results_path = repo_root / str(contract["input_results_path"])
    input_manifest_path = repo_root / str(contract["input_manifest_path"])
    g2_config_path = repo_root / str(contract["g2_config_path"])
    lexicon_path = repo_root / str(contract["lexicon_path"])
    pilot_test_path = repo_root / str(contract["pilot_test_path"])
    input_hashes = {
        "preregistration_sha256": prereg_hash,
        "input_results_sha256": _verify_hash(
            input_results_path,
            contract.get("expected_input_results_sha256"),
            "input results",
        ),
        "input_manifest_sha256": _verify_hash(
            input_manifest_path,
            contract.get("expected_input_manifest_sha256"),
            "input manifest",
        ),
        "g2_config_sha256": _verify_hash(
            g2_config_path,
            contract.get("expected_g2_config_sha256"),
            "G2 config",
        ),
        "lexicon_sha256": _verify_hash(
            lexicon_path,
            contract.get("expected_lexicon_sha256"),
            "runtime lexicon",
        ),
        # Pilot Test80 remains bytes-only and is never decoded.
        "pilot_test_sha256": _verify_hash(
            pilot_test_path,
            contract.get("expected_pilot_test_sha256"),
            "Pilot Test80",
        ),
    }

    input_manifest = _read_json(input_manifest_path)
    if not input_manifest.get("ready"):
        raise ValueError("frozen G2 input manifest is not ready")
    manifest_result = input_manifest.get("files", {}).get("results", {})
    if manifest_result.get("sha256") != input_hashes["input_results_sha256"]:
        raise ValueError("frozen G2 manifest does not bind the input results")
    results = _read_jsonl(input_results_path)
    lexicon = _read_json(lexicon_path)
    g2_config = _read_json(g2_config_path)
    for payload in (results, input_manifest, lexicon, g2_config):
        assert_no_gold_only_content(payload)

    expected_count = int(contract.get("expected_count", 40))
    if len(results) != expected_count:
        raise ValueError("Validation40 result count mismatch")
    input_method = str(contract["input_method"])
    output_method = str(contract["planned_output_method"])
    if input_method == output_method:
        raise ValueError("G2 input and G3 output methods must be distinct")
    candidate_field = str(contract["candidate_output_field"])
    evidence_field = str(contract["final_evidence_field"])
    required_trace_fields = set(contract["required_trace_fields"])

    output_rows: list[dict[str, Any]] = []
    manual_queue: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    trace_complete_count = 0
    candidate_identity_count = 0
    evidence_identity_count = 0
    seen_sample_ids: set[str] = set()
    for input_row in results:
        sample_id = str(input_row.get("sample_id", ""))
        if not sample_id or sample_id in seen_sample_ids:
            raise ValueError("Validation40 sample IDs must be non-empty and unique")
        seen_sample_ids.add(sample_id)
        question = str(input_row.get("question", ""))
        methods = input_row.get("methods")
        if not isinstance(methods, dict) or input_method not in methods:
            raise ValueError("frozen G2 method is missing")
        if output_method in methods:
            raise ValueError("G3 output method already exists in frozen input")
        g2_payload = methods[input_method]
        candidates = g2_payload.get(candidate_field)
        evidence = g2_payload.get(evidence_field)
        if not isinstance(candidates, list) or not isinstance(evidence, list):
            raise ValueError("frozen G2 candidate or evidence field is missing")
        if len(candidates) > int(contract["candidate_budget"]):
            raise ValueError("frozen G2 candidate budget exceeded")
        if len(evidence) > int(contract["final_evidence_k"]):
            raise ValueError("frozen G2 evidence budget exceeded")
        if len(set(_candidate_keys(candidates))) != len(candidates):
            raise ValueError("frozen G2 candidate keys must be unique")
        if len(set(_candidate_keys(evidence))) != len(evidence):
            raise ValueError("frozen G2 evidence keys must be unique")

        frozen_candidates = deepcopy(candidates)
        frozen_evidence = deepcopy(evidence)
        frozen_input_identity = _payload_sha256(g2_payload)
        trace = audit_graph_consistency(
            question,
            evidence_top4=evidence,
            lexicon=lexicon,
            contract=contract,
            upstream_boundary_refusal=_existing_boundary_refusal(input_row),
        )
        if candidates != frozen_candidates:
            raise RuntimeError("G3 auditor mutated frozen G2 candidates")
        if evidence != frozen_evidence:
            raise RuntimeError("G3 auditor mutated frozen G2 evidence")
        if _payload_sha256(g2_payload) != frozen_input_identity:
            raise RuntimeError("G3 auditor mutated frozen G2 input payload")
        trace.update(
            {
                "sample_id": sample_id,
                "input_method": input_method,
                "candidate_keys": _candidate_keys(candidates),
                "source_pages": _source_pages(evidence),
                "input_identity_sha256": frozen_input_identity,
            }
        )
        if not required_trace_fields <= set(trace):
            raise RuntimeError("G3 audit trace is incomplete")
        trace_complete_count += 1
        action_counts[trace["route_action"]] += 1
        label_counts.update(trace["summary_labels"])

        g3_payload = deepcopy(g2_payload)
        g3_payload["graph_consistency_audit"] = trace
        if g3_payload[candidate_field] != frozen_candidates:
            raise RuntimeError("G3 changed frozen G2 candidate objects or order")
        candidate_identity_count += 1
        if g3_payload[evidence_field] != frozen_evidence:
            raise RuntimeError("G3 changed frozen G2 evidence objects or order")
        evidence_identity_count += 1
        output_row = deepcopy(input_row)
        output_row["methods"][output_method] = g3_payload
        output_rows.append(output_row)

        if not str(trace["route_action"]).startswith("allow_"):
            manual_queue.append(
                {
                    "sample_id": sample_id,
                    "route_action": trace["route_action"],
                    "route_reasons": trace["route_reasons"],
                    "summary_labels": trace["summary_labels"],
                    "input_identity_sha256": trace["input_identity_sha256"],
                    "review_status": "pending_manual_adjudication",
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / str(config["results_filename"])
    audit_path = output_dir / str(config["audit_filename"])
    manual_queue_path = output_dir / str(config["manual_queue_filename"])
    manifest_path = output_dir / str(config["manifest_filename"])
    _atomic_write(results_path, _jsonl_bytes(output_rows))
    audit = {
        "audit_version": str(config["audit_version"]),
        "phase": str(config["phase"]),
        "dataset_version": str(contract["dataset_version"]),
        "kb_version": str(contract["kb_version"]),
        "sample_count": len(output_rows),
        "trace_complete_count": trace_complete_count,
        "trace_completeness": trace_complete_count / len(output_rows),
        "candidate_identity_count": candidate_identity_count,
        "evidence_identity_count": evidence_identity_count,
        "manual_adjudication_count": len(manual_queue),
        "route_action_counts": dict(sorted(action_counts.items())),
        "summary_label_counts": dict(sorted(label_counts.items())),
        "input_hashes": input_hashes,
        "retrieval_metrics_changed": False,
        "retrieval_gain_claimed": False,
        "safety_gain_claimed": False,
        "gold_accessed": False,
        "pilot_test_accessed": False,
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0.0,
        "clinical_validation_claimed": False,
    }
    _atomic_write(audit_path, _json_bytes(audit))
    _atomic_write(manual_queue_path, _jsonl_bytes(manual_queue))
    manifest = {
        "manifest_version": str(config["manifest_version"]),
        "ready": True,
        "files": {
            "results": {
                "path": results_path.name,
                "sha256": sha256_file(results_path),
            },
            "audit": {
                "path": audit_path.name,
                "sha256": sha256_file(audit_path),
            },
            "manual_queue": {
                "path": manual_queue_path.name,
                "sha256": sha256_file(manual_queue_path),
            },
        },
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0.0,
    }
    _atomic_write(manifest_path, _json_bytes(manifest))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    config = _read_json(args.config)
    audit = run_graph_consistency_audit(
        repo_root=root,
        output_dir=args.output_dir or root / str(config["output_dir"]),
        config=config,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
