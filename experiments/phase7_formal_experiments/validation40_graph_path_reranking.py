"""Gold-free G2 graph-path reranking over frozen Validation40 G1 candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from experiments.phase6_evidence_graph.graph_contract import (
    assert_no_gold_only_content,
)
from experiments.phase7_formal_experiments import graph_path_reranker as g2
from experiments.phase7_formal_experiments import runtime_graph_path_router as router
from experiments.phase7_formal_experiments.validation40_graph_candidate_paired_retrieval import (
    _deduplicate_evidence,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _json_bytes(payload: Any) -> bytes:
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


def _validate_execution_guards(config: dict[str, Any]) -> None:
    guards = config.get("execution_guards")
    expected = {
        "validation40_only": True,
        "gold_access": False,
        "pilot_test_content_access": False,
        "external_model_calls": False,
        "clinical_validation_claimed": False,
    }
    if guards != expected:
        raise ValueError("Gold-free G2 execution guards are not frozen")


def _load_frozen_inputs(
    *,
    input_results_path: Path,
    input_manifest_path: Path,
    graph_manifest_path: Path,
    lexicon_path: Path,
    routing_audit_path: Path,
    routing_manifest_path: Path,
    pilot_test_path: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    _validate_execution_guards(config)
    observed = {
        "input_results_sha256": _verify_hash(
            input_results_path,
            config.get("expected_input_results_sha256"),
            "input results",
        ),
        "input_manifest_sha256": _verify_hash(
            input_manifest_path,
            config.get("expected_input_manifest_sha256"),
            "input manifest",
        ),
        "graph_manifest_sha256": _verify_hash(
            graph_manifest_path,
            config.get("expected_graph_manifest_sha256"),
            "graph manifest",
        ),
        "lexicon_sha256": _verify_hash(
            lexicon_path,
            config.get("expected_lexicon_sha256"),
            "runtime lexicon",
        ),
        "routing_audit_sha256": _verify_hash(
            routing_audit_path,
            config.get("expected_routing_audit_sha256"),
            "routing audit",
        ),
        "routing_manifest_sha256": _verify_hash(
            routing_manifest_path,
            config.get("expected_routing_manifest_sha256"),
            "routing manifest",
        ),
        # Pilot Test80 is intentionally hashed as bytes and never decoded.
        "pilot_test_sha256": _verify_hash(
            pilot_test_path,
            config.get("expected_pilot_test_sha256"),
            "Pilot Test80",
        ),
    }

    input_manifest = _read_json(input_manifest_path)
    if not input_manifest.get("ready"):
        raise ValueError("frozen G1 input manifest is not ready")
    manifest_result = input_manifest.get("files", {}).get("results", {})
    if manifest_result.get("sha256") != observed["input_results_sha256"]:
        raise ValueError("frozen G1 manifest does not bind the input results")

    graph_manifest = _read_json(graph_manifest_path)
    if not graph_manifest.get("ready"):
        raise ValueError("graph manifest is not ready")
    graph_file = graph_manifest.get("files", {}).get("graph_index", {})
    graph_index_path = graph_manifest_path.parent / str(graph_file.get("path", ""))
    observed["graph_index_sha256"] = _verify_hash(
        graph_index_path, graph_file.get("sha256"), "graph index"
    )

    routing_manifest = _read_json(routing_manifest_path)
    if routing_manifest.get("router_version") != router.ROUTER_VERSION:
        raise ValueError("routing manifest router version mismatch")

    results = _read_jsonl(input_results_path)
    graph_index = _read_json(graph_index_path)
    lexicon = _read_json(lexicon_path)
    routing_audit = _read_json(routing_audit_path)
    for payload in (results, graph_index, lexicon, routing_audit):
        assert_no_gold_only_content(payload)
    return results, graph_index, lexicon, routing_audit, observed


def run_graph_path_reranking(
    *,
    input_results_path: Path,
    input_manifest_path: Path,
    graph_manifest_path: Path,
    lexicon_path: Path,
    routing_audit_path: Path,
    routing_manifest_path: Path,
    pilot_test_path: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Create G2 results without opening Validation40 Gold or Pilot Test80 content."""
    _require_empty_output_dir(output_dir)
    started = time.perf_counter()
    results, graph_index, lexicon, routing_audit, input_hashes = _load_frozen_inputs(
        input_results_path=input_results_path,
        input_manifest_path=input_manifest_path,
        graph_manifest_path=graph_manifest_path,
        lexicon_path=lexicon_path,
        routing_audit_path=routing_audit_path,
        routing_manifest_path=routing_manifest_path,
        pilot_test_path=pilot_test_path,
        config=config,
    )

    expected_count = int(config.get("expected_count", 40))
    if len(results) != expected_count:
        raise ValueError("Validation40 result count mismatch")
    route_details = routing_audit.get("question_details")
    if not isinstance(route_details, list) or len(route_details) != expected_count:
        raise ValueError("routing audit question count mismatch")
    route_by_id = {str(row.get("sample_id")): row for row in route_details}
    if len(route_by_id) != len(route_details):
        raise ValueError("routing audit sample IDs must be unique")

    candidate_field = str(config.get("candidate_output_field", "candidates_top24"))
    f_method = str(config["f_method"])
    g1_method = str(config["g1_method"])
    g2_method = str(config["g2_method"])
    if len({f_method, g1_method, g2_method}) != 3:
        raise ValueError("F, G1 and G2 method IDs must be distinct")
    final_evidence_k = int(config.get("final_evidence_k", 4))
    dedup_ngram_size = int(config.get("dedup_ngram_size", 3))
    dedup_overlap_threshold = float(config.get("dedup_overlap_threshold", 0.75))
    max_rank_shift = float(config.get("max_rank_shift", 2.0))
    tier_weights = config.get("tier_weights")
    if not isinstance(tier_weights, dict):
        raise ValueError("tier_weights must be a dictionary")

    output_rows: list[dict[str, Any]] = []
    eligible_origin_counts: Counter[str] = Counter()
    path_eligible_count = 0
    existing_trace_count = 0
    zero_shift_identity_count = 0
    candidate_set_preserved_count = 0
    candidate_order_changed_count = 0
    evidence_changed_count = 0
    no_eligible_sample_count = 0
    for input_row in results:
        sample_id = str(input_row.get("sample_id", ""))
        question = str(input_row.get("question", ""))
        route_detail = route_by_id.get(sample_id)
        if route_detail is None:
            raise ValueError("sample is missing from routing audit")
        methods = input_row.get("methods")
        if not isinstance(methods, dict) or f_method not in methods or g1_method not in methods:
            raise ValueError("frozen F/G1 methods are missing")
        g1_payload = methods[g1_method]
        g1_candidates = g1_payload.get(candidate_field)
        if not isinstance(g1_candidates, list):
            raise ValueError("frozen G1 candidate field is missing")
        if len(g1_candidates) > int(config.get("candidate_budget", 24)):
            raise ValueError("frozen G1 candidate budget exceeded")
        selected_paths = route_detail.get("selected_paths")
        if not isinstance(selected_paths, list):
            raise ValueError("routing audit selected paths are missing")

        zero_ranked, zero_audit = g2.graph_rerank_candidates(
            question=question,
            candidates=g1_candidates,
            selected_paths=selected_paths,
            graph_index=graph_index,
            lexicon=lexicon,
            tier_weights=tier_weights,
            max_rank_shift=0.0,
            allow_specific_condition_class_path=bool(
                config.get("allow_specific_condition_class_path", True)
            ),
        )
        zero_evidence, _ = _deduplicate_evidence(
            zero_ranked,
            max_evidence=final_evidence_k,
            ngram_size=dedup_ngram_size,
            overlap_threshold=dedup_overlap_threshold,
        )
        if _candidate_keys(zero_ranked) != _candidate_keys(g1_candidates):
            raise RuntimeError("zero-shift graph reranker changed G1 candidate order")
        if _candidate_keys(zero_evidence) != _candidate_keys(g1_payload["evidence_top4"]):
            raise RuntimeError("zero-shift graph reranker changed G1 evidence order")
        zero_shift_identity_count += 1

        ranked, sample_audit = g2.graph_rerank_candidates(
            question=question,
            candidates=g1_candidates,
            selected_paths=selected_paths,
            graph_index=graph_index,
            lexicon=lexicon,
            tier_weights=tier_weights,
            max_rank_shift=max_rank_shift,
            allow_specific_condition_class_path=bool(
                config.get("allow_specific_condition_class_path", True)
            ),
        )
        evidence, dedup_audit = _deduplicate_evidence(
            ranked,
            max_evidence=final_evidence_k,
            ngram_size=dedup_ngram_size,
            overlap_threshold=dedup_overlap_threshold,
        )
        if set(_candidate_keys(ranked)) != set(_candidate_keys(g1_candidates)):
            raise RuntimeError("G2 changed the frozen G1 candidate set")
        candidate_set_preserved_count += 1
        path_eligible_count += int(sample_audit["path_eligible_count"])
        existing_trace_count += int(sample_audit["existing_trace_verified_count"])
        eligible_origin_counts.update(sample_audit["path_eligible_origin_counts"])
        no_eligible_sample_count += int(sample_audit["path_eligible_count"] == 0)
        candidate_order_changed_count += int(
            _candidate_keys(ranked) != _candidate_keys(g1_candidates)
        )
        evidence_changed_count += int(
            _candidate_keys(evidence) != _candidate_keys(g1_payload["evidence_top4"])
        )

        output_rows.append(
            {
                "sample_id": sample_id,
                "question": question,
                "methods": {
                    f_method: deepcopy(methods[f_method]),
                    g1_method: deepcopy(g1_payload),
                    g2_method: {
                        candidate_field: ranked,
                        "evidence_top4": evidence,
                        "dedup_audit": dedup_audit,
                    },
                },
                "graph_rerank_audit": {
                    **sample_audit,
                    "zero_shift_identity": True,
                    "candidate_set_equal_to_g1": True,
                    "candidate_order_changed": _candidate_keys(ranked)
                    != _candidate_keys(g1_candidates),
                    "evidence_order_changed": _candidate_keys(evidence)
                    != _candidate_keys(g1_payload["evidence_top4"]),
                    "zero_shift_path_eligible_count": zero_audit[
                        "path_eligible_count"
                    ],
                },
            }
        )

    expected_origin_counts = {
        str(key): int(value)
        for key, value in config.get("expected_path_eligible_origin_counts", {}).items()
    }
    actual_origin_counts = dict(sorted(eligible_origin_counts.items()))
    if path_eligible_count != int(config.get("expected_path_eligible_count", -1)):
        raise RuntimeError("path-eligible candidate count changed")
    if existing_trace_count != int(config.get("expected_existing_trace_count", -1)):
        raise RuntimeError("existing graph trace verification count changed")
    if actual_origin_counts != dict(sorted(expected_origin_counts.items())):
        raise RuntimeError("path-eligible origin audit changed")

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / str(config["results_filename"])
    audit_path = output_dir / str(config["audit_filename"])
    manifest_path = output_dir / str(config["manifest_filename"])
    _atomic_write(results_path, _jsonl_bytes(output_rows))
    audit = {
        "audit_version": str(config["audit_version"]),
        "phase": str(config["phase"]),
        "dataset_version": str(config["dataset_version"]),
        "kb_version": str(config["kb_version"]),
        "sample_count": len(output_rows),
        "candidate_budget": int(config.get("candidate_budget", 24)),
        "final_evidence_k": final_evidence_k,
        "max_rank_shift": max_rank_shift,
        "path_eligible_count": path_eligible_count,
        "existing_trace_verified_count": existing_trace_count,
        "path_eligible_origin_counts": actual_origin_counts,
        "zero_shift_identity_count": zero_shift_identity_count,
        "candidate_set_preserved_count": candidate_set_preserved_count,
        "candidate_order_changed_sample_count": candidate_order_changed_count,
        "evidence_changed_sample_count": evidence_changed_count,
        "no_eligible_sample_count": no_eligible_sample_count,
        "candidate_origin_used_for_score": False,
        "input_hashes": input_hashes,
        "gold_accessed": False,
        "pilot_test_accessed": False,
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0.0,
        "clinical_validation_claimed": False,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    _atomic_write(audit_path, _json_bytes(audit))
    manifest = {
        "manifest_version": str(config["manifest_version"]),
        "ready": True,
        "files": {
            "results": {"path": results_path.name, "sha256": sha256_file(results_path)},
            "audit": {"path": audit_path.name, "sha256": sha256_file(audit_path)},
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
    audit = run_graph_path_reranking(
        input_results_path=root / config["input_results_path"],
        input_manifest_path=root / config["input_manifest_path"],
        graph_manifest_path=root / config["graph_manifest_path"],
        lexicon_path=root / config["lexicon_path"],
        routing_audit_path=root / config["routing_audit_path"],
        routing_manifest_path=root / config["routing_manifest_path"],
        pilot_test_path=root / config["pilot_test_path"],
        output_dir=args.output_dir or root / config["output_dir"],
        config=config,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
