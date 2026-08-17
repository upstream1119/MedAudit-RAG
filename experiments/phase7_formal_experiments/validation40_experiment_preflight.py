from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

RUNTIME_FIELDS = (
    "sample_id",
    "question",
    "dataset_version",
    "kb_version",
)

REQUIRED_METHOD_IDS = (
    "vanilla_llm",
    "naive_rag",
    "multi_granularity_rag",
    "trust_gated_rag",
    "graph_enhanced_full",
)

GOLD_ONLY_FIELDS = {
    "expected_decision",
    "required_evidence_type",
    "required_claims",
    "allowed_claims",
    "forbidden_claims",
    "risk_labels",
    "missing_evidence_type",
    "missing_information",
    "gold_evidence_status",
    "anchor_text_span",
    "page_number",
    "source_title",
    "source_filename",
}

OUTPUT_FILENAMES = {
    "retrieval_plan": "validation40_retrieval_plan_v0_1.jsonl",
    "method_call_plan": "validation40_method_call_plan_v0_1.jsonl",
    "audit": "validation40_experiment_preflight_audit_v0_1.json",
    "summary_markdown": "validation40_experiment_preflight_summary_v0_1.md",
}


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def compute_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _canonical_jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_number}: {path}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row must be an object at line {line_number}: {path}")
        rows.append(row)
    return rows


def _validate_config(config: dict[str, Any]) -> None:
    required_fields = (
        "config_version",
        "preflight_version",
        "expected_runtime_sha256",
        "expected_dataset_version",
        "expected_kb_version",
        "prompt_version",
    )
    missing = [field for field in required_fields if not _text(config.get(field))]
    if missing:
        raise ValueError(f"config fields missing: {', '.join(missing)}")

    if bool(config.get("execute_retrieval")) or bool(config.get("execute_model_calls")):
        raise ValueError("preflight config must remain non-executing")

    expected_count = int(config.get("expected_runtime_count", 0))
    if expected_count <= 0:
        raise ValueError("expected_runtime_count must be positive")

    context_min = int(config.get("evidence_context_min", 0))
    context_max = int(config.get("evidence_context_max", 0))
    if context_min <= 0 or context_max < context_min:
        raise ValueError("invalid evidence context bounds")

    methods = config.get("methods")
    if not isinstance(methods, list):
        raise ValueError("method matrix must be a list")
    method_ids = [_text(row.get("method_id")) for row in methods if isinstance(row, dict)]
    if tuple(method_ids) != REQUIRED_METHOD_IDS:
        raise ValueError(
            "method matrix must contain the five required methods in fixed order"
        )

    method_versions: set[str] = set()
    for method in methods:
        method_id = _text(method.get("method_id"))
        method_version = _text(method.get("method_version"))
        if not method_version or method_version in method_versions:
            raise ValueError(f"invalid or duplicate method_version for {method_id}")
        method_versions.add(method_version)

        retrieval_required = bool(method.get("retrieval_required"))
        retrieval_mode = _text(method.get("retrieval_mode"))
        retrieval_top_k = int(method.get("retrieval_top_k", -1))
        if method_id == "vanilla_llm":
            if retrieval_required or retrieval_mode != "none" or retrieval_top_k != 0:
                raise ValueError("vanilla_llm must not use retrieval")
        elif (
            not retrieval_required
            or retrieval_mode == "none"
            or retrieval_top_k < context_min
            or retrieval_top_k > context_max
        ):
            raise ValueError(f"invalid retrieval contract for {method_id}")

        if bool(method.get("use_graph")) != (method_id == "graph_enhanced_full"):
            raise ValueError(f"invalid graph flag for {method_id}")
        expected_gate = method_id in {"trust_gated_rag", "graph_enhanced_full"}
        if bool(method.get("use_trust_gate")) != expected_gate:
            raise ValueError(f"invalid trust gate flag for {method_id}")

    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("at least one model is required")
    model_keys: set[tuple[str, str, str]] = set()
    for model in models:
        if not isinstance(model, dict):
            raise ValueError("model matrix rows must be objects")
        model_key = (
            _text(model.get("model_provider")),
            _text(model.get("model_name")),
            _text(model.get("model_version")),
        )
        if not all(model_key) or model_key in model_keys:
            raise ValueError("model matrix contains an invalid or duplicate model")
        model_keys.add(model_key)


def _validate_runtime_rows(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> None:
    expected_count = int(config["expected_runtime_count"])
    if len(rows) != expected_count:
        raise ValueError(
            f"record count mismatch: expected={expected_count}, observed={len(rows)}"
        )

    allowed_fields = set(RUNTIME_FIELDS)
    seen_ids: set[str] = set()
    for row in rows:
        if set(row) != allowed_fields:
            raise ValueError(
                "runtime field allowlist mismatch: "
                f"expected={sorted(allowed_fields)}, observed={sorted(row)}"
            )
        sample_id = _text(row.get("sample_id"))
        question = _text(row.get("question"))
        if not sample_id or not question:
            raise ValueError("sample_id and question are required")
        if sample_id in seen_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen_ids.add(sample_id)

        dataset_version = _text(row.get("dataset_version"))
        if dataset_version != _text(config["expected_dataset_version"]):
            raise ValueError(
                f"{sample_id} dataset_version mismatch: {dataset_version}"
            )
        kb_version = _text(row.get("kb_version"))
        if kb_version != _text(config["expected_kb_version"]):
            raise ValueError(f"{sample_id} kb_version mismatch: {kb_version}")


def _count_gold_field_leakage(payload: Any) -> int:
    count = 0
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in GOLD_ONLY_FIELDS:
                count += 1
            count += _count_gold_field_leakage(value)
    elif isinstance(payload, list):
        count += sum(_count_gold_field_leakage(value) for value in payload)
    return count


def build_experiment_preflight(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    observed_runtime_sha256: str,
) -> dict[str, Any]:
    _validate_config(config)

    expected_hash = _text(config["expected_runtime_sha256"]).lower()
    observed_hash = _text(observed_runtime_sha256).lower()
    if observed_hash != expected_hash:
        raise ValueError(
            f"runtime SHA-256 mismatch: expected={expected_hash}, observed={observed_hash}"
        )
    _validate_runtime_rows(rows, config)

    retrieval_plan: list[dict[str, Any]] = []
    method_call_plan: list[dict[str, Any]] = []
    seed = int(config.get("seed", 0))
    prompt_version = _text(config["prompt_version"])
    preflight_version = _text(config["preflight_version"])
    context_min = int(config["evidence_context_min"])
    context_max = int(config["evidence_context_max"])

    for row in rows:
        for method in config["methods"]:
            method_id = _text(method["method_id"])
            method_version = _text(method["method_version"])
            retrieval_cache_key: str | None = None
            if bool(method["retrieval_required"]):
                retrieval_key_payload = {
                    "sample_id": row["sample_id"],
                    "question": row["question"],
                    "dataset_version": row["dataset_version"],
                    "kb_version": row["kb_version"],
                    "method_id": method_id,
                    "method_version": method_version,
                    "retrieval_mode": method["retrieval_mode"],
                    "retrieval_top_k": int(method["retrieval_top_k"]),
                    "use_graph": bool(method["use_graph"]),
                    "seed": seed,
                    "preflight_version": preflight_version,
                }
                retrieval_cache_key = _stable_sha256(retrieval_key_payload)
                retrieval_plan.append(
                    {
                        **retrieval_key_payload,
                        "retrieval_task_id": f"RET-{retrieval_cache_key[:16]}",
                        "retrieval_cache_key": retrieval_cache_key,
                        "evidence_context_min": context_min,
                        "evidence_context_max": context_max,
                        "execution_status": "planned_not_executed",
                    }
                )

            for model in config["models"]:
                call_key_payload = {
                    "sample_id": row["sample_id"],
                    "question": row["question"],
                    "dataset_version": row["dataset_version"],
                    "kb_version": row["kb_version"],
                    "method_id": method_id,
                    "method_version": method_version,
                    "model_provider": model["model_provider"],
                    "model_name": model["model_name"],
                    "model_version": model["model_version"],
                    "prompt_version": prompt_version,
                    "retrieval_cache_key": retrieval_cache_key,
                    "use_trust_gate": bool(method["use_trust_gate"]),
                    "use_graph": bool(method["use_graph"]),
                    "seed": seed,
                    "preflight_version": preflight_version,
                }
                cache_key = _stable_sha256(call_key_payload)
                method_call_plan.append(
                    {
                        **call_key_payload,
                        "call_id": f"CALL-{cache_key[:16]}",
                        "cache_key": cache_key,
                        "evidence_context_min": (
                            context_min if retrieval_cache_key is not None else 0
                        ),
                        "evidence_context_max": (
                            context_max if retrieval_cache_key is not None else 0
                        ),
                        "execution_status": "planned_not_executed",
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "estimated_cost": 0,
                    }
                )

    leakage_count = _count_gold_field_leakage(
        {"retrieval_plan": retrieval_plan, "method_call_plan": method_call_plan}
    )
    if leakage_count:
        raise ValueError(f"Gold field leakage detected: {leakage_count}")

    audit = {
        "status": "validation40_experiment_preflight_ready",
        "config_version": config["config_version"],
        "preflight_version": preflight_version,
        "runtime_projection_sha256": observed_hash,
        "runtime_record_count": len(rows),
        "dataset_version": config["expected_dataset_version"],
        "kb_version": config["expected_kb_version"],
        "prompt_version": prompt_version,
        "seed": seed,
        "method_ids": list(REQUIRED_METHOD_IDS),
        "model_count": len(config["models"]),
        "retrieval_task_count": len(retrieval_plan),
        "method_call_count": len(method_call_plan),
        "gold_field_leakage_count": 0,
        "pilot_test_accessed": False,
        "retrieval_executed": False,
        "model_calls_executed": False,
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
        "clinically_validated": False,
    }
    summary_markdown = "\n".join(
        [
            "# Validation40 experiment preflight v0.1",
            "",
            f"- Status: `{audit['status']}`",
            f"- Runtime records: `{audit['runtime_record_count']}`",
            f"- Retrieval tasks: `{audit['retrieval_task_count']}`",
            f"- Planned method calls: `{audit['method_call_count']}`",
            f"- Methods: `{', '.join(REQUIRED_METHOD_IDS)}`",
            f"- Model: `{config['models'][0]['model_provider']}/{config['models'][0]['model_name']}`",
            "- Gold field leakage: `0`",
            "- Pilot Test80 accessed: `false`",
            "- Retrieval executed: `false`",
            "- External model calls: `0`",
            "- Clinically validated: `false`",
            "",
        ]
    )
    return {
        "retrieval_plan": retrieval_plan,
        "method_call_plan": method_call_plan,
        "audit": audit,
        "summary_markdown": summary_markdown,
    }


def _preflight_payloads(result: dict[str, Any]) -> dict[str, bytes]:
    return {
        OUTPUT_FILENAMES["retrieval_plan"]: _canonical_jsonl_bytes(
            result["retrieval_plan"]
        ),
        OUTPUT_FILENAMES["method_call_plan"]: _canonical_jsonl_bytes(
            result["method_call_plan"]
        ),
        OUTPUT_FILENAMES["audit"]: _canonical_json_bytes(result["audit"]),
        OUTPUT_FILENAMES["summary_markdown"]: result["summary_markdown"].encode(
            "utf-8"
        ),
    }


def write_preflight_outputs(
    result: dict[str, Any], output_dir: str | Path
) -> dict[str, str]:
    output_path = Path(output_dir)
    payloads = _preflight_payloads(result)

    for filename, content in payloads.items():
        target = output_path / filename
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"immutable output conflict: refusing to overwrite {target}")

    output_path.mkdir(parents=True, exist_ok=True)
    for filename, content in payloads.items():
        target = output_path / filename
        if target.exists():
            continue
        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False, dir=output_path, prefix=f".{filename}."
        ) as handle:
            handle.write(content)
            temp_path = Path(handle.name)
        temp_path.replace(target)

    return {
        filename: _sha256_bytes(content) for filename, content in payloads.items()
    }


def run(config_path: str | Path, *, repo_root: str | Path = ROOT) -> dict[str, Any]:
    config_file = Path(config_path)
    config = json.loads(config_file.read_text(encoding="utf-8-sig"))
    root = Path(repo_root)
    runtime_path = root / _text(config.get("runtime_projection_path"))
    output_dir = root / _text(config.get("output_dir"))
    observed_hash = compute_sha256(runtime_path)
    rows = _read_jsonl(runtime_path)
    result = build_experiment_preflight(
        rows,
        config,
        observed_runtime_sha256=observed_hash,
    )
    output_hashes = write_preflight_outputs(result, output_dir)
    return {
        **result["audit"],
        "runtime_projection_path": str(runtime_path),
        "output_dir": str(output_dir),
        "output_sha256": output_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a non-executing experiment preflight for Validation40."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root", default=str(ROOT))
    args = parser.parse_args()
    summary = run(args.config, repo_root=args.repo_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
