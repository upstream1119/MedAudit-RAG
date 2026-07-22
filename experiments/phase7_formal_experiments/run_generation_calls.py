"""Execute Phase 7 paired generation calls with cache and budget guards.

The default path is non-executing. Real API requests require both ``--execute``
and ``--confirm-external-call`` so an accidental command cannot consume quota.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from experiments.baseline_round1 import run_phase5_model_calls as phase5_runtime


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    "experiments/phase7_formal_experiments/configs/"
    "phase7_generation_smoke3_v0_1.json"
)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_bundle(source_run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    prompts = phase5_runtime.load_jsonl(source_run_dir / "prompts.jsonl")
    call_plan = phase5_runtime.load_jsonl(source_run_dir / "call_plan.jsonl")
    evaluation_metadata = phase5_runtime.load_jsonl(
        source_run_dir / "evaluation_metadata.jsonl"
    )

    def unique_keys(rows: list[dict[str, Any]], artifact_name: str) -> set[str]:
        keys = [str(row.get("cache_key") or "") for row in rows]
        if any(not key for key in keys):
            raise ValueError(f"missing cache_key in {artifact_name}")
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate cache_key in {artifact_name}")
        return set(keys)

    prompt_keys = unique_keys(prompts, "prompts.jsonl")
    call_keys = unique_keys(call_plan, "call_plan.jsonl")
    metadata_keys = unique_keys(evaluation_metadata, "evaluation_metadata.jsonl")
    if prompt_keys != call_keys or prompt_keys != metadata_keys:
        raise ValueError(
            "cache_key sets differ across prompts, call plan, and evaluation metadata"
        )

    return (
        prompts,
        phase5_runtime.index_by_cache_key(call_plan),
        phase5_runtime.index_by_cache_key(evaluation_metadata),
    )


def _model_index(config: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    models = config.get("models") or []
    if not models:
        raise ValueError("models must contain at least one model")
    return {
        (str(model["model_provider"]), str(model["model_name"])): model
        for model in models
    }


def _select_prompts(
    prompts: list[dict[str, Any]],
    *,
    retry_failed_from: Path | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    if retry_failed_from is not None:
        retry_keys = phase5_runtime.retry_cache_keys(retry_failed_from)
        prompts = [row for row in prompts if row["cache_key"] in (retry_keys or set())]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be greater than zero")
        prompts = prompts[:limit]
    return prompts


def _validate_budget(
    config: dict[str, Any],
    prompts: list[dict[str, Any]],
    call_plan_by_key: dict[str, dict[str, Any]],
) -> None:
    estimated_input = sum(
        int(call_plan_by_key[row["cache_key"]].get("estimated_input_tokens") or 0)
        for row in prompts
    )
    estimated_output = sum(
        int(call_plan_by_key[row["cache_key"]].get("estimated_output_tokens") or 0)
        for row in prompts
    )
    limits = {
        "planned calls": (len(prompts), int(config["max_planned_calls"])),
        "estimated input tokens": (
            estimated_input,
            int(config["max_estimated_input_tokens"]),
        ),
        "estimated output tokens": (
            estimated_output,
            int(config["max_estimated_output_tokens"]),
        ),
    }
    exceeded = [
        f"{name}: {actual} > {maximum}"
        for name, (actual, maximum) in limits.items()
        if actual > maximum
    ]
    if exceeded:
        raise ValueError("experiment budget exceeded: " + "; ".join(exceeded))


def _write_summary(
    path: Path,
    *,
    config: dict[str, Any],
    source_run_dir: Path,
    rows: list[dict[str, Any]],
    external_model_calls: int,
) -> None:
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    lines = [
        "# Phase 7 Paired Generation Smoke",
        "",
        "## Boundary",
        "",
        f"- execute_model_calls: `{config['execute_model_calls']}`",
        f"- external_model_calls: `{external_model_calls}`",
        "- These are Dev50 development samples, not final Benchmark-v1 results.",
        "- No Graph-enhanced effectiveness claim may be made from this smoke run.",
        "",
        "## Source",
        "",
        f"- source_run_dir: `{source_run_dir}`",
        "",
        "## Status Counts",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in sorted(statuses.items()))
    lines.extend(
        [
            "",
            "## Token Usage",
            "",
            f"- input_tokens_total: `{sum(int(row.get('input_tokens') or 0) for row in rows)}`",
            f"- output_tokens_total: `{sum(int(row.get('output_tokens') or 0) for row in rows)}`",
            f"- estimated_cost_cny_total: `{sum(float(row.get('estimated_cost_cny') or 0) for row in rows):.8f}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_generation_calls(
    config: dict[str, Any],
    *,
    output_dir: Path | None = None,
    execute: bool | None = None,
    confirm_external_call: bool = False,
    retry_failed_from: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run or plan matched generation calls and persist an auditable record."""

    source_run_dir = _resolve_path(config["source_run_dir"])
    prompts, call_plan_by_key, metadata_by_key = _load_bundle(source_run_dir)
    prompts = _select_prompts(
        prompts,
        retry_failed_from=retry_failed_from,
        limit=limit,
    )
    _validate_budget(config, prompts, call_plan_by_key)

    execute_effective = (
        bool(config.get("execute_model_calls", False)) if execute is None else execute
    )
    cache_dir = _resolve_path(config["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing_cache_keys = [
        row["cache_key"]
        for row in prompts
        if not (cache_dir / f"{row['cache_key']}.json").exists()
    ]
    if execute_effective and missing_cache_keys and not confirm_external_call:
        raise ValueError(
            "external model calls require explicit confirm_external_call=True"
        )

    if bool(config.get("load_env_files", False)):
        phase5_runtime.load_env_file(REPO_ROOT / ".env")
        phase5_runtime.load_env_file(REPO_ROOT / "backend" / ".env")
    api_key = os.environ.get(str(config["api_key_env"]))
    models = _model_index(config)

    if output_dir is None:
        output_root = _resolve_path(config["output_root"])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = phase5_runtime.create_unique_run_dir(
            output_root,
            f"{config['run_id_prefix']}_{timestamp}",
        )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)

    raw_outputs: list[dict[str, Any]] = []
    for prompt_record in prompts:
        model_key = (
            str(prompt_record["model_provider"]),
            str(prompt_record["model_name"]),
        )
        if model_key not in models:
            raise ValueError(f"model missing from config: {model_key}")
        raw_outputs.append(
            phase5_runtime.cached_or_call(
                config,
                models[model_key],
                prompt_record,
                call_plan_by_key[prompt_record["cache_key"]],
                cache_dir,
                execute=execute_effective,
                api_key=api_key,
            )
        )

    selected_metadata = [metadata_by_key[row["cache_key"]] for row in prompts]
    failed_cases = [row for row in raw_outputs if row["status"] == "failed"]
    external_model_calls = sum(
        1 for row in raw_outputs if row["status"] == "success"
    )
    effective_config = {
        **config,
        "execute_model_calls": execute_effective,
        "source_run_dir": str(source_run_dir),
        "output_dir": str(output_dir),
        "retry_failed_from": str(retry_failed_from or ""),
        "limit": limit,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    phase5_runtime.write_json(output_dir / "run_config_effective.json", effective_config)
    phase5_runtime.write_jsonl(output_dir / "prompts.jsonl", prompts)
    phase5_runtime.write_jsonl(
        output_dir / "model_call_plan.jsonl",
        [call_plan_by_key[row["cache_key"]] for row in prompts],
    )
    phase5_runtime.write_jsonl(
        output_dir / "raw_model_outputs.jsonl", raw_outputs
    )
    phase5_runtime.write_jsonl(
        output_dir / "evaluation_metadata.jsonl", selected_metadata
    )
    phase5_runtime.write_jsonl(output_dir / "failed_cases.jsonl", failed_cases)
    phase5_runtime.write_token_usage_csv(
        output_dir / "token_usage_actual.csv", raw_outputs
    )
    _write_summary(
        output_dir / "summary.md",
        config=effective_config,
        source_run_dir=source_run_dir,
        rows=raw_outputs,
        external_model_calls=external_model_calls,
    )

    statuses: dict[str, int] = {}
    for row in raw_outputs:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    return {
        "output_dir": str(output_dir),
        "source_run_dir": str(source_run_dir),
        "selected_calls": len(prompts),
        "execute_model_calls": execute_effective,
        "external_model_calls": external_model_calls,
        "status_counts": statuses,
        "failed_cases": len(failed_cases),
        "input_tokens_total": sum(
            int(row.get("input_tokens") or 0) for row in raw_outputs
        ),
        "output_tokens_total": sum(
            int(row.get("output_tokens") or 0) for row in raw_outputs
        ),
        "estimated_cost_cny_total": round(
            sum(float(row.get("estimated_cost_cny") or 0) for row in raw_outputs),
            8,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-external-call", action="store_true")
    parser.add_argument("--retry-failed-from", default=None)
    args = parser.parse_args()

    config = phase5_runtime.load_json(_resolve_path(args.config))
    summary = run_generation_calls(
        config,
        execute=True if args.execute else None,
        confirm_external_call=args.confirm_external_call,
        retry_failed_from=(
            _resolve_path(args.retry_failed_from)
            if args.retry_failed_from
            else None
        ),
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
