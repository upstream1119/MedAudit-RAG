"""Run Phase 5 baseline pilot model calls with cache and cost logging.

The default mode is non-executing: it copies the latest dry-run prompts into a
pilot run directory and records what would be called. Add ``--execute`` only
when you intentionally want to call the configured model API.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = "experiments/baseline_round1/configs/phase5_pilot_config.json"


def find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "backend").exists() and (parent / "data").exists():
            return parent
    raise RuntimeError(f"Cannot locate repository root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def latest_run_dir(output_root: Path, prefix: str) -> Path:
    candidates = [
        path
        for path in output_root.iterdir()
        if path.is_dir() and path.name.startswith(prefix) and (path / "prompts.jsonl").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No source run directory found under {output_root} with prefix {prefix}")
    return sorted(candidates, key=lambda path: path.name)[-1]


def create_unique_run_dir(output_root: Path, name: str) -> Path:
    candidate = output_root / name
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    for index in range(1, 100):
        candidate = output_root / f"{name}_{index:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
    raise FileExistsError(f"Cannot create unique run directory for {output_root / name}")


def resolve_source_run_dir(config: dict[str, Any]) -> Path:
    source = str(config.get("source_run_dir", "latest"))
    if source == "latest":
        return latest_run_dir(repo_path(config["output_root"]), str(config["source_run_prefix"]))
    return repo_path(source)


def index_by_cache_key(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["cache_key"]: row for row in rows if row.get("cache_key")}


def retry_cache_keys(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    rows = load_jsonl(path / "failed_cases.jsonl")
    return {row["cache_key"] for row in rows if row.get("cache_key")}


def estimate_cost(input_tokens: int, output_tokens: int, model: dict[str, Any]) -> float:
    return (
        input_tokens * float(model.get("price_input_per_1m_cny", 0.0))
        + output_tokens * float(model.get("price_output_per_1m_cny", 0.0))
    ) / 1_000_000


def extract_output_text(response_json: dict[str, Any]) -> str:
    try:
        return str(response_json["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError):
        return ""


def extract_usage(response_json: dict[str, Any]) -> tuple[int, int, int]:
    usage = response_json.get("usage") or {}
    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
    return input_tokens, output_tokens, total_tokens


def call_chat_completion(config: dict[str, Any], model: dict[str, Any], prompt: str, api_key: str) -> dict[str, Any]:
    payload = {
        "model": model["model_name"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(config["temperature"]),
        "max_tokens": int(config["max_output_tokens"]),
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        str(config["api_base_url"]),
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=int(config["request_timeout_seconds"])) as response:
        return json.loads(response.read().decode("utf-8"))


def cached_or_call(
    config: dict[str, Any],
    model: dict[str, Any],
    prompt_record: dict[str, Any],
    call_record: dict[str, Any],
    cache_dir: Path,
    *,
    execute: bool,
    api_key: str | None,
) -> dict[str, Any]:
    cache_key = prompt_record["cache_key"]
    cache_path = cache_dir / f"{cache_key}.json"
    base = {
        key: prompt_record[key]
        for key in [
            "sample_id",
            "method_id",
            "model_provider",
            "model_name",
            "prompt_version",
            "dataset_version",
            "kb_version",
            "cache_key",
            "evidence_context_source",
            "evidence_snippet_count",
        ]
        if key in prompt_record
    }

    if cache_path.exists():
        cached = load_json(cache_path)
        input_tokens, output_tokens, total_tokens = extract_usage(cached["raw_response"])
        return {
            **base,
            "status": "cache_hit",
            "raw_output": cached.get("raw_output", ""),
            "raw_response": cached["raw_response"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_cny": round(estimate_cost(input_tokens, output_tokens, model), 8),
            "error_type": "",
            "error_message": "",
        }

    if not execute:
        return {
            **base,
            "status": "planned_not_executed",
            "raw_output": "",
            "raw_response": {},
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_input_tokens": call_record.get("estimated_input_tokens", 0),
            "estimated_output_tokens": call_record.get("estimated_output_tokens", 0),
            "estimated_cost_cny": call_record.get("estimated_cost_cny", 0),
            "error_type": "",
            "error_message": "",
        }

    if not api_key:
        return {
            **base,
            "status": "failed",
            "raw_output": "",
            "raw_response": {},
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_cny": 0,
            "error_type": "missing_api_key",
            "error_message": f"Missing API key env: {config['api_key_env']}",
        }

    last_error = ""
    for attempt in range(int(config["max_retries"]) + 1):
        try:
            raw_response = call_chat_completion(config, model, prompt_record["prompt"], api_key)
            raw_output = extract_output_text(raw_response)
            input_tokens, output_tokens, total_tokens = extract_usage(raw_response)
            payload = {
                "cache_key": cache_key,
                "cached_at": datetime.now().isoformat(timespec="seconds"),
                "raw_output": raw_output,
                "raw_response": raw_response,
            }
            write_json(cache_path, payload)
            return {
                **base,
                "status": "success",
                "raw_output": raw_output,
                "raw_response": raw_response,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_cny": round(estimate_cost(input_tokens, output_tokens, model), 8),
                "error_type": "",
                "error_message": "",
            }
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {body[:500]}"
        except Exception as exc:  # noqa: BLE001 - keep experiment runner fail-soft.
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < int(config["max_retries"]):
            time.sleep(1.5 * (attempt + 1))

    return {
        **base,
        "status": "failed",
        "raw_output": "",
        "raw_response": {},
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_cny": 0,
        "error_type": "api_call_failed",
        "error_message": last_error,
    }


def write_token_usage_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "method_id",
        "model_provider",
        "model_name",
        "prompt_version",
        "dataset_version",
        "kb_version",
        "cache_key",
        "status",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "estimated_cost_cny",
        "error_type",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary(path: Path, config: dict[str, Any], output_dir: Path, source_run_dir: Path, rows: list[dict[str, Any]]) -> None:
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    total_input = sum(int(row.get("input_tokens") or 0) for row in rows)
    total_output = sum(int(row.get("output_tokens") or 0) for row in rows)
    total_cost = sum(float(row.get("estimated_cost_cny") or 0) for row in rows)
    lines = [
        "# Phase 5 Baseline Round 1 Pilot Calls",
        "",
        "## Boundary",
        "",
        f"- execute_model_calls: `{config['execute_model_calls']}`",
        "- This is still a pilot run, not final paper evidence.",
        "- Gold evaluation labels are kept in `evaluation_metadata.jsonl` and are not sent to the model.",
        "",
        "## Paths",
        "",
        f"- source_run_dir: `{source_run_dir.relative_to(REPO_ROOT)}`",
        f"- output_dir: `{output_dir.relative_to(REPO_ROOT)}`",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in sorted(statuses.items()):
        lines.append(f"- `{status}`: `{count}`")
    lines.extend(
        [
            "",
            "## Token Usage",
            "",
            f"- input_tokens_total: `{total_input}`",
            f"- output_tokens_total: `{total_output}`",
            f"- estimated_cost_cny_total: `{total_cost:.8f}`",
            "",
            "## Next Gate",
            "",
            "Review failed cases and raw outputs before expanding beyond this pilot.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--source-run-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--no-env-file", action="store_true")
    parser.add_argument("--retry-failed-from", default=None)
    args = parser.parse_args()

    config_path = repo_path(args.config)
    config = load_json(config_path)
    if args.source_run_dir:
        config["source_run_dir"] = args.source_run_dir
    if args.execute:
        config["execute_model_calls"] = True
    if not args.no_env_file and bool(config.get("load_env_files", True)):
        load_env_file(REPO_ROOT / ".env")
        load_env_file(REPO_ROOT / "backend" / ".env")

    source_run_dir = resolve_source_run_dir(config)
    prompts = load_jsonl(source_run_dir / "prompts.jsonl")
    call_plan_by_key = index_by_cache_key(load_jsonl(source_run_dir / "call_plan.jsonl"))
    evaluation_metadata_by_key = index_by_cache_key(load_jsonl(source_run_dir / "evaluation_metadata.jsonl"))

    retry_keys = retry_cache_keys(repo_path(args.retry_failed_from) if args.retry_failed_from else None)
    if retry_keys is not None:
        prompts = [row for row in prompts if row["cache_key"] in retry_keys]
    if args.limit is not None:
        prompts = prompts[: args.limit]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = repo_path(config["output_root"])
    output_dir = create_unique_run_dir(output_root, f"phase5_pilot_{timestamp}")
    cache_dir = repo_path(config["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    model = config["models"][0]
    api_key = os.environ.get(str(config["api_key_env"]))
    execute = bool(config.get("execute_model_calls", False))

    raw_outputs: list[dict[str, Any]] = []
    for prompt_record in prompts:
        call_record = call_plan_by_key[prompt_record["cache_key"]]
        raw_outputs.append(
            cached_or_call(
                config,
                model,
                prompt_record,
                call_record,
                cache_dir,
                execute=execute,
                api_key=api_key,
            )
        )

    selected_keys = {row["cache_key"] for row in prompts}
    selected_metadata = [row for key, row in evaluation_metadata_by_key.items() if key in selected_keys]
    failed_cases = [row for row in raw_outputs if row["status"] == "failed"]

    effective_config = {
        **config,
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "source_run_dir": str(source_run_dir.relative_to(REPO_ROOT)),
        "output_dir": str(output_dir.relative_to(REPO_ROOT)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "limit": args.limit,
        "retry_failed_from": args.retry_failed_from,
        "load_env_files_effective": not args.no_env_file and bool(config.get("load_env_files", True)),
    }
    write_json(output_dir / "run_config_effective.json", effective_config)
    write_jsonl(output_dir / "prompts.jsonl", prompts)
    write_jsonl(output_dir / "model_call_plan.jsonl", [call_plan_by_key[row["cache_key"]] for row in prompts])
    write_jsonl(output_dir / "raw_model_outputs.jsonl", raw_outputs)
    write_jsonl(output_dir / "evaluation_metadata.jsonl", selected_metadata)
    write_jsonl(output_dir / "failed_cases.jsonl", failed_cases)
    write_token_usage_csv(output_dir / "token_usage_actual.csv", raw_outputs)
    write_summary(output_dir / "summary.md", effective_config, output_dir, source_run_dir, raw_outputs)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir.relative_to(REPO_ROOT)),
                "source_run_dir": str(source_run_dir.relative_to(REPO_ROOT)),
                "selected_calls": len(prompts),
                "execute_model_calls": execute,
                "status_counts": {status: sum(1 for row in raw_outputs if row["status"] == status) for status in sorted({row["status"] for row in raw_outputs})},
                "failed_cases": len(failed_cases),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
