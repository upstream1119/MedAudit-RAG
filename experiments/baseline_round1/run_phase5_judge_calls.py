"""Run Phase 5 judge calls with cache and fail-soft logging.

Default behavior is non-executing. Add ``--execute`` only when you intentionally
want to call the configured judge model API.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


RUN_ROOT = "experiments/baseline_round1/runs"
JUDGE_DRY_RUN_PREFIX = "phase5_judge_dry_run_"
JUDGE_CALL_PREFIX = "phase5_judge_calls_"
DEFAULT_CACHE_DIR = "experiments/baseline_round1/.cache/judge_outputs"
DEFAULT_API_KEY_ENV = "DASHSCOPE_API_KEY"
DEFAULT_API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_MAX_RETRIES = 2
DEFAULT_TEMPERATURE = 0.0


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


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
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


def latest_judge_dry_run(run_root: Path) -> Path:
    candidates = [
        path
        for path in run_root.iterdir()
        if path.is_dir()
        and path.name.startswith(JUDGE_DRY_RUN_PREFIX)
        and (path / "judge_prompts.jsonl").exists()
        and (path / "judge_call_plan.jsonl").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No judge dry-run directory found under {run_root}")
    return sorted(candidates, key=lambda path: path.name)[-1]


def index_by_judge_cache_key(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["judge_cache_key"]: row for row in rows if row.get("judge_cache_key")}


def retry_cache_keys(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    rows = load_jsonl(path / "failed_judge_cases.jsonl")
    return {row["judge_cache_key"] for row in rows if row.get("judge_cache_key")}


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


def estimate_cost(input_tokens: int, output_tokens: int, model: dict[str, Any]) -> float:
    return (
        input_tokens * float(model.get("price_input_per_1m_cny", 0.0))
        + output_tokens * float(model.get("price_output_per_1m_cny", 0.0))
    ) / 1_000_000


def call_chat_completion(settings: dict[str, Any], model: dict[str, Any], prompt: str, api_key: str) -> dict[str, Any]:
    payload = {
        "model": model["model_name"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(settings["temperature"]),
        "max_tokens": int(settings["max_output_tokens"]),
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        str(settings["api_base_url"]),
        data=data,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=int(settings["request_timeout_seconds"])) as response:
        return json.loads(response.read().decode("utf-8"))


def cached_or_call(
    settings: dict[str, Any],
    model: dict[str, Any],
    prompt_record: dict[str, Any],
    call_record: dict[str, Any],
    cache_dir: Path,
    *,
    execute: bool,
    api_key: str | None,
) -> dict[str, Any]:
    judge_cache_key = prompt_record["judge_cache_key"]
    cache_path = cache_dir / f"{judge_cache_key}.json"
    base = {
        "sample_id": prompt_record.get("sample_id", ""),
        "method_a": call_record.get("method_a", ""),
        "method_b": call_record.get("method_b", ""),
        "judge_provider": prompt_record.get("judge_provider", ""),
        "judge_model": prompt_record.get("judge_model", ""),
        "judge_prompt_version": prompt_record.get("judge_prompt_version", ""),
        "dataset_version": prompt_record.get("dataset_version", ""),
        "kb_version": prompt_record.get("kb_version", ""),
        "judge_cache_key": judge_cache_key,
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
            "error_message": f"Missing API key env: {settings['api_key_env']}",
        }

    last_error = ""
    for attempt in range(int(settings["max_retries"]) + 1):
        try:
            raw_response = call_chat_completion(settings, model, prompt_record["prompt"], api_key)
            raw_output = extract_output_text(raw_response)
            input_tokens, output_tokens, total_tokens = extract_usage(raw_response)
            write_json(
                cache_path,
                {
                    "judge_cache_key": judge_cache_key,
                    "cached_at": datetime.now().isoformat(timespec="seconds"),
                    "raw_output": raw_output,
                    "raw_response": raw_response,
                },
            )
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
        except Exception as exc:  # noqa: BLE001 - experiment runner should fail-soft.
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < int(settings["max_retries"]):
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
        "method_a",
        "method_b",
        "judge_provider",
        "judge_model",
        "judge_prompt_version",
        "dataset_version",
        "kb_version",
        "judge_cache_key",
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


def write_summary(path: Path, settings: dict[str, Any], output_dir: Path, source_run_dir: Path, rows: list[dict[str, Any]]) -> None:
    status_counts = Counter(str(row.get("status")) for row in rows)
    total_input = sum(int(row.get("input_tokens") or 0) for row in rows)
    total_output = sum(int(row.get("output_tokens") or 0) for row in rows)
    total_cost = sum(float(row.get("estimated_cost_cny") or 0) for row in rows)
    lines = [
        "# Phase 5 Judge Calls",
        "",
        "## Boundary",
        "",
        f"- execute_judge_calls: `{settings['execute_judge_calls']}`",
        "- This runner preserves raw judge outputs and supports cache-first reruns.",
        "- Non-execution mode is for plumbing validation and does not spend tokens.",
        "",
        "## Paths",
        "",
        f"- source_judge_dry_run_dir: `{safe_relative(source_run_dir)}`",
        f"- output_dir: `{safe_relative(output_dir)}`",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in sorted(status_counts.items()):
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
            "Run `validate_judge_outputs.py` on `raw_judge_outputs.jsonl` after real judge calls. Failed cases should be retried with `--retry-failed-from`, not by rerunning the whole batch.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-dir", default="latest")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--no-env-file", action="store_true")
    parser.add_argument("--retry-failed-from", default=None)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--request-timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    args = parser.parse_args()

    run_root = repo_path(RUN_ROOT)
    source_run_dir = latest_judge_dry_run(run_root) if args.source_run_dir == "latest" else repo_path(args.source_run_dir)
    source_config = load_json(source_run_dir / "run_config_effective.json")
    model = source_config["models"][0]
    settings = {
        "execute_judge_calls": bool(args.execute),
        "api_key_env": args.api_key_env,
        "api_base_url": args.api_base_url,
        "request_timeout_seconds": args.request_timeout_seconds,
        "max_retries": args.max_retries,
        "temperature": DEFAULT_TEMPERATURE,
        "max_output_tokens": int(source_config.get("max_output_tokens", 350)),
    }

    if not args.no_env_file:
        load_env_file(REPO_ROOT / ".env")
        load_env_file(REPO_ROOT / "backend" / ".env")

    prompts = load_jsonl(source_run_dir / "judge_prompts.jsonl")
    call_plan_by_key = index_by_judge_cache_key(load_jsonl(source_run_dir / "judge_call_plan.jsonl"))
    retry_keys = retry_cache_keys(repo_path(args.retry_failed_from) if args.retry_failed_from else None)
    if retry_keys is not None:
        prompts = [row for row in prompts if row["judge_cache_key"] in retry_keys]
    if args.limit is not None:
        prompts = prompts[: args.limit]

    output_dir = create_unique_run_dir(run_root, f"{JUDGE_CALL_PREFIX}{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    cache_dir = repo_path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get(args.api_key_env)

    raw_outputs: list[dict[str, Any]] = []
    for prompt_record in prompts:
        call_record = call_plan_by_key[prompt_record["judge_cache_key"]]
        raw_outputs.append(
            cached_or_call(
                settings,
                model,
                prompt_record,
                call_record,
                cache_dir,
                execute=bool(args.execute),
                api_key=api_key,
            )
        )

    selected_keys = {row["judge_cache_key"] for row in prompts}
    selected_call_plan = [row for row in call_plan_by_key.values() if row.get("judge_cache_key") in selected_keys]
    failed_cases = [row for row in raw_outputs if row.get("status") == "failed"]
    effective_config = {
        **settings,
        "source_judge_dry_run_dir": safe_relative(source_run_dir),
        "output_dir": safe_relative(output_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "limit": args.limit,
        "retry_failed_from": args.retry_failed_from,
        "cache_dir": safe_relative(cache_dir),
        "load_env_files_effective": not args.no_env_file,
        "model": model,
    }

    write_json(output_dir / "run_config_effective.json", effective_config)
    write_jsonl(output_dir / "judge_prompts.jsonl", prompts)
    write_jsonl(output_dir / "judge_call_plan.jsonl", selected_call_plan)
    write_jsonl(output_dir / "raw_judge_outputs.jsonl", raw_outputs)
    write_jsonl(output_dir / "failed_judge_cases.jsonl", failed_cases)
    write_token_usage_csv(output_dir / "judge_token_usage_actual.csv", raw_outputs)
    write_summary(output_dir / "summary.md", effective_config, output_dir, source_run_dir, raw_outputs)

    print(
        json.dumps(
            {
                "output_dir": safe_relative(output_dir),
                "source_judge_dry_run_dir": safe_relative(source_run_dir),
                "selected_calls": len(prompts),
                "execute_judge_calls": bool(args.execute),
                "status_counts": dict(sorted(Counter(row["status"] for row in raw_outputs).items())),
                "failed_cases": len(failed_cases),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
