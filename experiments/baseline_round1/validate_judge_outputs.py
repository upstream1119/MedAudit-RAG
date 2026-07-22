"""Validate Phase 5 judge JSON outputs.

This utility is intentionally local-only. It can validate saved judge outputs or
run a built-in self-test before real judge API calls are enabled.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_SCORE_FIELDS = ["evidence_support", "safety_boundary", "refusal_correctness"]
REQUIRED_BOOL_FIELDS = ["unsupported_claim", "unsafe_suggestion", "under_refusal"]
VALID_WINNERS = {"A", "B", "tie"}
DRY_RUN_STATUSES = {"dry_run_not_executed", "planned_not_executed"}


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            row.setdefault("_line_number", line_number)
            rows.append(row)
    return rows


def strip_code_fence(text: str) -> str:
    value = text.strip()
    if not value.startswith("```"):
        return value
    lines = value.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return value


def extract_output_text(row: dict[str, Any]) -> str:
    for key in ["raw_output", "judge_output", "output", "content"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raw_response = row.get("raw_response")
    if isinstance(raw_response, dict):
        try:
            content = raw_response["choices"][0]["message"]["content"]
            if isinstance(content, str):
                return content
        except (KeyError, IndexError, TypeError):
            return ""
    return ""


def parse_judge_json(text: str) -> dict[str, Any]:
    value = strip_code_fence(text)
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("judge output is not a JSON object")
    return parsed


def validate_answer_block(payload: dict[str, Any], answer_key: str) -> list[str]:
    errors: list[str] = []
    block = payload.get(answer_key)
    if not isinstance(block, dict):
        return [f"{answer_key}_missing_or_not_object"]

    for field in REQUIRED_SCORE_FIELDS:
        value = block.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1 or value > 3:
            errors.append(f"{answer_key}.{field}_not_int_1_to_3")

    for field in REQUIRED_BOOL_FIELDS:
        if not isinstance(block.get(field), bool):
            errors.append(f"{answer_key}.{field}_not_bool")
    return errors


def validate_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("winner") not in VALID_WINNERS:
        errors.append("winner_invalid")
    errors.extend(validate_answer_block(payload, "answer_a"))
    errors.extend(validate_answer_block(payload, "answer_b"))
    if not isinstance(payload.get("rationale"), str) or not payload.get("rationale", "").strip():
        errors.append("rationale_missing_or_not_string")
    return errors


def validate_rows(rows: list[dict[str, Any]], *, allow_dry_run_empty: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checked_rows: list[dict[str, Any]] = []
    errors_by_type: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()

    for index, row in enumerate(rows, start=1):
        row_id = row.get("judge_cache_key") or row.get("cache_key") or f"row_{index}"
        status = str(row.get("status") or "unknown")
        output_text = extract_output_text(row)
        result = {
            "row_index": index,
            "line_number": row.get("_line_number", ""),
            "row_id": row_id,
            "status": status,
            "valid": False,
            "skipped": False,
            "errors": [],
        }

        if not output_text.strip() and allow_dry_run_empty and status in DRY_RUN_STATUSES:
            result["skipped"] = True
            result["errors"] = ["dry_run_empty_output_skipped"]
            checked_rows.append(result)
            status_counts["skipped"] += 1
            continue

        if not output_text.strip():
            result["errors"] = ["empty_output"]
        else:
            try:
                payload = parse_judge_json(output_text)
                result["errors"] = validate_payload(payload)
                result["winner"] = payload.get("winner", "")
            except json.JSONDecodeError as exc:
                result["errors"] = [f"json_decode_error:{exc.msg}"]
            except Exception as exc:  # noqa: BLE001 - validator should report all malformed rows.
                result["errors"] = [f"parse_error:{type(exc).__name__}"]

        result["valid"] = not result["errors"]
        if result["valid"]:
            status_counts["valid"] += 1
        else:
            status_counts["invalid"] += 1
            for error in result["errors"]:
                errors_by_type[error] += 1
        checked_rows.append(result)

    summary = {
        "total_rows": len(rows),
        "valid_rows": status_counts["valid"],
        "invalid_rows": status_counts["invalid"],
        "skipped_rows": status_counts["skipped"],
        "errors_by_type": dict(sorted(errors_by_type.items())),
    }
    return checked_rows, summary


def self_test_rows() -> list[dict[str, Any]]:
    valid_payload = {
        "winner": "A",
        "answer_a": {
            "evidence_support": 3,
            "safety_boundary": 3,
            "refusal_correctness": 3,
            "unsupported_claim": False,
            "unsafe_suggestion": False,
            "under_refusal": False,
        },
        "answer_b": {
            "evidence_support": 1,
            "safety_boundary": 2,
            "refusal_correctness": 1,
            "unsupported_claim": True,
            "unsafe_suggestion": False,
            "under_refusal": True,
        },
        "rationale": "A is better supported by the provided evidence.",
    }
    invalid_payload = {
        "winner": "C",
        "answer_a": {"evidence_support": 4},
        "answer_b": "bad",
        "rationale": "",
    }
    return [
        {"judge_cache_key": "valid_json", "status": "success", "raw_output": json.dumps(valid_payload)},
        {"judge_cache_key": "fenced_json", "status": "success", "raw_output": "```json\n" + json.dumps(valid_payload) + "\n```"},
        {"judge_cache_key": "invalid_schema", "status": "success", "raw_output": json.dumps(invalid_payload)},
        {"judge_cache_key": "dry_run", "status": "dry_run_not_executed", "raw_output": ""},
    ]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="Path to raw judge outputs JSONL.")
    parser.add_argument("--output-summary", default=None, help="Optional JSON summary output path.")
    parser.add_argument("--self-test", action="store_true", help="Validate built-in examples without reading files.")
    parser.add_argument("--strict", action="store_true", help="Treat empty dry-run rows as invalid.")
    args = parser.parse_args()

    if args.self_test:
        rows = self_test_rows()
    elif args.input:
        rows = load_jsonl(repo_path(args.input))
    else:
        raise SystemExit("Provide --input or --self-test")

    checked_rows, summary = validate_rows(rows, allow_dry_run_empty=not args.strict)
    if args.output_summary:
        write_json(repo_path(args.output_summary), {"summary": summary, "rows": checked_rows})

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.self_test:
        expected = {"valid_rows": 2, "invalid_rows": 1, "skipped_rows": 1}
        for key, value in expected.items():
            if summary.get(key) != value:
                raise SystemExit(f"Self-test failed: expected {key}={value}, got {summary.get(key)}")


if __name__ == "__main__":
    main()
