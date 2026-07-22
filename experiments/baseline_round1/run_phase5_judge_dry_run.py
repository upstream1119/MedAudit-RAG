"""Prepare Phase 5-D pairwise judge dry-run artifacts.

This script does not call external LLM APIs. It converts saved Phase 5-B pilot
outputs into blinded pairwise judge prompts, cost estimates, and metadata files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = "experiments/baseline_round1/configs/phase5_judge_dry_run_config.json"


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


def compact_text(value: str, limit: int) -> str:
    value = " ".join((value or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 1.8))


def stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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


def index_by_cache_key(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["cache_key"]: row for row in rows if row.get("cache_key")}


def index_retrieval_by_sample(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["sample_id"]: row for row in rows if row.get("sample_id")}


def evidence_snippets_for_sample(
    sample_id: str,
    retrieval_by_sample: dict[str, dict[str, Any]],
    *,
    max_snippets: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    payload = retrieval_by_sample.get(sample_id) or {}
    snippets: list[dict[str, Any]] = []
    for rank, item in enumerate(payload.get("results", []), start=1):
        content = compact_text(str(item.get("content") or ""), max_chars)
        if not content:
            continue
        snippets.append(
            {
                "rank": rank,
                "source_file": item.get("source_file", ""),
                "page_number": item.get("page_number", ""),
                "content": content,
            }
        )
        if len(snippets) >= max_snippets:
            break
    return snippets


def render_evidence_context(snippets: list[dict[str, Any]]) -> str:
    if not snippets:
        return "No evidence snippets were provided for this sample."
    parts = []
    for item in snippets:
        parts.append(
            "\n".join(
                [
                    f"Evidence {item['rank']}",
                    f"Source: {item['source_file']}",
                    f"Page: {item['page_number']}",
                    f"Content: {item['content']}",
                ]
            )
        )
    return "\n\n".join(parts)


def build_judge_prompt(
    template: str,
    *,
    question: str,
    evidence_context: str,
    answer_a: str,
    answer_b: str,
) -> str:
    return f"""{template}

Case:

Question:
{question}

Evidence snippets:
{evidence_context}

Answer A:
{answer_a}

Answer B:
{answer_b}
"""


def pair_records(
    config: dict[str, Any],
    template: str,
    raw_rows: list[dict[str, Any]],
    meta_by_key: dict[str, dict[str, Any]],
    retrieval_by_sample: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    usable_rows = [
        row
        for row in raw_rows
        if row.get("status") in {"success", "cache_hit"} and str(row.get("raw_output") or "").strip()
    ]
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usable_rows:
        by_sample[str(row["sample_id"])].append(row)

    skipped_samples: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    judge_model = config["models"][0]

    for sample_id in sorted(by_sample):
        sample_rows = sorted(by_sample[sample_id], key=lambda row: str(row.get("method_id", "")))
        if len(sample_rows) < 2:
            skipped_samples.append(
                {
                    "sample_id": sample_id,
                    "usable_outputs": len(sample_rows),
                    "skip_reason": "fewer_than_two_usable_outputs",
                }
            )
            continue

        meta = meta_by_key.get(sample_rows[0]["cache_key"], {})
        snippets = evidence_snippets_for_sample(
            sample_id,
            retrieval_by_sample,
            max_snippets=int(config["max_evidence_snippets"]),
            max_chars=int(config["max_evidence_chars_per_snippet"]),
        )
        evidence_context = render_evidence_context(snippets)

        for row_a, row_b in combinations(sample_rows, 2):
            key_payload = {
                "sample_id": sample_id,
                "method_a": row_a["method_id"],
                "method_b": row_b["method_id"],
                "answer_a_cache_key": row_a["cache_key"],
                "answer_b_cache_key": row_b["cache_key"],
                "judge_prompt_version": config["judge_prompt_version"],
                "dataset_version": config["dataset_version"],
                "kb_version": config["kb_version"],
                "judge_provider": judge_model["model_provider"],
                "judge_model": judge_model["model_name"],
            }
            judge_cache_key = stable_hash(key_payload)
            answer_a = compact_text(str(row_a.get("raw_output") or ""), int(config["max_answer_chars"]))
            answer_b = compact_text(str(row_b.get("raw_output") or ""), int(config["max_answer_chars"]))
            prompt = build_judge_prompt(
                template,
                question=str(meta.get("question") or ""),
                evidence_context=evidence_context,
                answer_a=answer_a,
                answer_b=answer_b,
            )
            estimated_input_tokens = estimate_tokens(prompt)
            estimated_output_tokens = int(config["max_output_tokens"])
            estimated_cost_cny = (
                estimated_input_tokens * float(judge_model.get("price_input_per_1m_cny", 0.0))
                + estimated_output_tokens * float(judge_model.get("price_output_per_1m_cny", 0.0))
            ) / 1_000_000
            pairs.append(
                {
                    **key_payload,
                    "judge_cache_key": judge_cache_key,
                    "run_mode": config["run_mode"],
                    "generator_prompt_version": row_a.get("prompt_version", ""),
                    "evidence_snippet_count": len(snippets),
                    "question": meta.get("question", ""),
                    "expected_decision": meta.get("expected_decision", ""),
                    "scenario_type": meta.get("scenario_type", ""),
                    "risk_labels": meta.get("risk_labels", []),
                    "gold_evidence_status": meta.get("gold_evidence_status", ""),
                    "current_kb_support": meta.get("current_kb_support", ""),
                    "answer_a_preview": answer_a,
                    "answer_b_preview": answer_b,
                    "prompt": prompt,
                    "estimated_input_tokens": estimated_input_tokens,
                    "estimated_output_tokens": estimated_output_tokens,
                    "estimated_cost_cny": round(estimated_cost_cny, 8),
                    "should_call_model": False,
                    "skip_reason": "judge_dry_run_no_api_call",
                }
            )
    return pairs, skipped_samples


def write_token_estimate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
        "estimated_input_tokens",
        "estimated_output_tokens",
        "estimated_cost_cny",
        "should_call_model",
        "skip_reason",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary(
    path: Path,
    config: dict[str, Any],
    output_dir: Path,
    source_pilot_run_dir: Path,
    pairs: list[dict[str, Any]],
    skipped_samples: list[dict[str, Any]],
) -> None:
    total_input = sum(int(row["estimated_input_tokens"]) for row in pairs)
    total_output = sum(int(row["estimated_output_tokens"]) for row in pairs)
    total_cost = sum(float(row["estimated_cost_cny"]) for row in pairs)
    by_sample: dict[str, int] = defaultdict(int)
    for row in pairs:
        by_sample[row["sample_id"]] += 1

    lines = [
        "# Phase 5-D Judge Dry Run",
        "",
        "## Boundary",
        "",
        "- No external judge API was called.",
        "- Method names are hidden inside judge prompts as Answer A / Answer B.",
        "- This prepares LLM-as-a-judge plumbing only; it is not final paper evidence.",
        "",
        "## Paths",
        "",
        f"- source_pilot_run_dir: `{safe_relative(source_pilot_run_dir)}`",
        f"- output_dir: `{safe_relative(output_dir)}`",
        "",
        "## Versions",
        "",
        f"- dataset_version: `{config['dataset_version']}`",
        f"- kb_version: `{config['kb_version']}`",
        f"- judge_prompt_version: `{config['judge_prompt_version']}`",
        "",
        "## Scope",
        "",
        f"- judge_pairs: `{len(pairs)}`",
        f"- skipped_samples: `{len(skipped_samples)}`",
        f"- samples_with_pairs: `{len(by_sample)}`",
        "",
        "## Token Estimate",
        "",
        f"- estimated_input_tokens_total: `{total_input}`",
        f"- estimated_output_tokens_total: `{total_output}`",
        f"- estimated_cost_cny_total: `{total_cost:.8f}`",
        "",
        "## Pair Counts by Sample",
        "",
    ]
    for sample_id, count in sorted(by_sample.items()):
        lines.append(f"- `{sample_id}`: `{count}`")
    if skipped_samples:
        lines.extend(["", "## Skipped Samples", ""])
        for item in skipped_samples:
            lines.append(
                f"- `{item['sample_id']}`: {item['skip_reason']} "
                f"(usable_outputs={item['usable_outputs']})"
            )
    lines.extend(
        [
            "",
            "## Next Gate",
            "",
            "Review `judge_prompts.jsonl` before any real judge call. If accepted, add a separate execute runner that reuses `judge_cache_key` and writes raw judge outputs.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--source-pilot-run-dir", default=None)
    parser.add_argument("--limit-pairs", type=int, default=None)
    args = parser.parse_args()

    config_path = repo_path(args.config)
    config = load_json(config_path)
    if args.source_pilot_run_dir:
        config["source_pilot_run_dir"] = args.source_pilot_run_dir

    source_pilot_run_dir = repo_path(config["source_pilot_run_dir"])
    raw_rows = load_jsonl(source_pilot_run_dir / "raw_model_outputs.jsonl")
    meta_by_key = index_by_cache_key(load_jsonl(source_pilot_run_dir / "evaluation_metadata.jsonl"))
    retrieval_by_sample = index_retrieval_by_sample(load_jsonl(repo_path(config["retrieval_outputs_path"])))
    template = repo_path(config["judge_prompt_path"]).read_text(encoding="utf-8-sig")

    pairs, skipped_samples = pair_records(config, template, raw_rows, meta_by_key, retrieval_by_sample)
    if args.limit_pairs is not None:
        pairs = pairs[: args.limit_pairs]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = create_unique_run_dir(repo_path(config["output_root"]), f"phase5_judge_dry_run_{timestamp}")

    effective_config = {
        **config,
        "config_path": safe_relative(config_path),
        "source_pilot_run_dir": safe_relative(source_pilot_run_dir),
        "output_dir": safe_relative(output_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "limit_pairs": args.limit_pairs,
    }
    judge_prompts = [
        {
            "sample_id": row["sample_id"],
            "method_a_hidden": "Answer A",
            "method_b_hidden": "Answer B",
            "judge_provider": row["judge_provider"],
            "judge_model": row["judge_model"],
            "judge_prompt_version": row["judge_prompt_version"],
            "dataset_version": row["dataset_version"],
            "kb_version": row["kb_version"],
            "judge_cache_key": row["judge_cache_key"],
            "prompt": row["prompt"],
        }
        for row in pairs
    ]
    judge_call_plan = [
        {
            key: row[key]
            for key in [
                "sample_id",
                "method_a",
                "method_b",
                "judge_provider",
                "judge_model",
                "judge_prompt_version",
                "dataset_version",
                "kb_version",
                "judge_cache_key",
                "estimated_input_tokens",
                "estimated_output_tokens",
                "estimated_cost_cny",
                "should_call_model",
                "skip_reason",
            ]
        }
        for row in pairs
    ]
    judge_metadata = [
        {
            key: row.get(key)
            for key in [
                "sample_id",
                "method_a",
                "method_b",
                "answer_a_cache_key",
                "answer_b_cache_key",
                "judge_cache_key",
                "generator_prompt_version",
                "judge_prompt_version",
                "dataset_version",
                "kb_version",
                "question",
                "expected_decision",
                "scenario_type",
                "risk_labels",
                "gold_evidence_status",
                "current_kb_support",
                "evidence_snippet_count",
            ]
        }
        for row in pairs
    ]
    raw_judge_outputs = [
        {
            "sample_id": row["sample_id"],
            "method_a": row["method_a"],
            "method_b": row["method_b"],
            "judge_cache_key": row["judge_cache_key"],
            "status": "dry_run_not_executed",
            "raw_output": "",
            "raw_response": {},
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_cny": 0,
        }
        for row in pairs
    ]

    write_json(output_dir / "run_config_effective.json", effective_config)
    write_jsonl(output_dir / "judge_prompts.jsonl", judge_prompts)
    write_jsonl(output_dir / "judge_call_plan.jsonl", judge_call_plan)
    write_jsonl(output_dir / "judge_pair_metadata.jsonl", judge_metadata)
    write_jsonl(output_dir / "raw_judge_outputs.jsonl", raw_judge_outputs)
    write_jsonl(output_dir / "skipped_samples.jsonl", skipped_samples)
    write_token_estimate_csv(output_dir / "judge_token_estimate.csv", pairs)
    write_summary(output_dir / "summary.md", effective_config, output_dir, source_pilot_run_dir, pairs, skipped_samples)

    print(
        json.dumps(
            {
                "output_dir": safe_relative(output_dir),
                "source_pilot_run_dir": safe_relative(source_pilot_run_dir),
                "judge_pairs": len(pairs),
                "skipped_samples": len(skipped_samples),
                "external_model_calls": 0,
                "estimated_input_tokens": sum(int(row["estimated_input_tokens"]) for row in pairs),
                "estimated_output_tokens": sum(int(row["estimated_output_tokens"]) for row in pairs),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
