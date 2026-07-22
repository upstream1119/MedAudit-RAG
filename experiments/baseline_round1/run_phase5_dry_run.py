"""Generate Phase 5 baseline-round1 dry-run artifacts.

This script does not call external LLM APIs. It prepares prompts, call plans,
cache keys, and token/cost estimate placeholders for a fixed 10-sample subset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = "experiments/baseline_round1/configs/phase5_dry_run_config.json"


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


def compact_text(value: str, limit: int) -> str:
    value = " ".join((value or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def estimate_tokens(text: str) -> int:
    """Conservative rough estimate for mixed Chinese/English prompt text."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 1.8))


def stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def index_by_sample(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["sample_id"]: row for row in rows if row.get("sample_id")}


def load_retrieval_outputs(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return index_by_sample(load_jsonl(path))


def evidence_context_from_retrieval(
    sample_id: str,
    retrieval_by_id: dict[str, dict[str, Any]],
    *,
    max_snippets: int,
    max_chars: int,
) -> tuple[list[dict[str, Any]], str]:
    payload = retrieval_by_id.get(sample_id)
    if not payload:
        return [], "missing_retrieval_output"

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
                "relevance_score": item.get("relevance_score", ""),
                "authority_weight": item.get("authority_weight", ""),
                "content": content,
            }
        )
        if len(snippets) >= max_snippets:
            break

    return snippets, "cached_retrieval_smoke_output"


def render_evidence_context(snippets: list[dict[str, Any]]) -> str:
    if not snippets:
        return "本轮未提供可用检索证据片段。"

    parts = []
    for item in snippets:
        parts.append(
            "\n".join(
                [
                    f"证据 {item['rank']}",
                    f"来源: {item['source_file']}",
                    f"页码: {item['page_number']}",
                    f"相关性: {item['relevance_score']}",
                    f"权威权重: {item['authority_weight']}",
                    f"内容: {item['content']}",
                ]
            )
        )
    return "\n\n".join(parts)


def render_prompt(method_id: str, sample: dict[str, Any], snippets: list[dict[str, Any]]) -> str:
    question = sample["question"]
    safety = (
        "你正在参与一个儿科用药证据审计研究原型。"
        "本系统仅用于科研、教学和审核演示，不提供真实诊断或个体化处方。"
    )

    if method_id == "vanilla_llm":
        return f"""{safety}

请回答用户问题。若无法确定，必须说明需要医生或药师人工复核。

用户问题：
{question}
"""

    evidence_context = render_evidence_context(snippets)
    if method_id == "naive_rag":
        return f"""{safety}

请基于下方检索证据回答用户问题。不要编造证据中没有的信息。

检索证据：
{evidence_context}

用户问题：
{question}
"""

    if method_id == "trust_gated_rag":
        return f"""{safety}

请严格执行证据边界：
1. 只基于下方证据回答。
2. 证据不足、证据与问题不一致、或用户请求越过处方边界时，必须拒答或提示人工复核。
3. 不得输出个体化处方、确定治疗方案或脱离证据的剂量/频次建议。
4. 回答必须简洁，并说明依据来源。

检索证据：
{evidence_context}

用户问题：
{question}
"""

    raise ValueError(f"Unsupported method_id: {method_id}")


def resolve_samples(dataset_rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = index_by_sample(dataset_rows)
    resolved: list[dict[str, Any]] = []
    for manifest in manifest_rows:
        sample_id = manifest["sample_id"]
        if sample_id not in by_id:
            raise ValueError(f"Manifest references unknown sample_id: {sample_id}")
        sample = by_id[sample_id]
        resolved.append(
            {
                **manifest,
                "question": sample.get("question"),
                "scenario_type": sample.get("scenario_type"),
                "risk_labels": sample.get("risk_labels", []),
                "expected_decision": sample.get("expected_decision"),
                "gold_evidence_status": sample.get("gold_evidence_status"),
                "current_kb_support": sample.get("current_kb_support"),
                "dataset_version": sample.get("dataset_version"),
                "kb_version": sample.get("kb_version"),
            }
        )
    return resolved


def build_records(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_rows = load_jsonl(repo_path(config["dataset_path"]))
    manifest_rows = load_jsonl(repo_path(config["sample_manifest_path"]))
    retrieval_by_id = load_retrieval_outputs(repo_path(config["retrieval_outputs_path"]))
    resolved_samples = resolve_samples(dataset_rows, manifest_rows)

    prompt_records: list[dict[str, Any]] = []
    call_records: list[dict[str, Any]] = []
    raw_output_records: list[dict[str, Any]] = []
    evaluation_metadata_records: list[dict[str, Any]] = []

    for sample in resolved_samples:
        snippets, context_source = evidence_context_from_retrieval(
            sample["sample_id"],
            retrieval_by_id,
            max_snippets=int(config["max_evidence_snippets"]),
            max_chars=int(config["max_evidence_chars_per_snippet"]),
        )
        for method in config["methods"]:
            method_id = method["method_id"]
            if method_id == "vanilla_llm":
                method_snippets: list[dict[str, Any]] = []
                method_context_source = "none_no_retrieval_context"
            else:
                method_snippets = snippets
                method_context_source = context_source

            for model in config["models"]:
                prompt = render_prompt(method_id, sample, method_snippets)
                key_payload = {
                    "sample_id": sample["sample_id"],
                    "method_id": method_id,
                    "model_provider": model["model_provider"],
                    "model_name": model["model_name"],
                    "prompt_version": config["prompt_version"],
                    "dataset_version": config["dataset_version"],
                    "kb_version": config["kb_version"],
                }
                cache_key = stable_hash(key_payload)
                estimated_input_tokens = estimate_tokens(prompt)
                estimated_output_tokens = int(config["max_output_tokens"])
                estimated_cost_cny = (
                    estimated_input_tokens * float(model["price_input_per_1m_cny"])
                    + estimated_output_tokens * float(model["price_output_per_1m_cny"])
                ) / 1_000_000

                model_common = {
                    **key_payload,
                    "cache_key": cache_key,
                    "run_mode": config["run_mode"],
                    "evidence_context_source": method_context_source,
                    "evidence_snippet_count": len(method_snippets),
                }
                evaluation_metadata_records.append(
                    {
                        **key_payload,
                        "cache_key": cache_key,
                        "run_mode": config["run_mode"],
                        "question": sample["question"],
                        "scenario_type": sample["scenario_type"],
                        "risk_labels": sample.get("risk_labels", []),
                        "expected_decision": sample["expected_decision"],
                        "gold_evidence_status": sample["gold_evidence_status"],
                        "current_kb_support": sample.get("current_kb_support"),
                        "evidence_context_source": method_context_source,
                        "evidence_snippet_count": len(method_snippets),
                    }
                )
                prompt_records.append(
                    {
                        **model_common,
                        "question": sample["question"],
                        "prompt": prompt,
                    }
                )
                call_records.append(
                    {
                        **model_common,
                        "estimated_input_tokens": estimated_input_tokens,
                        "estimated_output_tokens": estimated_output_tokens,
                        "estimated_cost_cny": round(estimated_cost_cny, 8),
                        "should_call_model": False,
                        "skip_reason": "dry_run_no_api_call",
                    }
                )
                raw_output_records.append(
                    {
                        **model_common,
                        "raw_output": "",
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "estimated_cost_cny": 0,
                        "status": "dry_run_not_executed",
                    }
                )

    return resolved_samples, prompt_records, call_records, raw_output_records, evaluation_metadata_records


def write_token_estimate_csv(path: Path, call_records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "method_id",
        "model_provider",
        "model_name",
        "prompt_version",
        "dataset_version",
        "kb_version",
        "cache_key",
        "estimated_input_tokens",
        "estimated_output_tokens",
        "estimated_cost_cny",
        "should_call_model",
        "skip_reason",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in call_records:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary(path: Path, config: dict[str, Any], samples: list[dict[str, Any]], calls: list[dict[str, Any]]) -> None:
    total_input = sum(int(row["estimated_input_tokens"]) for row in calls)
    total_output = sum(int(row["estimated_output_tokens"]) for row in calls)
    total_cost = sum(float(row["estimated_cost_cny"]) for row in calls)
    methods = sorted({row["method_id"] for row in calls})
    models = sorted({row["model_name"] for row in calls})

    lines = [
        "# Phase 5 Baseline Round 1 Dry Run",
        "",
        "## Run Boundary",
        "",
        "- No external LLM API was called.",
        "- No real model output was generated.",
        "- This run only validates sample selection, prompt rendering, cache keys, and cost-log structure.",
        "",
        "## Versions",
        "",
        f"- dataset_version: `{config['dataset_version']}`",
        f"- kb_version: `{config['kb_version']}`",
        f"- prompt_version: `{config['prompt_version']}`",
        f"- run_mode: `{config['run_mode']}`",
        "",
        "## Scope",
        "",
        f"- samples: `{len(samples)}`",
        f"- methods: `{', '.join(methods)}`",
        f"- models: `{', '.join(models)}`",
        f"- planned_calls: `{len(calls)}`",
        f"- max_evidence_snippets: `{config['max_evidence_snippets']}`",
        "",
        "## Token Estimate",
        "",
        f"- estimated_input_tokens_total: `{total_input}`",
        f"- estimated_output_tokens_total: `{total_output}`",
        f"- estimated_cost_cny_total: `{total_cost:.8f}`",
        "",
        "## Selected Samples",
        "",
    ]
    for sample in samples:
        lines.append(
            f"- `{sample['sample_id']}` | `{sample['expected_decision']}` | "
            f"`{sample['scenario_type']}` | {sample['question']}"
        )

    lines.extend(
        [
            "",
            "## Next Gate",
            "",
            "Before real execution, review `prompts.jsonl` and confirm the model list, price fields, and output directory policy.",
            "Real model execution should reuse the same cache key convention and write failed cases separately.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config_path = repo_path(args.config)
    config = load_json(config_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = repo_path(config["output_root"]) / f"phase5_dry_run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    config_effective = {
        **config,
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "output_dir": str(output_dir.relative_to(REPO_ROOT)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    samples, prompts, calls, raw_outputs, evaluation_metadata = build_records(config)

    write_json(output_dir / "run_config_effective.json", config_effective)
    write_jsonl(output_dir / "sample_manifest_resolved.jsonl", samples)
    write_jsonl(output_dir / "prompts.jsonl", prompts)
    write_jsonl(output_dir / "call_plan.jsonl", calls)
    write_jsonl(output_dir / "raw_model_outputs.jsonl", raw_outputs)
    write_jsonl(output_dir / "evaluation_metadata.jsonl", evaluation_metadata)
    write_token_estimate_csv(output_dir / "token_usage_estimate.csv", calls)
    write_summary(output_dir / "summary.md", config_effective, samples, calls)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir.relative_to(REPO_ROOT)),
                "sample_count": len(samples),
                "planned_calls": len(calls),
                "run_mode": config["run_mode"],
                "external_model_calls": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
