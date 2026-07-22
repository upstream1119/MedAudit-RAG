"""Build Phase 6-B generation-side contrast dry-run artifacts.

This runner does not call external model APIs. It prepares vector-only and
graph-enhanced prompts from the same frozen reranking artifacts, so later model
calls can reuse the existing Phase 5 cache/cost discipline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .graph_contract import assert_no_gold_only_content


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    "experiments/phase6_evidence_graph/configs/"
    "phase6b_generation_contrast_smoke3_v0_1.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
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


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _compact_text(value: str, limit: int) -> str:
    compact = " ".join((value or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 1.8))


def _stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sample_ids(config: dict[str, Any]) -> list[str]:
    raw_ids = config.get("sample_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError("sample_ids must be a non-empty list")
    sample_ids: list[str] = []
    seen: set[str] = set()
    for raw_id in raw_ids:
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise ValueError("sample_ids must contain non-empty strings")
        sample_id = raw_id.strip()
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        sample_ids.append(sample_id)
    return sample_ids


def _load_question_index(config: dict[str, Any]) -> dict[str, str]:
    dataset_path = config.get("dataset_path")
    if not dataset_path:
        return {}
    rows = _read_jsonl(_resolve_path(dataset_path))
    index: dict[str, str] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "").strip()
        question = str(row.get("question") or "").strip()
        if sample_id and question:
            index[sample_id] = question
    return index


def _methods(config: dict[str, Any]) -> list[dict[str, Any]]:
    raw_methods = config.get("methods")
    if not isinstance(raw_methods, list) or not raw_methods:
        raise ValueError("methods must be a non-empty list")
    methods: list[dict[str, Any]] = []
    seen: set[str] = set()
    for method in raw_methods:
        if not isinstance(method, dict):
            raise ValueError("methods must contain objects")
        method_id = str(method.get("method_id") or "").strip()
        if method_id not in {"vector_only_rag", "graph_enhanced_rag"}:
            raise ValueError(f"unsupported method_id: {method_id}")
        if method_id in seen:
            raise ValueError(f"duplicate method_id: {method_id}")
        seen.add(method_id)
        methods.append(method)
    return methods


def _models(config: dict[str, Any]) -> list[dict[str, Any]]:
    raw_models = config.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("models must be a non-empty list")
    return raw_models


def _question_text(artifact: dict[str, Any]) -> str:
    parent = artifact.get("parent_question") or {}
    text = parent.get("question") or parent.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    for row in artifact.get("ranked_evidence", []):
        question = row.get("question")
        if isinstance(question, str) and question.strip():
            return question.strip()
    raise ValueError(f"missing question text for {artifact.get('sample_id')}")


def _evidence_rows(
    artifact: dict[str, Any],
    *,
    method_id: str,
    max_snippets: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    rows = list(artifact.get("ranked_evidence") or [])
    if method_id == "vector_only_rag":
        rows = sorted(rows, key=lambda row: (row.get("rank_before", 9999), row.get("evidence_id", "")))
    elif method_id == "graph_enhanced_rag":
        rows = sorted(rows, key=lambda row: (row.get("rank_after", 9999), row.get("rank_before", 9999), row.get("evidence_id", "")))
    else:
        raise ValueError(f"unsupported method_id: {method_id}")

    snippets: list[dict[str, Any]] = []
    for index, row in enumerate(rows[:max_snippets], start=1):
        content = _compact_text(str(row.get("content") or ""), max_chars)
        if not content:
            continue
        snippets.append(
            {
                "rank": index,
                "evidence_id": row.get("evidence_id", ""),
                "source_file": row.get("source_file", ""),
                "page_number": row.get("page_number", ""),
                "rank_before": row.get("rank_before", ""),
                "rank_after": row.get("rank_after", ""),
                "relevance_score": row.get("relevance_score", ""),
                "authority_weight": row.get("authority_weight", ""),
                "constraint_type_coverage": row.get("constraint_type_coverage", ""),
                "content": content,
            }
        )
    return snippets


def _render_evidence(snippets: list[dict[str, Any]]) -> str:
    if not snippets:
        return "本轮没有可用证据片段。"
    blocks = []
    for item in snippets:
        blocks.append(
            "\n".join(
                [
                    f"证据 {item['rank']}",
                    f"来源: {item['source_file']}",
                    f"页码: {item['page_number']}",
                    f"证据ID: {item['evidence_id']}",
                    f"原始排序: {item['rank_before']}",
                    f"图增强排序: {item['rank_after']}",
                    f"内容: {item['content']}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _render_constraint_audit(artifact: dict[str, Any], method_id: str) -> str:
    if method_id != "graph_enhanced_rag":
        return "本方法不提供运行时约束审计摘要。"
    rows = artifact.get("constraint_audit") or []
    if not rows:
        return "未抽取到可用于生成提示的运行时约束审计。"
    return "\n".join(
        f"- {row.get('constraint_type')}: 问题值={row.get('query_value')}; "
        f"证据值={','.join(str(v) for v in row.get('evidence_values', []))}; "
        f"状态={row.get('status')}"
        for row in rows
    )


def _render_prompt(
    *,
    method_id: str,
    question: str,
    artifact: dict[str, Any],
    snippets: list[dict[str, Any]],
) -> str:
    boundary = (
        "你正在参与一个儿科用药证据审计研究原型。本系统仅用于科研、教学和方法验证，"
        "不提供真实临床诊断或个体化处方。所有医学结论必须追溯到下方证据；证据不足、"
        "证据冲突或请求越过处方边界时，必须拒答或提示人工复核。"
    )
    evidence_context = _render_evidence(snippets)
    constraint_audit = _render_constraint_audit(artifact, method_id)
    method_instruction = (
        "请只基于原始向量检索证据回答。"
        if method_id == "vector_only_rag"
        else "请结合图增强排序后的证据和运行时约束审计回答。"
    )
    return f"""{boundary}

方法约束：
{method_instruction}

运行时约束审计：
{constraint_audit}

检索证据：
{evidence_context}

用户问题：
{question}

输出要求：
1. 回答必须简洁。
2. 若可回答，请说明依据来源。
3. 若证据不足或存在冲突，请明确提示人工复核。
"""


def _price(model: dict[str, Any], input_tokens: int, output_tokens: int) -> float:
    input_price = float(model.get("price_input_per_1m_cny", 0.0))
    output_price = float(model.get("price_output_per_1m_cny", 0.0))
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def build_records(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    input_run_dir = _resolve_path(config["input_run_dir"])
    artifact_dir = input_run_dir / "method_artifacts"
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"method artifact directory not found: {artifact_dir}")

    prompts: list[dict[str, Any]] = []
    call_plan: list[dict[str, Any]] = []
    raw_outputs: list[dict[str, Any]] = []
    evaluation_metadata: list[dict[str, Any]] = []
    question_index = _load_question_index(config)

    max_snippets = int(config["max_evidence_snippets"])
    max_chars = int(config["max_evidence_chars_per_snippet"])
    max_output_tokens = int(config["max_output_tokens"])

    for sample_id in _sample_ids(config):
        artifact = _read_json(artifact_dir / f"{sample_id}.json")
        assert_no_gold_only_content(artifact)
        question = question_index.get(sample_id) or _question_text(artifact)
        for method in _methods(config):
            method_id = method["method_id"]
            snippets = _evidence_rows(
                artifact,
                method_id=method_id,
                max_snippets=max_snippets,
                max_chars=max_chars,
            )
            prompt = _render_prompt(
                method_id=method_id,
                question=question,
                artifact=artifact,
                snippets=snippets,
            )
            assert_no_gold_only_content({"prompt": prompt})
            for model in _models(config):
                key_payload = {
                    "sample_id": sample_id,
                    "method_id": method_id,
                    "model_provider": model["model_provider"],
                    "model_name": model["model_name"],
                    "prompt_version": config["prompt_version"],
                    "dataset_version": config["dataset_version"],
                    "kb_version": config["kb_version"],
                }
                cache_key = _stable_hash(key_payload)
                input_tokens = _estimate_tokens(prompt)
                cost = _price(model, input_tokens, max_output_tokens)
                common = {
                    **key_payload,
                    "cache_key": cache_key,
                    "run_mode": config["run_mode"],
                    "evidence_context_source": method_id,
                    "evidence_snippet_count": len(snippets),
                }
                prompts.append({**common, "question": question, "prompt": prompt})
                call_plan.append(
                    {
                        **common,
                        "estimated_input_tokens": input_tokens,
                        "estimated_output_tokens": max_output_tokens,
                        "estimated_cost_cny": round(cost, 8),
                        "should_call_model": bool(config.get("execute_model_calls", False)),
                        "skip_reason": (
                            "" if config.get("execute_model_calls", False) else "dry_run_no_api_call"
                        ),
                    }
                )
                raw_outputs.append(
                    {
                        **common,
                        "raw_output": "",
                        "raw_response": {},
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "estimated_cost_cny": 0,
                        "status": "dry_run_not_executed",
                    }
                )
                evaluation_metadata.append(
                    {
                        **common,
                        "question": question,
                        "parent_reranking_artifact": str((artifact_dir / f"{sample_id}.json").relative_to(PROJECT_ROOT)),
                        "artifact_status": artifact.get("artifact_status"),
                        "rerank_applied": artifact.get("rerank_applied"),
                        "rerank_skip_reason": artifact.get("rerank_skip_reason"),
                        "query_constraint_count": len(artifact.get("query_constraints") or []),
                    }
                )

    return prompts, call_plan, raw_outputs, evaluation_metadata


def _write_token_estimate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _summary_markdown(config: dict[str, Any], prompts: list[dict[str, Any]], call_plan: list[dict[str, Any]]) -> str:
    methods = sorted({row["method_id"] for row in prompts})
    total_input = sum(int(row["estimated_input_tokens"]) for row in call_plan)
    total_output = sum(int(row["estimated_output_tokens"]) for row in call_plan)
    total_cost = sum(float(row["estimated_cost_cny"]) for row in call_plan)
    return "\n".join(
        [
            "# Phase 6-B generation contrast dry run",
            "",
            "- No external LLM API was called.",
            "- This run prepares matched vector-only and graph-enhanced generation prompts.",
            "- Gold labels and expected decisions are not included in model-facing files.",
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
            f"- samples: `{len(set(row['sample_id'] for row in prompts))}`",
            f"- methods: `{', '.join(methods)}`",
            f"- planned_calls: `{len(call_plan)}`",
            "",
            "## Token Estimate",
            "",
            f"- estimated_input_tokens_total: `{total_input}`",
            f"- estimated_output_tokens_total: `{total_output}`",
            f"- estimated_cost_cny_total: `{total_cost:.8f}`",
            "",
            "This is a development dry run, not a paper-level effectiveness result.",
            "",
        ]
    )


def run_builder(config: dict[str, Any], *, output_dir: Path | None = None) -> dict[str, Any]:
    if output_dir is None:
        output_root = _resolve_path(config["output_root"])
        run_id = (
            f"{config['run_id_prefix']}_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
        output_dir = output_root / run_id
    else:
        output_dir = Path(output_dir)
        run_id = output_dir.name

    output_dir.mkdir(parents=True, exist_ok=False)
    prompts, call_plan, raw_outputs, evaluation_metadata = build_records(config)
    run_manifest = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_version": config.get("config_version"),
        "dataset_version": config.get("dataset_version"),
        "kb_version": config.get("kb_version"),
        "prompt_version": config.get("prompt_version"),
        "input_run_dir": str(_resolve_path(config["input_run_dir"])),
        "external_model_calls": 0,
        "estimated_cost": 0,
    }
    summary = {
        "run_id": run_id,
        "config_version": config.get("config_version"),
        "sample_count": len(set(row["sample_id"] for row in prompts)),
        "planned_calls": len(call_plan),
        "method_count": len(set(row["method_id"] for row in prompts)),
        "external_model_calls": 0,
        "estimated_cost": 0,
    }

    _write_json(output_dir / "run_manifest.json", run_manifest)
    _write_json(output_dir / "summary.json", summary)
    _write_jsonl(output_dir / "prompts.jsonl", prompts)
    _write_jsonl(output_dir / "call_plan.jsonl", call_plan)
    _write_jsonl(output_dir / "raw_model_outputs.jsonl", raw_outputs)
    _write_jsonl(output_dir / "evaluation_metadata.jsonl", evaluation_metadata)
    _write_token_estimate_csv(output_dir / "token_usage_estimate.csv", call_plan)
    (output_dir / "summary.md").write_text(
        _summary_markdown(config, prompts, call_plan),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config = _read_json(_resolve_path(args.config))
    output_dir = _resolve_path(args.output_dir) if args.output_dir else None
    summary = run_builder(config, output_dir=output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
