"""Evaluate isolated Validation40 candidate-budget rankings against frozen Gold."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


def _load_base_module():
    path = Path(__file__).with_name("validation40_hybrid_retrieval.py")
    spec = importlib.util.spec_from_file_location("validation40_budget_eval_base", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot load retrieval base module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base_module()


def normalize_source_name(source: object) -> str:
    return Path(str(source or "").strip()).stem.casefold()


def join_gold_and_retrieval(
    gold_rows: list[dict[str, Any]], retrieval_rows: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    gold_by_id = {str(row.get("candidate_id")): row for row in gold_rows}
    retrieval_by_id = {str(row.get("sample_id")): row for row in retrieval_rows}
    if set(gold_by_id) != set(retrieval_by_id):
        raise ValueError("Gold and retrieval sample IDs do not match")
    return [(gold_by_id[key], retrieval_by_id[key]) for key in sorted(gold_by_id)]


def _rank_metrics(
    candidates: list[dict[str, Any]],
    *,
    gold_source: str,
    gold_page: int,
    adjacent_page_tolerance: int,
) -> dict[str, Any]:
    source_rank: int | None = None
    strict_rank: int | None = None
    adjacent_rank: int | None = None
    for rank, item in enumerate(candidates, start=1):
        if normalize_source_name(item.get("source_file")) != gold_source:
            continue
        source_rank = source_rank or rank
        page = int(item.get("page_number") or 0)
        if page == gold_page:
            strict_rank = strict_rank or rank
        if abs(page - gold_page) <= adjacent_page_tolerance:
            adjacent_rank = adjacent_rank or rank
    return {
        "source_recall": int(source_rank is not None),
        "strict_source_page_recall": int(strict_rank is not None),
        "adjacent_source_page_recall": int(adjacent_rank is not None),
        "source_rank": source_rank,
        "strict_source_page_rank": strict_rank,
        "strict_source_page_mrr": 0.0 if strict_rank is None else 1.0 / strict_rank,
    }


def evaluate_budget_rankings(
    *,
    sample_id: str,
    method_id: str,
    gold: dict[str, Any],
    rankings: dict[str, list[dict[str, Any]]],
    budgets: list[int] | tuple[int, ...],
    adjacent_page_tolerance: int,
) -> dict[str, Any]:
    gold_source = normalize_source_name(gold.get("source_filename"))
    gold_page = int(gold.get("page_number") or 0)
    budget_metrics: dict[str, dict[str, Any]] = {}
    first_strict_hit_budget: int | None = None
    for budget in budgets:
        key = str(budget)
        if key not in rankings:
            raise ValueError(f"Missing candidate budget {budget} for {sample_id}/{method_id}")
        metrics = _rank_metrics(
            list(rankings[key]),
            gold_source=gold_source,
            gold_page=gold_page,
            adjacent_page_tolerance=adjacent_page_tolerance,
        )
        budget_metrics[key] = metrics
        if metrics["strict_source_page_recall"] and first_strict_hit_budget is None:
            first_strict_hit_budget = budget
    return {
        "sample_id": sample_id,
        "method_id": method_id,
        "gold_source_filename": gold.get("source_filename"),
        "gold_page_number": gold_page,
        "budgets": budget_metrics,
        "first_strict_hit_budget": first_strict_hit_budget,
    }


def summarize_budget_curves(
    *,
    rows: list[dict[str, Any]],
    methods: list[str] | tuple[str, ...],
    budgets: list[int] | tuple[int, ...],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for method in methods:
        method_rows = [row for row in rows if row["method_id"] == method]
        sample_count = len(method_rows)
        if not sample_count:
            raise ValueError(f"No metric rows for method {method}")
        strict_hits = {
            str(budget): sum(
                int(row["budgets"][str(budget)]["strict_source_page_recall"])
                for row in method_rows
            )
            for budget in budgets
        }
        new_hits = {
            str(budget): sum(
                int(row.get("first_strict_hit_budget") == budget) for row in method_rows
            )
            for budget in budgets
        }
        max_key = str(budgets[-1])
        summaries.append(
            {
                "method_id": method,
                "sample_count": sample_count,
                "strict_hits_at_budget": strict_hits,
                "strict_recall_at_budget": {
                    key: value / sample_count for key, value in strict_hits.items()
                },
                "new_strict_hits_by_budget": new_hits,
                "source_recall_at_budget": {
                    str(budget): sum(
                        int(row["budgets"][str(budget)]["source_recall"])
                        for row in method_rows
                    )
                    / sample_count
                    for budget in budgets
                },
                "adjacent_recall_at_budget": {
                    str(budget): sum(
                        int(row["budgets"][str(budget)]["adjacent_source_page_recall"])
                        for row in method_rows
                    )
                    / sample_count
                    for budget in budgets
                },
                "strict_mrr_at_budget": {
                    str(budget): sum(
                        float(row["budgets"][str(budget)]["strict_source_page_mrr"])
                        for row in method_rows
                    )
                    / sample_count
                    for budget in budgets
                },
                "strict_miss_after_max_budget_count": sum(
                    1
                    for row in method_rows
                    if not row["budgets"][max_key]["strict_source_page_recall"]
                ),
                "strict_miss_after_max_budget_sample_ids": [
                    row["sample_id"]
                    for row in method_rows
                    if not row["budgets"][max_key]["strict_source_page_recall"]
                ],
            }
        )
    return summaries


def _summary_markdown(summaries: list[dict[str, Any]], budgets: list[int]) -> str:
    lines = [
        "# Validation40 Candidate Budget Curves",
        "",
        "本报告只衡量候选池覆盖能力，不代表最终回答质量或临床有效性。",
        "",
        "| Method | " + " | ".join(f"Strict@{budget}" for budget in budgets) + " | Miss@Max |",
        "|---|" + "---:|" * (len(budgets) + 1),
    ]
    for summary in summaries:
        recalls = [summary["strict_recall_at_budget"][str(budget)] for budget in budgets]
        lines.append(
            "| "
            + str(summary["method_id"])
            + " | "
            + " | ".join(f"{value:.4f}" for value in recalls)
            + f" | {summary['strict_miss_after_max_budget_count']} |"
        )
    lines.extend(
        [
            "",
            "- Retrieval 与 Gold 评测物理隔离。",
            "- Pilot Test80 未读取。",
            "- 外部模型/API 调用与 token 费用均为 0。",
            "",
        ]
    )
    return "\n".join(lines)


def run_candidate_budget_evaluation(
    *,
    gold_path: Path,
    retrieval_path: Path,
    retrieval_audit_path: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    expected_gold_sha = str(config["expected_gold_sha256"]).lower()
    if BASE.sha256_file(gold_path).lower() != expected_gold_sha:
        raise ValueError("Validation40 Gold SHA-256 mismatch")
    retrieval_audit = BASE._read_json(retrieval_audit_path)
    if retrieval_audit.get("execution_version") != config.get(
        "expected_retrieval_execution_version"
    ):
        raise ValueError("Candidate-budget retrieval version mismatch")
    if BASE.sha256_file(retrieval_path) != retrieval_audit.get("results_sha256"):
        raise ValueError("Candidate-budget retrieval SHA-256 mismatch")
    if retrieval_audit.get("gold_accessed") is not False:
        raise ValueError("Retrieval audit does not prove Gold isolation")
    if retrieval_audit.get("pilot_test_accessed") is not False:
        raise ValueError("Retrieval audit does not prove Pilot Test isolation")
    if retrieval_audit.get("pilot_test_sha256_before") != retrieval_audit.get(
        "pilot_test_sha256_after"
    ):
        raise ValueError("Pilot Test hash changed during retrieval")

    methods = tuple(str(value) for value in config["methods"])
    budgets = tuple(int(value) for value in config["candidate_budgets"])
    joined = join_gold_and_retrieval(
        BASE._read_jsonl(gold_path), BASE._read_jsonl(retrieval_path)
    )
    sample_metrics: list[dict[str, Any]] = []
    for gold, retrieval in joined:
        retrieval_methods = retrieval.get("methods") or {}
        if set(retrieval_methods) != set(methods):
            raise ValueError(f"Method set mismatch for {retrieval.get('sample_id')}")
        for method in methods:
            sample_metrics.append(
                evaluate_budget_rankings(
                    sample_id=str(retrieval["sample_id"]),
                    method_id=method,
                    gold=gold,
                    rankings=retrieval_methods[method],
                    budgets=budgets,
                    adjacent_page_tolerance=int(config.get("adjacent_page_tolerance", 1)),
                )
            )
    summaries = summarize_budget_curves(
        rows=sample_metrics, methods=methods, budgets=budgets
    )
    audit = {
        "evaluation_version": "validation40-candidate-budget-evaluation-v0.1",
        "phase": "Phase 7-C1c-4e-1",
        "gold_sha256": BASE.sha256_file(gold_path),
        "retrieval_sha256": BASE.sha256_file(retrieval_path),
        "retrieval_audit_sha256": BASE.sha256_file(retrieval_audit_path),
        "canonical_config_sha256": hashlib.sha256(BASE._json_bytes(config)).hexdigest(),
        "sample_count": len(joined),
        "method_count": len(methods),
        "metric_row_count": len(sample_metrics),
        "candidate_budgets": list(budgets),
        "pilot_test_accessed": False,
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0.0,
        "clinically_validated": False,
    }
    outputs = {
        "validation40_candidate_budget_sample_metrics_v0_1.jsonl": BASE._jsonl_bytes(sample_metrics),
        "validation40_candidate_budget_summary_v0_1.json": BASE._json_bytes(summaries),
        "validation40_candidate_budget_evaluation_audit_v0_1.json": BASE._json_bytes(audit),
        "validation40_candidate_budget_summary_v0_1.md": _summary_markdown(
            summaries, list(budgets)
        ).encode("utf-8"),
    }
    for name, content in outputs.items():
        BASE._atomic_write(output_dir / name, content)
    return {"sample_metrics": sample_metrics, "summaries": summaries, "audit": audit}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    config = BASE._read_json(args.config)
    root = args.repo_root.resolve()
    result = run_candidate_budget_evaluation(
        gold_path=root / config["gold_path"],
        retrieval_path=root / config["retrieval_path"],
        retrieval_audit_path=root / config["retrieval_audit_path"],
        output_dir=root / config["output_dir"],
        config=config,
    )
    print(json.dumps(result["summaries"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
