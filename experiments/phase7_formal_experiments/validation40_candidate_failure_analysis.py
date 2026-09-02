from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


def normalize_source_name(source: object) -> str:
    return str(source or "").strip().casefold()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
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
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Immutable output already exists with different bytes: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def build_route_union(
    methods: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    route_order = [
        ("rrf", "dense_sparse_rrf"),
        ("dense", "bge_m3_dense"),
        ("sparse", "bge_m3_sparse"),
    ]
    by_key: dict[str, dict[str, Any]] = {}
    ordered_keys: list[str] = []

    for route_name, method_id in route_order:
        for candidate in methods[method_id]["candidates_top20"]:
            key = str(candidate["candidate_key"])
            if key not in by_key:
                by_key[key] = dict(candidate)
                by_key[key]["union_routes"] = []
                ordered_keys.append(key)
            by_key[key]["union_routes"].append(route_name)

    return [by_key[key] for key in ordered_keys]


def aggregate_source_pages(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    ordered_keys: list[tuple[str, int]] = []

    for candidate in candidates:
        source_page_key = (
            normalize_source_name(candidate.get("source_file")),
            int(candidate.get("page_number", 0)),
        )
        if source_page_key not in grouped:
            grouped[source_page_key] = dict(candidate)
            grouped[source_page_key]["source_page_member_keys"] = []
            ordered_keys.append(source_page_key)
        grouped[source_page_key]["source_page_member_keys"].append(
            str(candidate["candidate_key"])
        )

    for item in grouped.values():
        item["source_page_member_count"] = len(item["source_page_member_keys"])
    return [grouped[key] for key in ordered_keys]


def audit_candidate_budget_availability(
    *,
    methods: dict[str, dict[str, Any]],
    requested_budgets: list[int],
) -> dict[str, Any]:
    exposed_counts = [
        len(methods[method_id]["candidates_top20"])
        for method_id in ("bge_m3_dense", "bge_m3_sparse", "dense_sparse_rrf")
    ]
    single_route_exposed_k = min(exposed_counts)
    route_union_count = len(build_route_union(methods))
    return {
        "single_route_exposed_k": single_route_exposed_k,
        "route_union_exposed_count": route_union_count,
        "route_union_is_true_top_k": False,
        "true_retrieval_budget_status": {
            str(budget): (
                "available" if budget <= single_route_exposed_k else "requires_rerun"
            )
            for budget in requested_budgets
        },
    }


def _has_exact_page(candidates: list[dict[str, Any]], gold: dict[str, Any]) -> bool:
    gold_source = normalize_source_name(gold.get("source_filename"))
    gold_page = int(gold.get("page_number", 0))
    return any(
        normalize_source_name(item.get("source_file")) == gold_source
        and int(item.get("page_number", 0)) == gold_page
        for item in candidates
    )


def _has_source(candidates: list[dict[str, Any]], gold: dict[str, Any]) -> bool:
    gold_source = normalize_source_name(gold.get("source_filename"))
    return any(
        normalize_source_name(item.get("source_file")) == gold_source
        for item in candidates
    )


def _has_adjacent_page(
    candidates: list[dict[str, Any]],
    gold: dict[str, Any],
    tolerance: int,
) -> bool:
    gold_source = normalize_source_name(gold.get("source_filename"))
    gold_page = int(gold.get("page_number", 0))
    return any(
        normalize_source_name(item.get("source_file")) == gold_source
        and 0 < abs(int(item.get("page_number", 0)) - gold_page) <= tolerance
        for item in candidates
    )


def classify_candidate_failure(
    *,
    gold: dict[str, Any],
    methods: dict[str, dict[str, Any]],
    adjacent_page_tolerance: int,
) -> dict[str, Any]:
    rrf_candidates = methods["dense_sparse_rrf"]["candidates_top20"]
    dense_hit = _has_exact_page(
        methods["bge_m3_dense"]["candidates_top20"], gold
    )
    sparse_hit = _has_exact_page(
        methods["bge_m3_sparse"]["candidates_top20"], gold
    )
    rrf_hit = _has_exact_page(rrf_candidates, gold)
    rrf_source_hit = _has_source(rrf_candidates, gold)
    rrf_adjacent_hit = _has_adjacent_page(
        rrf_candidates, gold, adjacent_page_tolerance
    )

    if dense_hit and not rrf_hit:
        primary_failure_type = "fusion_dropped_valid_dense"
    elif not rrf_hit and rrf_adjacent_hit:
        primary_failure_type = "adjacent_page_only"
    elif not rrf_hit and rrf_source_hit:
        primary_failure_type = "source_present_page_absent"
    elif not rrf_hit:
        primary_failure_type = "source_absent"
    else:
        primary_failure_type = ""
    return {
        "primary_failure_type": primary_failure_type,
        "dense_exact_page_hit": dense_hit,
        "sparse_exact_page_hit": sparse_hit,
        "rrf_exact_page_hit": rrf_hit,
        "rrf_source_hit": rrf_source_hit,
        "rrf_adjacent_page_hit": rrf_adjacent_hit,
        "all_routes_exact_page_miss": not (dense_hit or sparse_hit or rrf_hit),
    }


def _source_crowding(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts = Counter(
        normalize_source_name(item.get("source_file")) for item in candidates
    )
    candidate_count = len(candidates)
    maximum = max(source_counts.values(), default=0)
    return {
        "candidate_count": candidate_count,
        "unique_source_count": len(source_counts),
        "maximum_candidates_from_one_source": maximum,
        "maximum_source_share": maximum / candidate_count if candidate_count else 0.0,
    }


def _join_rows(
    gold_rows: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    gold_by_id = {str(row["candidate_id"]): row for row in gold_rows}
    retrieval_by_id = {str(row["sample_id"]): row for row in retrieval_rows}
    if len(gold_by_id) != len(gold_rows) or len(retrieval_by_id) != len(retrieval_rows):
        raise ValueError("Duplicate sample IDs detected")
    if set(gold_by_id) != set(retrieval_by_id):
        raise ValueError("Gold and retrieval sample IDs do not match")
    return [(gold_by_id[sample_id], retrieval_by_id[sample_id]) for sample_id in gold_by_id]


def _control_metrics(
    *,
    gold: dict[str, Any],
    methods: dict[str, dict[str, Any]],
    requested_budgets: list[int],
) -> dict[str, Any]:
    rrf_candidates = methods["dense_sparse_rrf"]["candidates_top20"]
    route_union = build_route_union(methods)
    source_pages = aggregate_source_pages(route_union)
    return {
        "rrf_top20_strict_page_hit": _has_exact_page(rrf_candidates, gold),
        "route_union_exposed_strict_page_hit": _has_exact_page(route_union, gold),
        "source_page_aggregate_strict_page_hit": _has_exact_page(source_pages, gold),
        "route_union_candidate_count": len(route_union),
        "source_page_aggregate_count": len(source_pages),
        "rrf_source_crowding": _source_crowding(rrf_candidates),
        "route_union_source_crowding": _source_crowding(route_union),
        "budget_availability": audit_candidate_budget_availability(
            methods=methods,
            requested_budgets=requested_budgets,
        ),
    }


def _summary_markdown(
    *,
    sample_count: int,
    failure_cases: list[dict[str, Any]],
    taxonomy: dict[str, Any],
    controls: dict[str, Any],
) -> str:
    lines = [
        "# Validation40 候选缺页失败审计 v0.1",
        "",
        f"- Validation40 样本数：`{sample_count}`",
        f"- RRF Top-20 严格来源页缺失：`{len(failure_cases)}`",
        f"- 路由并集恢复严格来源页：`{controls['route_union_recovered_count']}`",
        f"- 三路均未暴露严格来源页：`{taxonomy['all_routes_exact_page_miss_count']}`",
        "",
        "## 唯一主失败类型",
        "",
    ]
    for failure_type, count in taxonomy["primary_failure_type_counts"].items():
        lines.append(f"- `{failure_type}`：`{count}`")
    lines.extend(
        [
            "",
            "## 非图控制边界",
            "",
            "当前冻结结果每条单路只保存 Top-20。路由并集是已暴露候选的去重并集，",
            "不能冒充真实单路 Top-40/80。真实 Top-40/80 必须重新执行检索后另行评测。",
            "",
            "本步骤不读取 Pilot Test80 内容，不调用外部模型，不产生 token 或费用。",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(
    *,
    gold_path: Path,
    retrieval_path: Path,
    retrieval_audit_path: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    if sha256_file(gold_path).lower() != str(config["expected_gold_sha256"]).lower():
        raise ValueError("Validation40 Gold SHA-256 mismatch")
    if sha256_file(retrieval_path).lower() != str(
        config["expected_retrieval_sha256"]
    ).lower():
        raise ValueError("Validation40 retrieval SHA-256 mismatch")
    if sha256_file(retrieval_audit_path).lower() != str(
        config["expected_retrieval_audit_sha256"]
    ).lower():
        raise ValueError("Validation40 retrieval audit SHA-256 mismatch")

    retrieval_audit = _read_json(retrieval_audit_path)
    if retrieval_audit.get("pilot_test_accessed") is not False:
        raise ValueError("Retrieval audit does not prove Pilot Test isolation")

    joined = _join_rows(_read_jsonl(gold_path), _read_jsonl(retrieval_path))
    tolerance = int(config.get("adjacent_page_tolerance", 1))
    budgets = [int(value) for value in config.get("requested_candidate_budgets", [20, 40, 80])]
    failure_cases: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []

    for gold, retrieval in joined:
        methods = retrieval.get("methods") or {}
        required_methods = {"bge_m3_dense", "bge_m3_sparse", "dense_sparse_rrf"}
        if not required_methods.issubset(methods):
            raise ValueError(f"Required retrieval methods missing for {retrieval.get('sample_id')}")
        diagnosis = classify_candidate_failure(
            gold=gold,
            methods=methods,
            adjacent_page_tolerance=tolerance,
        )
        controls = _control_metrics(
            gold=gold,
            methods=methods,
            requested_budgets=budgets,
        )
        control_rows.append({"sample_id": str(retrieval["sample_id"]), **controls})
        if diagnosis["rrf_exact_page_hit"]:
            continue
        failure_cases.append(
            {
                "sample_id": str(retrieval["sample_id"]),
                "question": str(retrieval.get("question", "")),
                "gold_source_filename": str(gold["source_filename"]),
                "gold_page_number": int(gold["page_number"]),
                **diagnosis,
                "route_union_exposed_strict_page_hit": controls[
                    "route_union_exposed_strict_page_hit"
                ],
                "rrf_source_crowding": controls["rrf_source_crowding"],
            }
        )

    primary_counts = Counter(row["primary_failure_type"] for row in failure_cases)
    taxonomy = {
        "analysis_version": str(config["analysis_version"]),
        "validation_sample_count": len(joined),
        "rrf_strict_page_miss_count": len(failure_cases),
        "primary_failure_type_counts": dict(sorted(primary_counts.items())),
        "rrf_source_absent_count": sum(
            int(not row["rrf_source_hit"]) for row in failure_cases
        ),
        "rrf_source_present_page_absent_count": sum(
            int(row["rrf_source_hit"] and not row["rrf_exact_page_hit"])
            for row in failure_cases
        ),
        "rrf_adjacent_page_only_count": sum(
            int(row["rrf_adjacent_page_hit"] and not row["rrf_exact_page_hit"])
            for row in failure_cases
        ),
        "dense_exact_rrf_miss_count": sum(
            int(row["dense_exact_page_hit"] and not row["rrf_exact_page_hit"])
            for row in failure_cases
        ),
        "all_routes_exact_page_miss_count": sum(
            int(row["all_routes_exact_page_miss"]) for row in failure_cases
        ),
    }
    route_union_recovered = sum(
        int(
            not row["rrf_top20_strict_page_hit"]
            and row["route_union_exposed_strict_page_hit"]
        )
        for row in control_rows
    )
    controls_summary = {
        "analysis_version": str(config["analysis_version"]),
        "sample_count": len(control_rows),
        "requested_true_retrieval_budgets": budgets,
        "single_route_saved_candidate_k": min(
            row["budget_availability"]["single_route_exposed_k"]
            for row in control_rows
        ),
        "true_retrieval_budget_status": {
            str(budget): (
                "available"
                if all(
                    row["budget_availability"]["true_retrieval_budget_status"][str(budget)]
                    == "available"
                    for row in control_rows
                )
                else "requires_rerun"
            )
            for budget in budgets
        },
        "route_union_is_true_top_k": False,
        "route_union_recovered_count": route_union_recovered,
        "route_union_strict_page_hit_count": sum(
            int(row["route_union_exposed_strict_page_hit"]) for row in control_rows
        ),
        "source_page_aggregate_strict_page_hit_count": sum(
            int(row["source_page_aggregate_strict_page_hit"]) for row in control_rows
        ),
        "mean_rrf_maximum_source_share": sum(
            row["rrf_source_crowding"]["maximum_source_share"] for row in control_rows
        )
        / len(control_rows),
        "mean_route_union_candidate_count": sum(
            row["route_union_candidate_count"] for row in control_rows
        )
        / len(control_rows),
        "mean_source_page_aggregate_count": sum(
            row["source_page_aggregate_count"] for row in control_rows
        )
        / len(control_rows),
        "mean_source_page_duplicates_removed": sum(
            row["route_union_candidate_count"] - row["source_page_aggregate_count"]
            for row in control_rows
        )
        / len(control_rows),
    }
    audit = {
        "analysis_version": str(config["analysis_version"]),
        "gold_sha256": sha256_file(gold_path),
        "retrieval_sha256": sha256_file(retrieval_path),
        "retrieval_audit_sha256": sha256_file(retrieval_audit_path),
        "sample_count": len(joined),
        "failure_case_count": len(failure_cases),
        "pilot_test_accessed": False,
        "external_model_calls": int(config.get("external_model_calls", 0)),
        "input_tokens": int(config.get("input_tokens", 0)),
        "output_tokens": int(config.get("output_tokens", 0)),
        "estimated_cost": float(config.get("estimated_cost", 0.0)),
        "clinically_validated": False,
    }
    if any(
        audit[field] != 0
        for field in ("external_model_calls", "input_tokens", "output_tokens", "estimated_cost")
    ):
        raise ValueError("Candidate failure analysis must not call external models")

    outputs = {
        "candidate_failure_cases_v0_1.jsonl": _jsonl_bytes(failure_cases),
        "candidate_failure_taxonomy_v0_1.json": _json_bytes(taxonomy),
        "non_graph_candidate_controls_v0_1.json": _json_bytes(controls_summary),
        "candidate_failure_audit_v0_1.json": _json_bytes(audit),
        "candidate_failure_summary_v0_1.md": _summary_markdown(
            sample_count=len(joined),
            failure_cases=failure_cases,
            taxonomy=taxonomy,
            controls=controls_summary,
        ).encode("utf-8"),
    }
    for name, content in outputs.items():
        _atomic_write(output_dir / name, content)
    return {
        "failure_cases": failure_cases,
        "taxonomy": taxonomy,
        "controls": controls_summary,
        "audit": audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    config = _read_json(args.config)
    root = args.repo_root.resolve()
    result = run_analysis(
        gold_path=root / config["gold_path"],
        retrieval_path=root / config["retrieval_path"],
        retrieval_audit_path=root / config["retrieval_audit_path"],
        output_dir=root / config["output_dir"],
        config=config,
    )
    print(
        json.dumps(
            {
                "failure_case_count": len(result["failure_cases"]),
                "output_dir": str(root / config["output_dir"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
