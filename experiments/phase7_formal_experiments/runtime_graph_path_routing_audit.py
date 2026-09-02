"""Gold-free audit for source-aware Phase 7 graph path routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from experiments.phase6_evidence_graph.graph_contract import (
    assert_no_gold_only_content,
)
from experiments.phase7_formal_experiments.runtime_graph_path_router import (
    ROUTER_VERSION,
    build_runtime_path_catalog,
    route_graph_paths,
)


AUDIT_VERSION = "phase7-runtime-graph-path-routing-audit-v0.1"


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"expected JSONL objects: {path}")
    return rows


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _deterministic_hash(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _distribution(values: list[int]) -> dict:
    if not values:
        return {"minimum": 0, "median": 0, "p95": 0, "maximum": 0}
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "maximum": ordered[-1],
    }


def _quota_violations(
    selected_paths: list[dict],
    *,
    max_paths_per_source: int,
    max_paths_per_source_page: int,
) -> tuple[int, int]:
    source_counts = Counter(
        str(row.get("source_file", "")) for row in selected_paths
    )
    source_page_counts = Counter(
        (
            str(row.get("source_file", "")),
            int(row.get("page_number", 0) or 0),
        )
        for row in selected_paths
    )
    source_violations = sum(
        1 for count in source_counts.values() if count > max_paths_per_source
    )
    source_page_violations = sum(
        1
        for count in source_page_counts.values()
        if count > max_paths_per_source_page
    )
    return source_violations, source_page_violations


def build_routing_audit(
    *,
    runtime_projection_path: Path,
    graph_index_path: Path,
    lexicon_path: Path,
    allow_specific_condition_class_path: bool,
    max_total_paths: int,
    max_paths_per_source: int,
    max_paths_per_source_page: int,
    broad_path_threshold: int,
    audit_version: str = AUDIT_VERSION,
) -> dict:
    """Measure routing coverage and noise without reading benchmark Gold."""
    if broad_path_threshold < 1:
        raise ValueError("broad_path_threshold must be positive")
    runtime_rows = _read_jsonl(runtime_projection_path)
    graph_index = _read_json(graph_index_path)
    lexicon = _read_json(lexicon_path)
    assert_no_gold_only_content(runtime_rows)
    assert_no_gold_only_content(graph_index)
    assert_no_gold_only_content(lexicon)
    catalog = build_runtime_path_catalog(graph_index, lexicon)

    question_details: list[dict] = []
    total_drop_reasons: Counter[str] = Counter()
    source_quota_violation_count = 0
    source_page_quota_violation_count = 0
    for row in runtime_rows:
        sample_id = str(row.get("sample_id", "")).strip()
        question = str(row.get("question", "")).strip()
        if not sample_id or not question:
            raise ValueError("runtime row requires sample_id and question")
        routed = route_graph_paths(
            question,
            catalog=catalog,
            lexicon=lexicon,
            allow_specific_condition_class_path=(
                allow_specific_condition_class_path
            ),
            max_total_paths=max_total_paths,
            max_paths_per_source=max_paths_per_source,
            max_paths_per_source_page=max_paths_per_source_page,
        )
        total_drop_reasons.update(routed["drop_reason_counts"])
        source_violations, source_page_violations = _quota_violations(
            routed["selected_paths"],
            max_paths_per_source=max_paths_per_source,
            max_paths_per_source_page=max_paths_per_source_page,
        )
        source_quota_violation_count += source_violations
        source_page_quota_violation_count += source_page_violations
        tier_counts = Counter(
            str(path["graph_source_condition_tier"])
            for path in routed["selected_paths"]
        )
        question_details.append(
            {
                "sample_id": sample_id,
                "query_constraint_count": len(routed["query_constraints"]),
                "raw_path_count": routed["raw_path_count"],
                "routed_path_count": routed["selected_path_count"],
                "candidate_source_count": len(
                    {
                        str(path.get("source_file", ""))
                        for path in routed["selected_paths"]
                    }
                ),
                "selected_tier_counts": dict(sorted(tier_counts.items())),
                "drop_reason_counts": routed["drop_reason_counts"],
                "selected_paths": [
                    {
                        "candidate_key": str(path["candidate_key"]),
                        "source_file": str(path.get("source_file", "")),
                        "page_number": int(path.get("page_number", 0) or 0),
                        "graph_route_rank": int(path["graph_route_rank"]),
                        "graph_source_condition_tier": int(
                            path["graph_source_condition_tier"]
                        ),
                        "graph_source_condition_tier_label": str(
                            path["graph_source_condition_tier_label"]
                        ),
                        "graph_path_constraint_types": list(
                            path["graph_path_constraint_types"]
                        ),
                    }
                    for path in routed["selected_paths"]
                ],
            }
        )

    raw_counts = [detail["raw_path_count"] for detail in question_details]
    routed_counts = [
        detail["routed_path_count"] for detail in question_details
    ]
    audit = {
        "audit_version": audit_version,
        "router_version": ROUTER_VERSION,
        "graph_index_version": catalog["graph_index_version"],
        "lexicon_version": catalog["lexicon_version"],
        "question_count": len(question_details),
        "candidate_count": len(catalog["candidates"]),
        "raw_path_coverage_count": sum(count > 0 for count in raw_counts),
        "routed_path_coverage_count": sum(count > 0 for count in routed_counts),
        "raw_zero_path_count": sum(count == 0 for count in raw_counts),
        "routed_zero_path_count": sum(count == 0 for count in routed_counts),
        "raw_path_distribution": _distribution(raw_counts),
        "routed_path_distribution": _distribution(routed_counts),
        "raw_broad_path_question_count": sum(
            count > broad_path_threshold for count in raw_counts
        ),
        "routed_over_total_budget_count": sum(
            count > max_total_paths for count in routed_counts
        ),
        "source_quota_violation_count": source_quota_violation_count,
        "source_page_quota_violation_count": (
            source_page_quota_violation_count
        ),
        "drop_reason_counts": dict(sorted(total_drop_reasons.items())),
        "routing_policy": {
            "allow_specific_condition_class_path": (
                allow_specific_condition_class_path
            ),
            "max_total_paths": max_total_paths,
            "max_paths_per_source": max_paths_per_source,
            "max_paths_per_source_page": max_paths_per_source_page,
            "broad_path_threshold": broad_path_threshold,
        },
        "question_details": question_details,
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0.0,
        "guards": {
            "gold_access": False,
            "pilot_test_content_access": False,
            "external_model_calls": False,
        },
    }
    audit["deterministic_payload_sha256"] = _deterministic_hash(audit)
    return audit


def _render_report(audit: dict) -> str:
    raw = audit["raw_path_distribution"]
    routed = audit["routed_path_distribution"]
    policy = audit["routing_policy"]
    return "\n".join(
        [
            "# Phase 7 来源感知图路径路由审计 v0.1",
            "",
            "本报告仅审计运行时路径覆盖、来源优先级和固定配额，",
            "不读取 Gold-only 字段，不代表严格来源页召回或医学效果。",
            "",
            f"- 问题数：{audit['question_count']}",
            f"- 图候选数：{audit['candidate_count']}",
            f"- 原始路径覆盖：{audit['raw_path_coverage_count']}/{audit['question_count']}",
            f"- 路由后路径覆盖：{audit['routed_path_coverage_count']}/{audit['question_count']}",
            (
                "- 原始路径分布 min/median/p95/max："
                f"{raw['minimum']}/{raw['median']}/{raw['p95']}/{raw['maximum']}"
            ),
            (
                "- 路由后路径分布 min/median/p95/max："
                f"{routed['minimum']}/{routed['median']}/{routed['p95']}/{routed['maximum']}"
            ),
            (
                "- 路由配额 total/source/source-page："
                f"{policy['max_total_paths']}/{policy['max_paths_per_source']}/"
                f"{policy['max_paths_per_source_page']}"
            ),
            f"- 总预算超限问题：{audit['routed_over_total_budget_count']}",
            f"- 来源配额违规：{audit['source_quota_violation_count']}",
            f"- 来源页配额违规：{audit['source_page_quota_violation_count']}",
            f"- 外部模型调用：{audit['external_model_calls']}",
            f"- token / 费用：0 / 0.0",
            "",
        ]
    )


def run_from_config(config_path: Path) -> dict:
    config = _read_json(config_path)
    guards = config.get("execution_guards", {})
    required_guards = {
        "gold_access": False,
        "pilot_test_content_access": False,
        "external_model_calls": False,
    }
    for key, expected in required_guards.items():
        if guards.get(key) is not expected:
            raise ValueError(f"unsafe execution guard: {key}")
    root = Path.cwd()
    runtime_path = root / config["runtime_projection_path"]
    graph_path = root / config["graph_index_path"]
    lexicon_path = root / config["lexicon_path"]
    output_dir = root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = build_routing_audit(
        runtime_projection_path=runtime_path,
        graph_index_path=graph_path,
        lexicon_path=lexicon_path,
        allow_specific_condition_class_path=bool(
            config["allow_specific_condition_class_path"]
        ),
        max_total_paths=int(config["max_total_paths"]),
        max_paths_per_source=int(config["max_paths_per_source"]),
        max_paths_per_source_page=int(config["max_paths_per_source_page"]),
        broad_path_threshold=int(config["broad_path_threshold"]),
        audit_version=str(config.get("audit_version", AUDIT_VERSION)),
    )

    audit_path = output_dir / config["audit_filename"]
    report_path = output_dir / config["report_filename"]
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_render_report(audit), encoding="utf-8")
    manifest = {
        "manifest_version": "phase7-runtime-graph-path-routing-manifest-v0.1",
        "audit_version": audit["audit_version"],
        "router_version": audit["router_version"],
        "inputs": {
            str(config_path): _sha256_path(config_path),
            str(runtime_path): _sha256_path(runtime_path),
            str(graph_path): _sha256_path(graph_path),
            str(lexicon_path): _sha256_path(lexicon_path),
        },
        "outputs": {
            str(audit_path): _sha256_path(audit_path),
            str(report_path): _sha256_path(report_path),
        },
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0.0,
    }
    manifest_path = output_dir / config["manifest_filename"]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    audit = run_from_config(args.config)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
