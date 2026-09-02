"""Gold-free coverage audit for Phase 7 runtime graph constraints."""

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
from experiments.phase7_formal_experiments.runtime_graph_constraint_extractor import (
    assess_constraint_path,
    extract_graph_runtime_constraints,
)


AUDIT_VERSION = "phase7-runtime-constraint-coverage-v0.2"
BROAD_PATH_THRESHOLD = 200


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


def _nearest_rank_percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def build_coverage_audit(
    *,
    runtime_projection_path: Path,
    graph_index_path: Path,
    lexicon_path: Path,
    minimum_matched_constraint_types: int,
    broad_path_threshold: int = BROAD_PATH_THRESHOLD,
    allow_specific_condition_class_path: bool = False,
    audit_version: str = AUDIT_VERSION,
) -> dict:
    if minimum_matched_constraint_types < 1:
        raise ValueError("minimum_matched_constraint_types must be positive")

    runtime_rows = _read_jsonl(runtime_projection_path)
    graph_index = _read_json(graph_index_path)
    lexicon = _read_json(lexicon_path)
    assert_no_gold_only_content(runtime_rows)
    assert_no_gold_only_content(graph_index)
    assert_no_gold_only_content(lexicon)

    raw_candidates = graph_index.get("candidates")
    if not isinstance(raw_candidates, dict):
        raise ValueError("graph index candidates must be a dictionary")

    if broad_path_threshold < 1:
        raise ValueError("broad_path_threshold must be positive")

    candidate_context_constraints: dict[str, list[dict]] = {}
    candidate_content_constraints: dict[str, list[dict]] = {}
    candidate_sources: dict[str, str] = {}
    for candidate_key, row in sorted(raw_candidates.items()):
        if not isinstance(row, dict):
            raise TypeError("graph candidate must be a dictionary")
        key = str(candidate_key)
        content = str(row.get("content", ""))
        candidate_content_constraints[key] = extract_graph_runtime_constraints(
            content,
            lexicon=lexicon,
        )
        candidate_context_constraints[key] = extract_graph_runtime_constraints(
                content,
                str(row.get("source_file", "")),
                str(row.get("chapter_title", "")),
                lexicon=lexicon,
        )
        candidate_sources[key] = str(row.get("source_file", ""))

    context_posting_counts: Counter[tuple[str, str]] = Counter()
    content_posting_counts: Counter[tuple[str, str]] = Counter()
    for constraints in candidate_context_constraints.values():
        context_posting_counts.update(
            {
                (
                    str(constraint["constraint_type"]),
                    str(constraint["normalized_value"]),
                )
                for constraint in constraints
            }
        )
    for constraints in candidate_content_constraints.values():
        content_posting_counts.update(
            {
                (
                    str(constraint["constraint_type"]),
                    str(constraint["normalized_value"]),
                )
                for constraint in constraints
            }
        )

    question_details: list[dict] = []
    path_failure_reason_counts: Counter[str] = Counter()
    for row in runtime_rows:
        sample_id = str(row.get("sample_id", "")).strip()
        question = str(row.get("question", "")).strip()
        if not sample_id or not question:
            raise ValueError("runtime row requires sample_id and question")
        query_constraints = extract_graph_runtime_constraints(
            question,
            lexicon=lexicon,
        )
        potential_path_count = 0
        candidate_source_set: set[str] = set()
        matched_signature_counts: Counter[str] = Counter()
        local_failure_reasons: Counter[str] = Counter()
        maximum_candidate_matched_type_count = 0
        for candidate_key, constraints in candidate_context_constraints.items():
            assessment = assess_constraint_path(
                query_constraints,
                constraints,
                candidate_content_constraints[candidate_key],
                minimum_matched_constraint_types=(
                    minimum_matched_constraint_types
                ),
                allow_specific_condition_class_path=(
                    allow_specific_condition_class_path
                ),
            )
            maximum_candidate_matched_type_count = max(
                maximum_candidate_matched_type_count,
                len(assessment["matched_constraint_types"]),
            )
            if assessment["qualified"]:
                potential_path_count += 1
                candidate_source_set.add(candidate_sources[candidate_key])
                signature = "+".join(
                    assessment["matched_constraint_types"]
                )
                matched_signature_counts[signature] += 1
            else:
                reason = str(assessment["reason"])
                local_failure_reasons[reason] += 1
                path_failure_reason_counts[reason] += 1

        query_pairs = sorted(
            {
                (
                    str(constraint["constraint_type"]),
                    str(constraint["normalized_value"]),
                )
                for constraint in query_constraints
            }
        )
        constraint_posting_counts = {
            f"{constraint_type}::{normalized_value}": {
                "context": context_posting_counts[
                    (constraint_type, normalized_value)
                ],
                "content": content_posting_counts[
                    (constraint_type, normalized_value)
                ],
            }
            for constraint_type, normalized_value in query_pairs
        }
        posting_constraint_types = {
            constraint_type
            for constraint_type, normalized_value in query_pairs
            if context_posting_counts[(constraint_type, normalized_value)] > 0
        }
        top_failure_reasons = [
            {"reason": reason, "count": count}
            for reason, count in sorted(
                local_failure_reasons.items(),
                key=lambda item: (-item[1], item[0]),
            )[:5]
        ]

        if not query_constraints:
            zero_path_reason = "zero_query_constraints"
            zero_path_diagnosis = "zero_query_constraints"
        elif len({row["constraint_type"] for row in query_constraints}) < (
            minimum_matched_constraint_types
        ):
            zero_path_reason = "insufficient_query_types"
            zero_path_diagnosis = "insufficient_query_types"
        elif potential_path_count > 0:
            zero_path_reason = "not_applicable"
            zero_path_diagnosis = "not_applicable"
        elif local_failure_reasons.get("no_content_supported_match", 0):
            zero_path_reason = "no_content_supported_match"
            zero_path_diagnosis = "path_policy_filtered"
        elif local_failure_reasons.get("broad_content_only", 0):
            zero_path_reason = "broad_content_only"
            zero_path_diagnosis = "path_policy_filtered"
        else:
            zero_path_reason = "no_posting"
            if len(posting_constraint_types) < minimum_matched_constraint_types:
                zero_path_diagnosis = "insufficient_posting_types"
            elif maximum_candidate_matched_type_count < (
                minimum_matched_constraint_types
            ):
                zero_path_diagnosis = "no_candidate_cooccurrence"
            else:
                zero_path_diagnosis = "path_policy_filtered"
        question_details.append(
            {
                "sample_id": sample_id,
                "constraint_count": len(query_constraints),
                "constraint_type_count": len(
                    {
                        constraint["constraint_type"]
                        for constraint in query_constraints
                    }
                ),
                "potential_path_count": potential_path_count,
                "candidate_source_count": len(candidate_source_set),
                "matched_signature_counts": dict(
                    sorted(matched_signature_counts.items())
                ),
                "zero_path_reason": zero_path_reason,
                "zero_path_diagnosis": zero_path_diagnosis,
                "maximum_candidate_matched_type_count": (
                    maximum_candidate_matched_type_count
                ),
                "constraint_posting_counts": constraint_posting_counts,
                "top_failure_reasons": top_failure_reasons,
            }
        )

    potential_path_counts = [
        detail["potential_path_count"]
        for detail in question_details
    ]
    potential_path_distribution = {
        "minimum": min(potential_path_counts, default=0),
        "median": statistics.median(potential_path_counts)
        if potential_path_counts
        else 0,
        "p95": _nearest_rank_percentile(potential_path_counts, 0.95),
        "maximum": max(potential_path_counts, default=0),
    }
    zero_path_diagnosis_counts = Counter(
        detail["zero_path_diagnosis"]
        for detail in question_details
        if detail["potential_path_count"] == 0
    )

    payload = {
        "audit_version": audit_version,
        "lexicon_version": lexicon.get("lexicon_version"),
        "minimum_matched_constraint_types": minimum_matched_constraint_types,
        "question_count": len(runtime_rows),
        "candidate_count": len(candidate_context_constraints),
        "zero_query_constraint_count": sum(
            detail["constraint_count"] == 0
            for detail in question_details
        ),
        "query_two_type_coverage_count": sum(
            detail["constraint_type_count"] >= minimum_matched_constraint_types
            for detail in question_details
        ),
        "candidate_nonempty_constraint_count": sum(
            bool(constraints)
            for constraints in candidate_context_constraints.values()
        ),
        "question_with_potential_path_count": sum(
            detail["potential_path_count"] > 0
            for detail in question_details
        ),
        "zero_potential_path_count": sum(
            detail["potential_path_count"] == 0
            for detail in question_details
        ),
        "zero_path_diagnosis_counts": dict(
            sorted(zero_path_diagnosis_counts.items())
        ),
        "broad_path_threshold": broad_path_threshold,
        "broad_path_question_count": sum(
            detail["potential_path_count"] > broad_path_threshold
            for detail in question_details
        ),
        "potential_path_distribution": potential_path_distribution,
        "path_failure_reason_counts": dict(
            sorted(path_failure_reason_counts.items())
        ),
        "cross_condition_conflict_count": path_failure_reason_counts.get(
            "condition_conflict",
            0,
        ),
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
    if audit_version != AUDIT_VERSION or allow_specific_condition_class_path:
        payload["path_policy"] = {
            "allow_specific_condition_class_path": (
                allow_specific_condition_class_path
            ),
            "broad_condition_class_remains_blocked": True,
        }
    payload["deterministic_payload_sha256"] = _deterministic_hash(payload)
    return payload


def _render_report(audit: dict) -> str:
    report_version = str(audit["audit_version"]).rsplit("-v", maxsplit=1)[-1]
    lines = [
            f"# Phase 7 运行时图约束覆盖审计 v{report_version}",
            "",
            f"- 问题数：{audit['question_count']}",
            f"- 候选数：{audit['candidate_count']}",
            f"- 零约束问题：{audit['zero_query_constraint_count']}",
            (
                "- 达到最少约束类型的问题："
                f"{audit['query_two_type_coverage_count']}"
            ),
            (
                "- 至少存在一条潜在图路径的问题："
                f"{audit['question_with_potential_path_count']}"
            ),
            f"- 零潜在图路径问题：{audit['zero_potential_path_count']}",
            (
                "- 零路径诊断分布："
                + "；".join(
                    f"{reason}={count}"
                    for reason, count in audit[
                        "zero_path_diagnosis_counts"
                    ].items()
                )
            ),
            (
                "- 路径数量分布（最小/中位/P95/最大）："
                f"{audit['potential_path_distribution']['minimum']}/"
                f"{audit['potential_path_distribution']['median']}/"
                f"{audit['potential_path_distribution']['p95']}/"
                f"{audit['potential_path_distribution']['maximum']}"
            ),
            f"- 异常宽泛路径问题：{audit['broad_path_question_count']}",
    ]
    if "path_policy" in audit:
        lines.append(
            (
                "- 明确疾病-特定药物类别路径："
                + (
                    "启用"
                    if audit["path_policy"][
                        "allow_specific_condition_class_path"
                    ]
                    else "关闭"
                )
            ),
        )
    lines.extend(
        [
            f"- 疾病冲突过滤计数：{audit['cross_condition_conflict_count']}",
            "- 外部模型调用：0",
            "- Pilot Test80 内容访问：否",
            "",
            "该审计只评估 Gold-free 运行时约束覆盖，不代表最终检索召回提升。",
            "",
        ]
    )
    return "\n".join(lines)


def run_from_config(config_path: Path) -> dict:
    config = _read_json(config_path)
    base_dir = Path.cwd()
    runtime_path = base_dir / config["runtime_projection_path"]
    graph_path = base_dir / config["graph_index_path"]
    lexicon_path = base_dir / config["lexicon_path"]
    output_dir = base_dir / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    audit = build_coverage_audit(
        runtime_projection_path=runtime_path,
        graph_index_path=graph_path,
        lexicon_path=lexicon_path,
        minimum_matched_constraint_types=int(
            config["minimum_matched_constraint_types"]
        ),
        broad_path_threshold=int(
            config.get("broad_path_threshold", BROAD_PATH_THRESHOLD)
        ),
        allow_specific_condition_class_path=bool(
            config.get("allow_specific_condition_class_path", False)
        ),
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
        "audit_version": audit["audit_version"],
        "inputs": {
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
