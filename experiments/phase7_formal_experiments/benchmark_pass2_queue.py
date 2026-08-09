from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

VISIBLE_SOURCE_FIELDS = (
    "source_id",
    "source_title",
    "source_filename",
    "source_sha256",
    "source_type",
    "source_year",
    "jurisdiction",
    "source_can_support",
    "source_cannot_support",
    "page_number",
    "anchor_text_span",
    "evidence_scope",
    "age_scope",
    "applicability_conditions",
    "supported_claim_types",
    "scope_check",
    "policy_rule_ids",
)

PASS2_EMPTY_FIELDS = (
    "pass2_reviewer_id",
    "pass2_annotator_role",
    "pass2_reviewed_at",
    "pass2_outcome",
    "pass2_final_question",
    "pass2_expected_decision",
    "pass2_current_kb_support",
    "pass2_gold_evidence_status",
    "pass2_required_evidence_type",
    "pass2_required_claims",
    "pass2_allowed_claims",
    "pass2_forbidden_claims",
    "pass2_missing_evidence_type",
    "pass2_missing_information",
    "pass2_risk_labels",
    "pass2_issues_found",
    "pass2_review_reason",
)

BLINDED_FORBIDDEN_FIELDS = {
    "candidate_id",
    "annotation_order",
    "candidate_role",
    "challenge_type",
    "current_kb_support",
    "missing_evidence_type",
}


def zero_usage() -> dict[str, int | float]:
    return {
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
    }


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").split())


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _row_sha256(row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return _sha256_bytes(payload)


def _blind_id(candidate_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}|{candidate_id}".encode("utf-8")).hexdigest()[:20]
    return f"P2-{digest}"


def _assert_clean_text(row: dict[str, Any], *, context: str) -> None:
    for key, value in row.items():
        text = str(value or "")
        if "\ufffd" in text or "???" in text:
            raise ValueError(f"{context} 存在乱码字段 {key}")


def _validate_unique_ids(
    original_rows: list[dict[str, Any]],
    supplement_rows: list[dict[str, Any]],
) -> None:
    candidate_ids = [
        _normalize(row.get("candidate_id"))
        for row in [*original_rows, *supplement_rows]
    ]
    if any(not candidate_id for candidate_id in candidate_ids):
        raise ValueError("candidate_id 不能为空")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("原池与补充池存在重复 candidate_id")


def _is_promotable(row: dict[str, Any]) -> bool:
    outcome = _normalize(row.get("pass1_outcome"))
    if not outcome:
        raise ValueError(f"第一轮尚未完成: {row.get('candidate_id', '')}")
    if outcome == "accepted":
        return True
    if outcome == "revise":
        if _normalize(row.get("pass1_overlap_reaudit_status")) != "clear":
            raise ValueError(f"修订题复审未通过: {row.get('candidate_id', '')}")
        return True
    if outcome == "reject":
        return False
    raise ValueError(f"未知第一轮 outcome: {outcome}")


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "config_version",
        "dataset_version",
        "kb_version",
        "protocol_version",
        "blind_seed",
        "expected_original_promotable_before_scope",
        "expected_scope_exclusions",
        "expected_supplement_promotable",
        "expected_merged_count",
        "target_final_count",
        "target_final_decision_distribution",
        "external_model_calls",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Pass 2 配置缺少字段: {missing}")
    if int(config["external_model_calls"]) != 0:
        raise ValueError("Pass 2 队列构建不允许外部模型调用")
    target_distribution = config["target_final_decision_distribution"]
    if not isinstance(target_distribution, dict) or not target_distribution:
        raise ValueError("最终目标决策分布必须是非空对象")
    target_total = sum(int(value) for value in target_distribution.values())
    if target_total != int(config["target_final_count"]):
        raise ValueError("最终目标数量与目标决策分布合计不一致")


def _make_visible_row(
    row: dict[str, Any],
    *,
    pass2_order: int,
    pass2_item_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    question = _normalize(row.get("pass1_final_question")) or _normalize(row.get("question"))
    if not question:
        raise ValueError(f"第二轮问题为空: {row.get('candidate_id', '')}")
    visible: dict[str, Any] = {
        "pass2_order": pass2_order,
        "pass2_item_id": pass2_item_id,
        "pass2_annotation_version": config["config_version"],
        "dataset_version": config["dataset_version"],
        "kb_version": config["kb_version"],
        "protocol_version": config["protocol_version"],
        "question": question,
    }
    for key in VISIBLE_SOURCE_FIELDS:
        visible[key] = row.get(key, "")
    for key in PASS2_EMPTY_FIELDS:
        visible[key] = ""
    _assert_blinded_row(visible)
    _assert_clean_text(visible, context=f"第二轮盲化记录 {pass2_item_id}")
    return visible


def _assert_blinded_row(row: dict[str, Any]) -> None:
    keys = set(row)
    leaked = sorted(
        key
        for key in keys
        if key in BLINDED_FORBIDDEN_FIELDS
        or key.startswith("pass1_")
        or key.startswith("provisional_")
    )
    if leaked:
        raise ValueError(f"第二轮队列泄露第一轮或候选结论字段: {leaked}")


def _make_linkage_row(
    row: dict[str, Any],
    *,
    pass2_item_id: str,
    origin_pool: str,
) -> dict[str, Any]:
    return {
        "pass2_item_id": pass2_item_id,
        "candidate_id": row["candidate_id"],
        "origin_pool": origin_pool,
        "source_row_sha256": _row_sha256(row),
        "independence_unit_id": row.get("independence_unit_id", ""),
        "evidence_anchor_ids": row.get("evidence_anchor_ids", ""),
        "evidence_anchor_group_id": row.get("evidence_anchor_group_id", ""),
        "provisional_fact_cluster_id": row.get("provisional_fact_cluster_id", ""),
        "pass1_outcome": row.get("pass1_outcome", ""),
        "pass1_expected_decision": row.get("pass1_expected_decision", ""),
        "pass1_current_kb_support": row.get("pass1_current_kb_support", ""),
        "pass1_gold_evidence_status": row.get("pass1_gold_evidence_status", ""),
        "pass1_final_question": row.get("pass1_final_question", ""),
        "pass1_review_reason": row.get("pass1_review_reason", ""),
    }


def build_pass2_artifacts(
    original_rows: list[dict[str, Any]],
    supplement_rows: list[dict[str, Any]],
    scope_audit: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build a blinded second-pass queue while preserving an unblinded audit linkage."""
    _validate_config(config)
    _validate_unique_ids(original_rows, supplement_rows)
    for row in [*original_rows, *supplement_rows]:
        _assert_clean_text(row, context=f"第一轮记录 {row.get('candidate_id', '')}")

    original_promotable = [row for row in original_rows if _is_promotable(row)]
    supplement_promotable = [row for row in supplement_rows if _is_promotable(row)]
    expected_original = int(config["expected_original_promotable_before_scope"])
    expected_supplement = int(config["expected_supplement_promotable"])
    if len(original_promotable) != expected_original:
        raise ValueError(
            f"原池可晋升数量应为 {expected_original}，实际为 {len(original_promotable)}"
        )
    if len(supplement_promotable) != expected_supplement:
        raise ValueError(
            f"补充池可晋升数量应为 {expected_supplement}，实际为 {len(supplement_promotable)}"
        )

    flagged_ids = {
        _normalize(row.get("candidate_id"))
        for row in scope_audit.get("flagged_rows", [])
    }
    promotable_original_ids = {row["candidate_id"] for row in original_promotable}
    if not flagged_ids <= promotable_original_ids:
        unknown = sorted(flagged_ids - promotable_original_ids)
        raise ValueError(f"范围审计引用非原池可晋升记录: {unknown}")
    expected_exclusions = int(config["expected_scope_exclusions"])
    if len(flagged_ids) != expected_exclusions:
        raise ValueError(
            f"范围审计排除数量应为 {expected_exclusions}，实际为 {len(flagged_ids)}"
        )

    merged: list[tuple[str, dict[str, Any]]] = [
        ("original", row)
        for row in original_promotable
        if row["candidate_id"] not in flagged_ids
    ]
    merged.extend(("supplement", row) for row in supplement_promotable)
    expected_merged = int(config["expected_merged_count"])
    if len(merged) != expected_merged:
        raise ValueError(f"合并候选数量应为 {expected_merged}，实际为 {len(merged)}")

    seed = int(config["blind_seed"])
    shuffled = list(merged)
    random.Random(seed).shuffle(shuffled)
    review_queue: list[dict[str, Any]] = []
    linkage_records: list[dict[str, Any]] = []
    decisions: Counter[str] = Counter()
    units: set[str] = set()
    for order, (origin_pool, row) in enumerate(shuffled, start=1):
        candidate_id = str(row["candidate_id"])
        item_id = _blind_id(candidate_id, seed)
        review_queue.append(
            _make_visible_row(
                row,
                pass2_order=order,
                pass2_item_id=item_id,
                config=config,
            )
        )
        linkage_records.append(
            _make_linkage_row(
                row,
                pass2_item_id=item_id,
                origin_pool=origin_pool,
            )
        )
        decisions[str(row["pass1_expected_decision"])] += 1
        units.add(str(row.get("independence_unit_id", "")))

    if len({row["pass2_item_id"] for row in review_queue}) != len(review_queue):
        raise ValueError("第二轮 blind ID 冲突")
    target_distribution = Counter(
        {
            str(key): int(value)
            for key, value in config["target_final_decision_distribution"].items()
        }
    )
    shortages = {
        decision: target - decisions.get(decision, 0)
        for decision, target in target_distribution.items()
        if decisions.get(decision, 0) < target
    }
    if shortages:
        raise ValueError(f"目标决策分布不可满足: {shortages}")
    surplus = {
        decision: decisions.get(decision, 0) - target
        for decision, target in sorted(target_distribution.items())
    }
    summary = {
        "status": "pass2_blinded_queue_ready",
        "candidate_count": len(review_queue),
        "origin_distribution": {
            "original": sum(origin == "original" for origin, _ in merged),
            "supplement": sum(origin == "supplement" for origin, _ in merged),
        },
        "decision_distribution": dict(sorted(decisions.items())),
        "target_final_count": int(config["target_final_count"]),
        "target_final_decision_distribution": dict(sorted(target_distribution.items())),
        "target_distribution_feasible": True,
        "surplus_by_decision": surplus,
        "independence_unit_count": len(units),
        "scope_excluded_candidate_ids": sorted(flagged_ids),
        "blind_seed": seed,
        "usage": zero_usage(),
        "workflow_boundary": (
            "B3 second-pass author review queue only; not frozen and "
            "not expert-validated"
        ),
    }
    return {
        "review_queue": review_queue,
        "linkage_records": linkage_records,
        "summary": summary,
    }


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("不能写入空的第二轮队列")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    original_path = ROOT / config["original_pass1_queue"]
    supplement_path = ROOT / config["supplement_pass1_queue"]
    scope_path = ROOT / config["scope_audit"]
    output_queue = ROOT / config["pass2_review_queue_output"]
    output_linkage = ROOT / config["pass2_linkage_output"]
    output_report = ROOT / config["pass2_report_output"]

    artifacts = build_pass2_artifacts(
        _load_csv(original_path),
        _load_csv(supplement_path),
        json.loads(scope_path.read_text(encoding="utf-8")),
        config,
    )
    _write_csv(output_queue, artifacts["review_queue"])
    _write_json(
        output_linkage,
        {
            "linkage_version": config["config_version"],
            "dataset_version": config["dataset_version"],
            "records": artifacts["linkage_records"],
            "workflow_boundary": artifacts["summary"]["workflow_boundary"],
        },
    )
    report = {
        **artifacts["summary"],
        "config_version": config["config_version"],
        "dataset_version": config["dataset_version"],
        "kb_version": config["kb_version"],
        "input_files": {
            "original_pass1_queue": {
                "path": config["original_pass1_queue"],
                "sha256": _sha256_file(original_path),
            },
            "supplement_pass1_queue": {
                "path": config["supplement_pass1_queue"],
                "sha256": _sha256_file(supplement_path),
            },
            "scope_audit": {
                "path": config["scope_audit"],
                "sha256": _sha256_file(scope_path),
            },
        },
        "output_files": {
            "pass2_review_queue": {
                "path": config["pass2_review_queue_output"],
                "sha256": _sha256_file(output_queue),
            },
            "pass2_linkage": {
                "path": config["pass2_linkage_output"],
                "sha256": _sha256_file(output_linkage),
            },
        },
    }
    _write_json(output_report, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Benchmark-v1 blinded Pass 2 queue.")
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT
            / "experiments"
            / "phase7_formal_experiments"
            / "configs"
            / "benchmark_pass2_queue_v0_1.json"
        ),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
