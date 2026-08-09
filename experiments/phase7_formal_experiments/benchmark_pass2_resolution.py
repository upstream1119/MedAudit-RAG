from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

CORE_COMPARISON_FIELDS = (
    ("expected_decision", "pass1_expected_decision", "pass2_expected_decision"),
    ("current_kb_support", "pass1_current_kb_support", "pass2_current_kb_support"),
    (
        "gold_evidence_status",
        "pass1_gold_evidence_status",
        "pass2_gold_evidence_status",
    ),
)

LINKAGE_FIELDS = (
    "candidate_id",
    "origin_pool",
    "source_row_sha256",
    "independence_unit_id",
    "evidence_anchor_ids",
    "evidence_anchor_group_id",
    "provisional_fact_cluster_id",
    "pass1_outcome",
    "pass1_expected_decision",
    "pass1_current_kb_support",
    "pass1_gold_evidence_status",
    "pass1_final_question",
    "pass1_review_reason",
)

PASS2_SOURCE_FIELDS = (
    "pass2_order",
    "pass2_item_id",
    "question",
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

PASS2_REVIEW_FIELDS = (
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

RESOLUTION_EMPTY_FIELDS = (
    "resolution_reviewer_id",
    "resolution_annotator_role",
    "resolution_reviewed_at",
    "resolution_status",
    "resolution_final_decision",
    "resolution_final_kb_support",
    "resolution_final_gold_evidence_status",
    "resolution_reason",
)


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


def _canonical_records_sha256(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _assert_clean_text(row: dict[str, Any], *, context: str) -> None:
    for key, value in row.items():
        text = str(value or "")
        if "\ufffd" in text or "???" in text:
            raise ValueError(f"{context} 存在乱码字段 {key}")


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "config_version",
        "resolution_version",
        "dataset_version",
        "kb_version",
        "expected_candidate_count",
        "expected_disagreement_count",
        "expected_excluded_count",
        "target_final_count",
        "target_final_decision_distribution",
        "allowed_decisions",
        "allowed_kb_support",
        "allowed_gold_evidence_status",
        "external_model_calls",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"裁决配置缺少字段: {missing}")
    if int(config["external_model_calls"]) != 0:
        raise ValueError("裁决队列生成不允许外部模型调用")
    target_distribution = config["target_final_decision_distribution"]
    if not isinstance(target_distribution, dict) or not target_distribution:
        raise ValueError("最终目标决策分布必须是非空对象")
    target_total = sum(int(value) for value in target_distribution.values())
    if target_total != int(config["target_final_count"]):
        raise ValueError("最终目标数量与目标决策分布合计不一致")


def _validate_progress(
    progress: dict[str, Any],
    *,
    candidate_count: int,
    pass2_queue_sha256: str,
) -> None:
    if (
        _normalize(progress.get("status")) != "complete"
        or int(progress.get("pending_count", -1)) != 0
        or int(progress.get("completed_count", -1)) != candidate_count
    ):
        raise ValueError("第二轮尚未完整完成，禁止生成裁决队列")
    if int(progress.get("candidate_count", -1)) != candidate_count:
        raise ValueError("第二轮进度中的候选数量与队列不一致")
    if _normalize(progress.get("queue_sha256")) != pass2_queue_sha256:
        raise ValueError("第二轮进度记录的队列哈希与输入文件不一致")


def _index_unique(
    rows: list[dict[str, Any]],
    *,
    id_field: str,
    context: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = _normalize(row.get(id_field))
        if not item_id:
            raise ValueError(f"{context} 的 {id_field} 不能为空")
        if item_id in indexed:
            raise ValueError(f"{context} 存在重复 {id_field}: {item_id}")
        indexed[item_id] = row
    return indexed


def _validate_annotation_values(
    linkage_row: dict[str, Any],
    pass2_row: dict[str, Any],
    config: dict[str, Any],
) -> None:
    item_id = _normalize(pass2_row.get("pass2_item_id"))
    if _normalize(pass2_row.get("pass2_outcome")) not in {"accepted", "reject"}:
        raise ValueError(f"第二轮记录尚未形成可裁决终态: {item_id}")
    for prefix, row, decision_key, support_key, evidence_key in (
        (
            "第一轮",
            linkage_row,
            "pass1_expected_decision",
            "pass1_current_kb_support",
            "pass1_gold_evidence_status",
        ),
        (
            "第二轮",
            pass2_row,
            "pass2_expected_decision",
            "pass2_current_kb_support",
            "pass2_gold_evidence_status",
        ),
    ):
        if _normalize(row.get(decision_key)) not in set(config["allowed_decisions"]):
            raise ValueError(f"{prefix}决策值非法: {item_id}")
        if _normalize(row.get(support_key)) not in set(config["allowed_kb_support"]):
            raise ValueError(f"{prefix}知识库支持状态非法: {item_id}")
        if _normalize(row.get(evidence_key)) not in set(
            config["allowed_gold_evidence_status"]
        ):
            raise ValueError(f"{prefix}证据状态非法: {item_id}")


def _make_resolution_row(
    linkage_row: dict[str, Any],
    pass2_row: dict[str, Any],
    *,
    resolution_order: int,
    disagreement_fields: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "resolution_order": resolution_order,
        "resolution_version": config["resolution_version"],
        "dataset_version": config["dataset_version"],
        "kb_version": config["kb_version"],
    }
    for field in LINKAGE_FIELDS:
        row[field] = linkage_row.get(field, "")
    for field in PASS2_SOURCE_FIELDS:
        row[field] = pass2_row.get(field, "")
    for field in PASS2_REVIEW_FIELDS:
        row[field] = pass2_row.get(field, "")
    row["disagreement_fields"] = json.dumps(
        disagreement_fields, ensure_ascii=False
    )
    for field in RESOLUTION_EMPTY_FIELDS:
        row[field] = ""
    _assert_clean_text(row, context=f"裁决记录 {row.get('pass2_item_id', '')}")
    return row


def build_resolution_artifacts(
    linkage: dict[str, Any],
    pass2_rows: list[dict[str, Any]],
    progress: dict[str, Any],
    config: dict[str, Any],
    *,
    linkage_sha256: str,
    pass2_queue_sha256: str,
    progress_sha256: str,
) -> dict[str, Any]:
    """Link two completed author passes and emit unresolved core disagreements."""
    _validate_config(config)
    expected_count = int(config["expected_candidate_count"])
    if len(pass2_rows) != expected_count:
        raise ValueError(
            f"第二轮候选数量应为 {expected_count}，实际为 {len(pass2_rows)}"
        )
    if _normalize(linkage.get("dataset_version")) != _normalize(
        config["dataset_version"]
    ):
        raise ValueError("关联表与配置的数据集版本不一致")
    _validate_progress(
        progress,
        candidate_count=expected_count,
        pass2_queue_sha256=pass2_queue_sha256,
    )

    linkage_records = list(linkage.get("records", []))
    linkage_by_id = _index_unique(
        linkage_records,
        id_field="pass2_item_id",
        context="第二轮关联表",
    )
    pass2_by_id = _index_unique(
        pass2_rows,
        id_field="pass2_item_id",
        context="第二轮队列",
    )
    if set(linkage_by_id) != set(pass2_by_id):
        missing_linkage = sorted(set(pass2_by_id) - set(linkage_by_id))
        missing_queue = sorted(set(linkage_by_id) - set(pass2_by_id))
        raise ValueError(
            "第二轮关联映射不完整: "
            f"missing_linkage={missing_linkage}, missing_queue={missing_queue}"
        )

    resolution_queue: list[dict[str, Any]] = []
    excluded_records: list[dict[str, str]] = []
    agreement_count = 0
    transition_counter: Counter[str] = Counter()
    pass1_decisions: Counter[str] = Counter()
    pass2_decisions: Counter[str] = Counter()
    origin_disagreements: Counter[str] = Counter()

    ordered_rows = sorted(pass2_rows, key=lambda row: int(row["pass2_order"]))
    for pass2_row in ordered_rows:
        item_id = _normalize(pass2_row.get("pass2_item_id"))
        linkage_row = linkage_by_id[item_id]
        _assert_clean_text(linkage_row, context=f"关联记录 {item_id}")
        _assert_clean_text(pass2_row, context=f"第二轮记录 {item_id}")
        _validate_annotation_values(linkage_row, pass2_row, config)

        pass1_question = _normalize(linkage_row.get("pass1_final_question"))
        visible_question = _normalize(pass2_row.get("question"))
        pass2_question = _normalize(pass2_row.get("pass2_final_question"))
        if not pass1_question or not visible_question or not pass2_question:
            raise ValueError(f"问题文本不能为空: {item_id}")
        if len({pass1_question, visible_question, pass2_question}) != 1:
            raise ValueError(f"第一、二轮问题文本漂移: {item_id}")

        pass1_decision = _normalize(linkage_row.get("pass1_expected_decision"))
        pass2_decision = _normalize(pass2_row.get("pass2_expected_decision"))
        if _normalize(pass2_row.get("pass2_outcome")) == "reject":
            excluded_records.append(
                {
                    "pass2_item_id": item_id,
                    "candidate_id": _normalize(linkage_row.get("candidate_id")),
                    "pass2_expected_decision": pass2_decision,
                    "pass2_review_reason": _normalize(
                        pass2_row.get("pass2_review_reason")
                    ),
                }
            )
            continue

        pass1_decisions[pass1_decision] += 1
        pass2_decisions[pass2_decision] += 1
        transition_counter[f"{pass1_decision}->{pass2_decision}"] += 1
        disagreement_fields = [
            label
            for label, pass1_field, pass2_field in CORE_COMPARISON_FIELDS
            if _normalize(linkage_row.get(pass1_field))
            != _normalize(pass2_row.get(pass2_field))
        ]
        if not disagreement_fields:
            agreement_count += 1
            continue

        origin_disagreements[_normalize(linkage_row.get("origin_pool"))] += 1
        resolution_queue.append(
            _make_resolution_row(
                linkage_row,
                pass2_row,
                resolution_order=len(resolution_queue) + 1,
                disagreement_fields=disagreement_fields,
                config=config,
            )
        )

    expected_disagreements = int(config["expected_disagreement_count"])
    if len(resolution_queue) != expected_disagreements:
        raise ValueError(
            "第一、二轮核心争议数量不符合固定配置: "
            f"expected={expected_disagreements}, actual={len(resolution_queue)}"
        )
    expected_excluded = int(config["expected_excluded_count"])
    if len(excluded_records) != expected_excluded:
        raise ValueError(
            "第二轮排除数量不符合固定配置: "
            f"expected={expected_excluded}, actual={len(excluded_records)}"
        )
    actual_promotable = len(pass2_rows) - len(excluded_records)
    if int(progress.get("promotable_count", -1)) != actual_promotable:
        raise ValueError(
            "第二轮进度中的可晋升数量与实际 outcome 不一致: "
            f"progress={progress.get('promotable_count')}, actual={actual_promotable}"
        )

    target_distribution = {
        key: int(value)
        for key, value in config["target_final_decision_distribution"].items()
    }
    pass2_distribution = {
        key: int(pass2_decisions.get(key, 0)) for key in target_distribution
    }
    target_gap = {
        key: pass2_distribution[key] - target_distribution[key]
        for key in target_distribution
    }
    summary: dict[str, Any] = {
        "resolution_version": config["resolution_version"],
        "dataset_version": config["dataset_version"],
        "kb_version": config["kb_version"],
        "input_hashes": {
            "linkage_sha256": linkage_sha256,
            "pass2_queue_sha256": pass2_queue_sha256,
            "progress_sha256": progress_sha256,
        },
        "linked_count": len(pass2_rows),
        "promotable_count": actual_promotable,
        "core_agreement_count": agreement_count,
        "resolution_candidate_count": len(resolution_queue),
        "excluded_count": len(excluded_records),
        "excluded_records": excluded_records,
        "pass1_decision_distribution": dict(sorted(pass1_decisions.items())),
        "pass2_decision_distribution": dict(sorted(pass2_decisions.items())),
        "decision_transition_distribution": dict(sorted(transition_counter.items())),
        "disagreement_origin_distribution": dict(
            sorted(origin_disagreements.items())
        ),
        "target_final_count": int(config["target_final_count"]),
        "target_final_decision_distribution": target_distribution,
        "raw_pass2_minus_target": target_gap,
        "resolution_records_sha256": _canonical_records_sha256(resolution_queue),
        "status": "resolution_pending",
        "usage": zero_usage(),
        "workflow_boundary": (
            "本文件只关联同一作者的两轮独立复核并暴露核心争议，不自动裁决，"
            "不构成专家盲评、临床验证或最终 benchmark 冻结。"
        ),
    }
    return {"resolution_queue": resolution_queue, "summary": summary}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("裁决队列为空，拒绝写出空 CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def run_prepare_resolution(config_path: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    input_paths = config.get("input_paths", {})
    output_paths = config.get("output_paths", {})
    linkage_path = ROOT / input_paths["linkage"]
    pass2_queue_path = ROOT / input_paths["pass2_queue"]
    progress_path = ROOT / input_paths["progress"]
    resolution_queue_path = ROOT / output_paths["resolution_queue"]
    summary_path = ROOT / output_paths["summary"]

    artifacts = build_resolution_artifacts(
        _read_json(linkage_path),
        _read_csv(pass2_queue_path),
        _read_json(progress_path),
        config,
        linkage_sha256=_sha256_file(linkage_path),
        pass2_queue_sha256=_sha256_file(pass2_queue_path),
        progress_sha256=_sha256_file(progress_path),
    )
    _write_csv_atomic(resolution_queue_path, artifacts["resolution_queue"])
    summary = dict(artifacts["summary"])
    summary["output_hashes"] = {
        "resolution_queue_sha256": _sha256_file(resolution_queue_path)
    }
    _write_json_atomic(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成第一、二轮核心争议的人工裁决队列。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "experiments"
        / "phase7_formal_experiments"
        / "configs"
        / "benchmark_pass2_resolution_v0_1.json",
    )
    args = parser.parse_args()
    summary = run_prepare_resolution(args.config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
