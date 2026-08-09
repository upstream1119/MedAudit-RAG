from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


PASS2_OUTPUT_FIELDS = [
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
]

LIST_OUTPUT_FIELDS = {
    "pass2_required_evidence_type",
    "pass2_required_claims",
    "pass2_allowed_claims",
    "pass2_forbidden_claims",
    "pass2_missing_evidence_type",
    "pass2_missing_information",
    "pass2_risk_labels",
    "pass2_issues_found",
}

DECISION_SUPPORT = {
    "answer": "supported_by_current_kb",
    "review_required": "partial_current_kb_support",
    "insufficient_evidence": "not_supported_by_current_kb",
    "boundary_refusal": "policy_rule",
}


def _contains_garbled_text(value: Any) -> bool:
    if isinstance(value, str):
        return "\ufffd" in value or "???" in value
    if isinstance(value, list):
        return any(_contains_garbled_text(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_garbled_text(item) for item in value.values())
    return False


def _parse_json_list(value: Any, field_name: str) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} 必须是 JSON 数组") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{field_name} 必须是 JSON 数组")
    return parsed


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "annotation_version",
        "dataset_version",
        "kb_version",
        "expected_candidate_count",
        "annotator_role",
        "allowed_outcomes",
        "allowed_decisions",
        "allowed_kb_support",
        "allowed_gold_evidence_status",
        "policy_rule_id",
        "external_model_calls",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"第二轮配置缺少字段: {', '.join(missing)}")
    if config["external_model_calls"] != 0:
        raise ValueError("第二轮 benchmark 标注禁止调用外部模型")


def _validate_batch_metadata(batch: dict[str, Any], config: dict[str, Any]) -> None:
    for field in (
        "annotation_version",
        "batch_id",
        "batch_scope",
        "dataset_version",
        "kb_version",
        "record_count",
        "review_mode",
        "external_model_calls",
        "input_tokens",
        "output_tokens",
        "estimated_cost",
        "records",
    ):
        if field not in batch:
            raise ValueError(f"第二轮批次缺少字段: {field}")
    if _contains_garbled_text(batch):
        raise ValueError("第二轮批次包含乱码")
    if batch["annotation_version"] != config["annotation_version"]:
        raise ValueError("第二轮标注版本不一致")
    if batch["dataset_version"] != config["dataset_version"]:
        raise ValueError("第二轮数据集版本不一致")
    if batch["kb_version"] != config["kb_version"]:
        raise ValueError("第二轮知识库版本不一致")
    if not isinstance(batch["records"], list):
        raise ValueError("第二轮 records 必须是数组")
    if batch["record_count"] != len(batch["records"]):
        raise ValueError("第二轮 record_count 与实际记录数不一致")
    usage = (
        batch["external_model_calls"],
        batch["input_tokens"],
        batch["output_tokens"],
        batch["estimated_cost"],
    )
    if any(value != 0 for value in usage):
        raise ValueError("第二轮人工核验批次必须保持零模型调用、零 token、零费用")


def _validate_queue(queue_rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    if len(queue_rows) != int(config["expected_candidate_count"]):
        raise ValueError("第二轮候选数量与配置不一致")
    blind_ids = [str(row.get("pass2_item_id", "")).strip() for row in queue_rows]
    if not all(blind_ids) or len(blind_ids) != len(set(blind_ids)):
        raise ValueError("第二轮队列存在空或重复 blind ID")
    orders = [int(row.get("pass2_order", 0)) for row in queue_rows]
    if orders != list(range(1, len(queue_rows) + 1)):
        raise ValueError("第二轮队列顺序必须从 1 连续递增")


def _validate_record(
    record: dict[str, Any],
    queue_row: dict[str, Any],
    config: dict[str, Any],
) -> None:
    missing = sorted(
        field
        for field in PASS2_OUTPUT_FIELDS
        if field not in record
    )
    if missing:
        raise ValueError(f"第二轮记录缺少字段: {', '.join(missing)}")
    if _contains_garbled_text(record):
        raise ValueError("第二轮记录包含乱码")

    outcome = record["pass2_outcome"]
    decision = record["pass2_expected_decision"]
    support = record["pass2_current_kb_support"]
    if outcome not in config["allowed_outcomes"]:
        raise ValueError(f"不允许的第二轮结论: {outcome}")
    if decision not in config["allowed_decisions"]:
        raise ValueError(f"不允许的预期决策: {decision}")
    if support not in config["allowed_kb_support"]:
        raise ValueError(f"不允许的知识库支持状态: {support}")
    if support != DECISION_SUPPORT[decision]:
        raise ValueError("预期决策与知识库支持状态不匹配")
    if record["pass2_gold_evidence_status"] not in config["allowed_gold_evidence_status"]:
        raise ValueError("不允许的 gold evidence 状态")
    if record["pass2_annotator_role"] != config["annotator_role"]:
        raise ValueError("标注者角色与配置不一致")
    for field in (
        "pass2_reviewer_id",
        "pass2_reviewed_at",
        "pass2_final_question",
        "pass2_review_reason",
    ):
        if not str(record[field]).strip():
            raise ValueError(f"第二轮记录字段不能为空: {field}")
    for field in LIST_OUTPUT_FIELDS:
        if not isinstance(record[field], list):
            raise ValueError(f"第二轮记录字段必须是数组: {field}")

    original_question = str(queue_row["question"]).strip()
    final_question = str(record["pass2_final_question"]).strip()
    if outcome == "revise" and final_question == original_question:
        raise ValueError("修改题必须提供不同的问题")
    if outcome == "accepted" and final_question != original_question:
        raise ValueError("accepted 样本不得静默改写问题")

    if decision == "boundary_refusal":
        policy_ids = _parse_json_list(queue_row.get("policy_rule_ids", ""), "policy_rule_ids")
        if config["policy_rule_id"] not in policy_ids:
            raise ValueError("安全政策未绑定，不能标为 boundary_refusal")
    elif record["pass2_gold_evidence_status"] == "page_span_located":
        for field in ("source_title", "page_number", "anchor_text_span"):
            if not str(queue_row.get(field, "")).strip():
                raise ValueError(f"page_span_located 缺少可定位字段: {field}")


def _build_summary(
    rows: list[dict[str, Any]],
    batch: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    completed = [row for row in rows if str(row.get("pass2_outcome", "")).strip()]
    promotable = [row for row in completed if row["pass2_outcome"] == "accepted"]
    revision_required = [row for row in completed if row["pass2_outcome"] == "revise"]
    return {
        "annotation_version": config["annotation_version"],
        "dataset_version": config["dataset_version"],
        "kb_version": config["kb_version"],
        "last_batch_id": batch["batch_id"],
        "candidate_count": len(rows),
        "completed_count": len(completed),
        "pending_count": len(rows) - len(completed),
        "promotable_count": len(promotable),
        "revision_required_count": len(revision_required),
        "outcome_distribution": dict(sorted(Counter(row["pass2_outcome"] for row in completed).items())),
        "decision_distribution": dict(sorted(Counter(row["pass2_expected_decision"] for row in completed).items())),
        "status": "complete" if len(completed) == len(rows) else "in_progress",
        "usage": {
            "external_model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost": 0,
        },
    }


def apply_pass2_batch(
    queue_rows: list[dict[str, Any]],
    batch: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_config(config)
    _validate_batch_metadata(batch, config)
    _validate_queue(queue_rows, config)

    records = batch["records"]
    record_ids = [str(record.get("pass2_item_id", "")).strip() for record in records]
    if not all(record_ids) or len(record_ids) != len(set(record_ids)):
        raise ValueError("第二轮批次存在空或重复 blind ID")

    by_id = {row["pass2_item_id"]: row for row in queue_rows}
    unknown = sorted(set(record_ids) - set(by_id))
    if unknown:
        raise ValueError(f"第二轮批次包含未知 blind ID: {', '.join(unknown)}")

    pending_ids = [
        row["pass2_item_id"]
        for row in queue_rows
        if not str(row.get("pass2_outcome", "")).strip()
    ]
    if record_ids != pending_ids[: len(record_ids)]:
        raise ValueError("第二轮批次必须按待处理连续顺序提交")

    updated = deepcopy(queue_rows)
    updated_by_id = {row["pass2_item_id"]: row for row in updated}
    for record in records:
        queue_row = by_id[record["pass2_item_id"]]
        if str(queue_row.get("pass2_outcome", "")).strip():
            raise ValueError("第二轮批次不得覆盖已完成记录")
        _validate_record(record, queue_row, config)
        target = updated_by_id[record["pass2_item_id"]]
        for field in PASS2_OUTPUT_FIELDS:
            value = record[field]
            target[field] = json.dumps(value, ensure_ascii=False) if field in LIST_OUTPUT_FIELDS else value

    return updated, _build_summary(updated, batch, config)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return payload


def load_queue(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def _atomic_write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8-sig",
        newline="",
        delete=False,
        dir=output_path.parent,
        suffix=".tmp",
    ) as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(file_obj.name)
    os.replace(temp_path, output_path)


def _atomic_write_json(payload: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        delete=False,
        dir=output_path.parent,
        suffix=".tmp",
    ) as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
        temp_path = Path(file_obj.name)
    os.replace(temp_path, output_path)


def run_apply_pass2_batch(
    config_path: str | Path,
    queue_path: str | Path,
    batch_path: str | Path,
    progress_path: str | Path,
) -> dict[str, Any]:
    config = load_json(config_path)
    batch = load_json(batch_path)
    queue_sha = sha256_file(queue_path)
    expected_sha = str(batch.get("parent_queue_sha256", "")).strip()
    if not expected_sha or expected_sha != queue_sha:
        raise ValueError("第二轮批次 parent_queue_sha256 与当前队列不一致")
    rows = load_queue(queue_path)
    updated, summary = apply_pass2_batch(rows, batch, config)
    _atomic_write_csv(updated, queue_path)
    summary["parent_queue_sha256"] = queue_sha
    summary["queue_sha256"] = sha256_file(queue_path)
    summary["batch_path"] = str(Path(batch_path).as_posix())
    _atomic_write_json(summary, progress_path)
    return summary

