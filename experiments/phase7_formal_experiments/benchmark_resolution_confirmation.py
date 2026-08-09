from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

RESOLUTION_FIELDS = (
    "resolution_reviewer_id",
    "resolution_annotator_role",
    "resolution_reviewed_at",
    "resolution_status",
    "resolution_final_decision",
    "resolution_final_kb_support",
    "resolution_final_gold_evidence_status",
    "resolution_reason",
)

DRAFT_AUTHOR_FIELDS = (
    "author_confirmed_decision",
    "author_confirmation_reason",
    "author_confirmed_at",
)

AUTHOR_FIELDS = (
    "author_reviewer_id",
    "author_annotator_role",
    "author_confirmation_status",
    "author_final_decision",
    "author_final_kb_support",
    "author_final_gold_evidence_status",
    "author_reason",
    "author_reviewed_at",
)

LINK_FIELDS = (
    "candidate_id",
    "source_row_sha256",
    "source_id",
    "source_title",
    "source_filename",
    "source_sha256",
    "page_number",
    "evidence_anchor_ids",
    "question",
    "anchor_text_span",
    "policy_rule_ids",
    "pass1_expected_decision",
    "pass1_current_kb_support",
    "pass1_review_reason",
    "pass2_expected_decision",
    "pass2_current_kb_support",
    "pass2_review_reason",
    "pass2_missing_evidence_type",
)

REVIEW_FIELDS = (
    "review_order",
    "review_version",
    "confidence_group",
    "draft_order",
    "draft_version",
    "decision_semantics_version",
    "dataset_version",
    "kb_version",
    *LINK_FIELDS,
    "assistant_system",
    "assistant_recommended_decision",
    "assistant_recommended_kb_support",
    "assistant_recommended_gold_evidence_status",
    "assistant_rationale",
    "assistant_confidence",
    "assistant_confidence_score",
    "draft_status",
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


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "config_version",
        "review_version",
        "resolved_version",
        "dataset_version",
        "kb_version",
        "expected_resolution_count",
        "expected_resolution_queue_sha256",
        "expected_assistant_draft_sha256",
        "allowed_decisions",
        "allowed_kb_support",
        "allowed_gold_evidence_status",
        "allowed_confidence",
        "required_confirmation_status",
        "required_annotator_role",
        "external_model_calls",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"confirmation config missing fields: {missing}")
    if int(config["external_model_calls"]) != 0:
        raise ValueError("confirmation workflow must not call external models")


def _validate_hashes(
    config: dict[str, Any],
    *,
    resolution_queue_sha256: str,
    assistant_draft_sha256: str,
) -> None:
    if resolution_queue_sha256 != config["expected_resolution_queue_sha256"]:
        raise ValueError("resolution queue hash mismatch")
    if assistant_draft_sha256 != config["expected_assistant_draft_sha256"]:
        raise ValueError("assistant draft hash mismatch")


def _index_unique(
    rows: list[dict[str, Any]], *, id_field: str, context: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = _normalize(row.get(id_field))
        if not item_id:
            raise ValueError(f"{context} has empty {id_field}")
        if item_id in indexed:
            raise ValueError(f"{context} has duplicate {id_field}: {item_id}")
        indexed[item_id] = row
    return indexed


def _validate_source_inputs(
    resolution_rows: list[dict[str, Any]],
    draft_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    expected_count = int(config["expected_resolution_count"])
    if len(resolution_rows) != expected_count or len(draft_rows) != expected_count:
        raise ValueError(
            "confirmation source count mismatch: "
            f"expected={expected_count}, queue={len(resolution_rows)}, draft={len(draft_rows)}"
        )

    queue_by_id = _index_unique(
        resolution_rows, id_field="candidate_id", context="resolution queue"
    )
    draft_by_id = _index_unique(
        draft_rows, id_field="candidate_id", context="assistant draft"
    )
    if set(queue_by_id) != set(draft_by_id):
        raise ValueError("candidate IDs differ between resolution queue and assistant draft")

    for candidate_id, queue_row in queue_by_id.items():
        draft_row = draft_by_id[candidate_id]
        if _normalize(queue_row.get("dataset_version")) != _normalize(
            config["dataset_version"]
        ) or _normalize(draft_row.get("dataset_version")) != _normalize(
            config["dataset_version"]
        ):
            raise ValueError(f"dataset version mismatch for {candidate_id}")
        if _normalize(queue_row.get("kb_version")) != _normalize(
            config["kb_version"]
        ) or _normalize(draft_row.get("kb_version")) != _normalize(
            config["kb_version"]
        ):
            raise ValueError(f"KB version mismatch for {candidate_id}")
        for field in RESOLUTION_FIELDS:
            if _normalize(queue_row.get(field)):
                raise ValueError(
                    f"formal resolution queue is already populated for {candidate_id}: {field}"
                )
        for field in DRAFT_AUTHOR_FIELDS:
            if _normalize(draft_row.get(field)):
                raise ValueError(
                    f"AI assistant draft must not contain author confirmation for {candidate_id}"
                )
        for field in LINK_FIELDS:
            if _normalize(queue_row.get(field)) != _normalize(draft_row.get(field)):
                raise ValueError(f"source linkage drift for {candidate_id}: {field}")
        if _normalize(draft_row.get("assistant_recommended_decision")) not in set(
            config["allowed_decisions"]
        ):
            raise ValueError(f"invalid assistant decision for {candidate_id}")
        if _normalize(draft_row.get("assistant_recommended_kb_support")) not in set(
            config["allowed_kb_support"]
        ):
            raise ValueError(f"invalid assistant KB support for {candidate_id}")
        if _normalize(
            draft_row.get("assistant_recommended_gold_evidence_status")
        ) not in set(config["allowed_gold_evidence_status"]):
            raise ValueError(f"invalid assistant evidence status for {candidate_id}")
        if _normalize(draft_row.get("assistant_confidence")) not in set(
            config["allowed_confidence"]
        ):
            raise ValueError(f"invalid assistant confidence for {candidate_id}")
    return draft_by_id


def build_author_review_pack(
    resolution_rows: list[dict[str, Any]],
    draft_rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    resolution_queue_sha256: str,
    assistant_draft_sha256: str,
) -> dict[str, Any]:
    """Build a deterministic review pack without asserting author confirmation."""
    _validate_config(config)
    _validate_hashes(
        config,
        resolution_queue_sha256=resolution_queue_sha256,
        assistant_draft_sha256=assistant_draft_sha256,
    )
    draft_by_id = _validate_source_inputs(resolution_rows, draft_rows, config)

    confidence_rank = {"high": 0, "medium": 1, "low": 2}
    ordered_drafts = sorted(
        draft_by_id.values(),
        key=lambda row: (
            confidence_rank.get(_normalize(row.get("assistant_confidence")), 99),
            int(row["draft_order"]),
        ),
    )
    review_rows: list[dict[str, Any]] = []
    confidence_counts: Counter[str] = Counter()
    for review_order, draft_row in enumerate(ordered_drafts, start=1):
        confidence = _normalize(draft_row.get("assistant_confidence"))
        confidence_counts[confidence] += 1
        row: dict[str, Any] = {
            "review_order": review_order,
            "review_version": config["review_version"],
            "confidence_group": confidence,
        }
        for field in REVIEW_FIELDS[3:]:
            row[field] = draft_row.get(field, "")
        for field in AUTHOR_FIELDS:
            row[field] = ""
        review_rows.append(row)

    summary = {
        "review_version": config["review_version"],
        "dataset_version": config["dataset_version"],
        "kb_version": config["kb_version"],
        "source_hashes": {
            "resolution_queue_sha256": resolution_queue_sha256,
            "assistant_draft_sha256": assistant_draft_sha256,
        },
        "record_count": len(review_rows),
        "confidence_distribution": dict(sorted(confidence_counts.items())),
        "review_records_sha256": _canonical_records_sha256(review_rows),
        "confirmed_count": 0,
        "pending_count": len(review_rows),
        "status": "pending_author_confirmation",
        "usage": zero_usage(),
    }
    return {"review_rows": review_rows, "summary": summary}


def _validate_timestamp(value: str, *, candidate_id: str) -> None:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid author_reviewed_at for {candidate_id}") from exc


def apply_author_confirmations(
    resolution_rows: list[dict[str, Any]],
    draft_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    resolution_queue_sha256: str,
    assistant_draft_sha256: str,
) -> dict[str, Any]:
    """Apply only complete, explicit author confirmations to a new artifact."""
    expected = build_author_review_pack(
        resolution_rows,
        draft_rows,
        config,
        resolution_queue_sha256=resolution_queue_sha256,
        assistant_draft_sha256=assistant_draft_sha256,
    )["review_rows"]
    expected_count = int(config["expected_resolution_count"])
    if len(review_rows) != expected_count:
        raise ValueError(f"all {expected_count} author confirmations are required")

    expected_by_id = _index_unique(
        expected, id_field="candidate_id", context="expected review pack"
    )
    review_by_id = _index_unique(
        review_rows, id_field="candidate_id", context="author review pack"
    )
    if set(expected_by_id) != set(review_by_id):
        raise ValueError("author review candidate IDs do not match the source pack")

    allowed_decisions = set(config["allowed_decisions"])
    allowed_support = set(config["allowed_kb_support"])
    allowed_evidence = set(config["allowed_gold_evidence_status"])
    required_status = _normalize(config["required_confirmation_status"])
    required_role = _normalize(config["required_annotator_role"])
    for candidate_id, row in review_by_id.items():
        expected_row = expected_by_id[candidate_id]
        for field in REVIEW_FIELDS:
            if _normalize(row.get(field)) != _normalize(expected_row.get(field)):
                raise ValueError(f"immutable review field drift for {candidate_id}: {field}")
        if any(not _normalize(row.get(field)) for field in AUTHOR_FIELDS):
            raise ValueError(f"all {expected_count} author confirmations are required")
        if _normalize(row.get("author_confirmation_status")) != required_status:
            raise ValueError(f"invalid author_confirmation_status for {candidate_id}")
        if _normalize(row.get("author_annotator_role")) != required_role:
            raise ValueError(f"invalid author_annotator_role for {candidate_id}")
        if _normalize(row.get("author_final_decision")) not in allowed_decisions:
            raise ValueError(f"invalid author_final_decision for {candidate_id}")
        if _normalize(row.get("author_final_kb_support")) not in allowed_support:
            raise ValueError(f"invalid author_final_kb_support for {candidate_id}")
        if _normalize(row.get("author_final_gold_evidence_status")) not in allowed_evidence:
            raise ValueError(
                f"invalid author_final_gold_evidence_status for {candidate_id}"
            )
        _validate_timestamp(
            _normalize(row.get("author_reviewed_at")), candidate_id=candidate_id
        )

    resolution_by_id = _index_unique(
        resolution_rows, id_field="candidate_id", context="resolution queue"
    )
    resolved_rows: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    for source_row in sorted(
        resolution_by_id.values(), key=lambda row: int(row["resolution_order"])
    ):
        candidate_id = _normalize(source_row.get("candidate_id"))
        review_row = review_by_id[candidate_id]
        resolved = dict(source_row)
        resolved["resolution_reviewer_id"] = review_row["author_reviewer_id"]
        resolved["resolution_annotator_role"] = review_row["author_annotator_role"]
        resolved["resolution_reviewed_at"] = review_row["author_reviewed_at"]
        resolved["resolution_status"] = "resolved"
        resolved["resolution_final_decision"] = review_row["author_final_decision"]
        resolved["resolution_final_kb_support"] = review_row[
            "author_final_kb_support"
        ]
        resolved["resolution_final_gold_evidence_status"] = review_row[
            "author_final_gold_evidence_status"
        ]
        resolved["resolution_reason"] = review_row["author_reason"]
        decision_counts[resolved["resolution_final_decision"]] += 1
        resolved_rows.append(resolved)

    summary = {
        "resolved_version": config["resolved_version"],
        "dataset_version": config["dataset_version"],
        "kb_version": config["kb_version"],
        "source_hashes": {
            "resolution_queue_sha256": resolution_queue_sha256,
            "assistant_draft_sha256": assistant_draft_sha256,
        },
        "resolved_count": len(resolved_rows),
        "decision_distribution": dict(sorted(decision_counts.items())),
        "resolved_records_sha256": _canonical_records_sha256(resolved_rows),
        "status": "author_resolution_complete",
        "usage": zero_usage(),
    }
    return {"resolved_rows": resolved_rows, "summary": summary}


def render_review_guide(
    review_rows: list[dict[str, Any]], summary: dict[str, Any]
) -> str:
    lines = [
        "# Benchmark-v1 分歧样本作者确认指南",
        "",
        "> 本文件仅用于作者确认。AI 建议不是最终标签，作者必须结合问题、证据片段和两轮理由独立判断。",
        "",
        "## 使用规则",
        "",
        "1. 先审核高置信组，再审核中置信组。",
        "2. 每条必须填写审核者 ID、角色、确认状态、最终决策、KB 支持状态、证据状态、理由和时间。",
        "3. 34 条未全部明确确认前，脚本会拒绝生成正式 resolved 文件。",
        "4. 不得为了满足目标分布而修改证据标签。",
        "",
        f"- 待确认：{summary['pending_count']} 条",
        f"- 高置信：{summary['confidence_distribution'].get('high', 0)} 条",
        f"- 中置信：{summary['confidence_distribution'].get('medium', 0)} 条",
        "",
    ]
    current_group = ""
    for row in review_rows:
        group = _normalize(row.get("confidence_group"))
        if group != current_group:
            current_group = group
            group_name = "高置信建议" if group == "high" else "中置信建议"
            lines.extend([f"## {group_name}", ""])
        lines.extend(
            [
                f"### {row['review_order']}. {row['candidate_id']}",
                "",
                f"- 问题：{row['question']}",
                f"- 来源：{row['source_title']}，第 {row['page_number']} 页",
                f"- 证据片段：{row['anchor_text_span']}",
                f"- Pass 1：{row['pass1_expected_decision']} / {row['pass1_current_kb_support']}",
                f"- Pass 2：{row['pass2_expected_decision']} / {row['pass2_current_kb_support']}",
                f"- AI 建议：{row['assistant_recommended_decision']} / {row['assistant_recommended_kb_support']}",
                f"- AI 理由：{row['assistant_rationale']}",
                "- 作者结论：待填写 CSV 中的 `author_*` 字段",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if path.exists() and path.read_bytes() != temp_path.read_bytes():
        temp_path.unlink()
        raise ValueError(f"refusing to overwrite changed artifact: {path}")
    temp_path.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = text.encode("utf-8")
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"refusing to overwrite changed artifact: {path}")
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_bytes(payload)
    temp_path.replace(path)


def run_prepare_review(config_path: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    input_paths = config["input_paths"]
    output_paths = config["output_paths"]
    queue_path = ROOT / input_paths["resolution_queue"]
    draft_path = ROOT / input_paths["assistant_draft"]
    review_path = ROOT / output_paths["author_review"]
    guide_path = ROOT / output_paths["author_review_guide"]
    queue_hash = _sha256_file(queue_path)
    draft_hash = _sha256_file(draft_path)
    artifacts = build_author_review_pack(
        _read_csv(queue_path),
        _read_csv(draft_path),
        config,
        resolution_queue_sha256=queue_hash,
        assistant_draft_sha256=draft_hash,
    )
    _write_csv_atomic(review_path, artifacts["review_rows"])
    _write_text_atomic(
        guide_path,
        render_review_guide(artifacts["review_rows"], artifacts["summary"]),
    )
    summary = dict(artifacts["summary"])
    summary["output_hashes"] = {
        "author_review_sha256": _sha256_file(review_path),
        "author_review_guide_sha256": _sha256_file(guide_path),
    }
    return summary


def run_apply_confirmations(config_path: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    input_paths = config["input_paths"]
    output_paths = config["output_paths"]
    queue_path = ROOT / input_paths["resolution_queue"]
    draft_path = ROOT / input_paths["assistant_draft"]
    review_path = ROOT / output_paths["author_review"]
    resolved_path = ROOT / output_paths["resolved_queue"]
    summary_path = ROOT / output_paths["resolved_summary"]
    artifacts = apply_author_confirmations(
        _read_csv(queue_path),
        _read_csv(draft_path),
        _read_csv(review_path),
        config,
        resolution_queue_sha256=_sha256_file(queue_path),
        assistant_draft_sha256=_sha256_file(draft_path),
    )
    _write_csv_atomic(resolved_path, artifacts["resolved_rows"])
    summary = dict(artifacts["summary"])
    summary["output_hashes"] = {
        "resolved_queue_sha256": _sha256_file(resolved_path)
    }
    _write_text_atomic(
        summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare or apply explicit author confirmations for Benchmark-v1."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "experiments"
        / "phase7_formal_experiments"
        / "configs"
        / "benchmark_resolution_confirmation_v0_1.json",
    )
    parser.add_argument("--mode", choices=("prepare", "apply"), default="prepare")
    args = parser.parse_args()
    summary = (
        run_prepare_review(args.config)
        if args.mode == "prepare"
        else run_apply_confirmations(args.config)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
