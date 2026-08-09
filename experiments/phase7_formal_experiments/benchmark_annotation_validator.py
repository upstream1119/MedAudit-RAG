"""Prepare and validate the first author-verification pass for Benchmark-v1."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PASS1_QUEUE_FIELDS = [
    "annotation_order",
    "annotation_version",
    "candidate_id",
    "independence_unit_id",
    "question",
    "candidate_role",
    "candidate_status",
    "dataset_version",
    "kb_version",
    "schema_version",
    "protocol_version",
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
    "evidence_anchor_ids",
    "evidence_anchor_group_id",
    "provisional_fact_cluster_id",
    "provisional_expected_decision",
    "provisional_scenario_type",
    "provisional_risk_labels",
    "current_kb_support",
    "missing_evidence_type",
    "policy_rule_ids",
    "overlap_decision",
    "dev50_overlap_status",
    "internal_overlap_status",
    "pass1_reviewer_id",
    "pass1_annotator_role",
    "pass1_reviewed_at",
    "pass1_outcome",
    "pass1_final_question",
    "pass1_expected_decision",
    "pass1_current_kb_support",
    "pass1_gold_evidence_status",
    "pass1_required_evidence_type",
    "pass1_required_claims",
    "pass1_allowed_claims",
    "pass1_forbidden_claims",
    "pass1_missing_evidence_type",
    "pass1_missing_information",
    "pass1_risk_labels",
    "pass1_issues_found",
    "pass1_review_reason",
    "pass1_overlap_reaudit_status",
    "pass1_overlap_reaudit_report_path",
    "pass1_overlap_reaudit_report_sha256",
    "pass1_overlap_reaudited_at",
]

JSON_LIST_FIELDS = {
    "source_can_support",
    "source_cannot_support",
    "supported_claim_types",
    "evidence_anchor_ids",
    "provisional_risk_labels",
    "missing_evidence_type",
    "policy_rule_ids",
    "pass1_required_evidence_type",
    "pass1_required_claims",
    "pass1_allowed_claims",
    "pass1_forbidden_claims",
    "pass1_missing_evidence_type",
    "pass1_missing_information",
    "pass1_risk_labels",
    "pass1_issues_found",
}

INTEGER_FIELDS = {"annotation_order", "source_year", "page_number"}

REQUIRED_CONFIG_FIELDS = {
    "config_version",
    "annotation_version",
    "input_dataset_version",
    "output_dataset_version",
    "schema_version",
    "protocol_version",
    "kb_version",
    "expected_candidate_count",
    "expected_independence_unit_count",
    "pass1_shuffle_seed",
    "annotator_role",
    "allowed_outcomes",
    "allowed_decisions",
    "allowed_kb_support",
    "allowed_gold_evidence_status",
    "policy_rule_id",
    "fail_closed",
    "external_model_calls",
}

PASS1_REQUIRED_SCALAR_FIELDS = {
    "pass1_reviewer_id",
    "pass1_annotator_role",
    "pass1_reviewed_at",
    "pass1_outcome",
    "pass1_final_question",
    "pass1_expected_decision",
    "pass1_current_kb_support",
    "pass1_gold_evidence_status",
    "pass1_review_reason",
}

PASS1_REVIEW_FIELDS = tuple(
    field for field in PASS1_QUEUE_FIELDS if field.startswith("pass1_")
)
PASS1_REAUDIT_FIELDS = (
    "pass1_overlap_reaudit_status",
    "pass1_overlap_reaudit_report_path",
    "pass1_overlap_reaudit_report_sha256",
    "pass1_overlap_reaudited_at",
)
PASS1_REAUDIT_VERSION = "benchmark-revision-overlap-reaudit-v0.1"
PASS1_CORRECTION_FIELDS = {"pass1_review_reason"}
GARBLED_PASS1_METADATA_RE = re.compile(r"\?{2,}|�")


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _compute_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return payload


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL 第 {line_number} 行必须是对象: {path}")
        rows.append(row)
    return rows


def load_annotation_config(path: str | Path) -> dict[str, Any]:
    config = load_json(path)
    missing = sorted(REQUIRED_CONFIG_FIELDS - set(config))
    if missing:
        raise ValueError(f"标注配置缺少字段: {', '.join(missing)}")
    if config["annotator_role"] != "author":
        raise ValueError("annotator_role 只能为 author")
    if int(config["external_model_calls"]) != 0:
        raise ValueError("B3.1 不允许外部模型调用")
    if not config["fail_closed"]:
        raise ValueError("B3.1 必须启用 fail_closed")
    return config


def _manifest_source_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise ValueError("source manifest 缺少 sources 数组")
    source_map: dict[str, dict[str, Any]] = {}
    for source in sources:
        source_id = str(source.get("source_id", ""))
        if not source_id:
            raise ValueError("source manifest 存在空 source_id")
        if source_id in source_map:
            raise ValueError(f"source manifest 存在重复 source_id: {source_id}")
        source_map[source_id] = source
    return source_map


def _anchor_map(anchors: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for anchor in anchors:
        anchor_id = str(anchor.get("anchor_id", ""))
        if not anchor_id:
            raise ValueError("锚点存在空 anchor_id")
        if anchor_id in result:
            raise ValueError(f"锚点 ID 重复: {anchor_id}")
        if anchor.get("verification_status") != "author_verified_anchor":
            raise ValueError(f"锚点未经作者核验: {anchor_id}")
        result[anchor_id] = anchor
    return result


def _require_equal(
    candidate_id: str,
    label: str,
    left: Any,
    right: Any,
) -> None:
    if isinstance(left, list) or isinstance(right, list):
        equal = list(left or []) == list(right or [])
    else:
        equal = _normalize_text(left) == _normalize_text(right)
    if not equal:
        raise ValueError(f"{candidate_id} 的{label}不一致")


def _validate_candidate_provenance(
    candidate: dict[str, Any],
    anchors_by_id: dict[str, dict[str, Any]],
    sources_by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_id = str(candidate.get("candidate_id", ""))
    if candidate.get("overlap_decision") != "keep":
        raise ValueError(f"{candidate_id} 的 overlap_decision 不是 keep")
    if candidate.get("candidate_status") != "overlap_audited_draft":
        raise ValueError(f"{candidate_id} 尚未进入 overlap_audited_draft")
    if candidate.get("dataset_version") != config["input_dataset_version"]:
        raise ValueError(f"{candidate_id} 的 dataset_version 不一致")

    anchor_ids = candidate.get("evidence_anchor_ids")
    if not isinstance(anchor_ids, list) or len(anchor_ids) != 1:
        raise ValueError(f"{candidate_id} 必须绑定且仅绑定一个 evidence anchor")
    anchor = anchors_by_id.get(anchor_ids[0])
    if anchor is None:
        raise ValueError(f"{candidate_id} 绑定未知 evidence anchor")

    source_id = str(candidate.get("source_id", ""))
    source = sources_by_id.get(source_id)
    if source is None:
        raise ValueError(f"{candidate_id} 绑定未知 source_id: {source_id}")
    if source.get("status") not in {"approved", "indexed"} or not source.get(
        "included_in_kb"
    ):
        raise ValueError(f"{candidate_id} 的来源不是正式准入资料")

    _require_equal(candidate_id, "来源 ID", source_id, anchor.get("source_id"))
    _require_equal(candidate_id, "来源标题", candidate.get("source_title"), anchor.get("source_title"))
    _require_equal(candidate_id, "来源标题", candidate.get("source_title"), source.get("title"))
    _require_equal(candidate_id, "来源文件名", candidate.get("source_filename"), anchor.get("source_filename"))
    _require_equal(candidate_id, "来源文件名", candidate.get("source_filename"), source.get("filename"))
    _require_equal(candidate_id, "来源 SHA-256", candidate.get("source_sha256"), anchor.get("source_sha256"))
    _require_equal(candidate_id, "来源 SHA-256", candidate.get("source_sha256"), source.get("sha256"))
    if int(candidate.get("page_number") or 0) != int(anchor.get("page_number") or 0):
        raise ValueError(f"{candidate_id} 的证据页码不一致")
    _require_equal(candidate_id, "证据片段", candidate.get("anchor_text_span"), anchor.get("text_span"))
    _require_equal(candidate_id, "证据范围", candidate.get("evidence_scope"), anchor.get("evidence_scope"))
    _require_equal(candidate_id, "年龄范围", candidate.get("age_scope"), anchor.get("age_scope"))
    _require_equal(
        candidate_id,
        "适用条件",
        candidate.get("applicability_conditions"),
        anchor.get("applicability_conditions"),
    )
    _require_equal(
        candidate_id,
        "支持主张类型",
        candidate.get("supported_claim_types"),
        anchor.get("supported_claim_types"),
    )
    _require_equal(candidate_id, "范围核验", candidate.get("scope_check"), anchor.get("scope_check"))
    return anchor, source


def _pending_pass1_fields() -> dict[str, Any]:
    return {
        "pass1_reviewer_id": "",
        "pass1_annotator_role": "",
        "pass1_reviewed_at": "",
        "pass1_outcome": "",
        "pass1_final_question": "",
        "pass1_expected_decision": "",
        "pass1_current_kb_support": "",
        "pass1_gold_evidence_status": "",
        "pass1_required_evidence_type": [],
        "pass1_required_claims": [],
        "pass1_allowed_claims": [],
        "pass1_forbidden_claims": [],
        "pass1_missing_evidence_type": [],
        "pass1_missing_information": [],
        "pass1_risk_labels": [],
        "pass1_issues_found": [],
        "pass1_review_reason": "",
        "pass1_overlap_reaudit_status": "",
        "pass1_overlap_reaudit_report_path": "",
        "pass1_overlap_reaudit_report_sha256": "",
        "pass1_overlap_reaudited_at": "",
    }


def build_pass1_queue(
    candidates: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build a deterministic, provenance-checked pending Pass 1 queue."""
    expected_count = int(config["expected_candidate_count"])
    if len(candidates) != expected_count:
        raise ValueError(f"候选数量应为 {expected_count}，实际为 {len(candidates)}")

    anchors_by_id = _anchor_map(anchors)
    sources_by_id = _manifest_source_map(manifest)
    seen_candidate_ids: set[str] = set()
    prepared: list[dict[str, Any]] = []

    for candidate in sorted(candidates, key=lambda row: row.get("candidate_id", "")):
        candidate_id = str(candidate.get("candidate_id", ""))
        if not candidate_id or candidate_id in seen_candidate_ids:
            raise ValueError(f"候选 ID 为空或重复: {candidate_id}")
        seen_candidate_ids.add(candidate_id)
        _, source = _validate_candidate_provenance(
            candidate,
            anchors_by_id,
            sources_by_id,
            config,
        )
        row = {
            "annotation_order": 0,
            "annotation_version": config["annotation_version"],
            "candidate_id": candidate_id,
            "independence_unit_id": candidate["independence_unit_id"],
            "question": candidate["question"],
            "candidate_role": candidate["candidate_role"],
            "candidate_status": candidate["candidate_status"],
            "dataset_version": candidate["dataset_version"],
            "kb_version": candidate["kb_version"],
            "schema_version": candidate["schema_version"],
            "protocol_version": candidate["protocol_version"],
            "source_id": candidate["source_id"],
            "source_title": candidate["source_title"],
            "source_filename": candidate["source_filename"],
            "source_sha256": candidate["source_sha256"],
            "source_type": source.get("source_type", ""),
            "source_year": int(source.get("year") or 0),
            "jurisdiction": source.get("jurisdiction", ""),
            "source_can_support": list(source.get("can_support") or []),
            "source_cannot_support": list(source.get("cannot_support") or []),
            "page_number": int(candidate["page_number"]),
            "anchor_text_span": candidate["anchor_text_span"],
            "evidence_scope": candidate["evidence_scope"],
            "age_scope": candidate["age_scope"],
            "applicability_conditions": candidate["applicability_conditions"],
            "supported_claim_types": list(candidate["supported_claim_types"]),
            "scope_check": candidate["scope_check"],
            "evidence_anchor_ids": list(candidate["evidence_anchor_ids"]),
            "evidence_anchor_group_id": candidate["evidence_anchor_group_id"],
            "provisional_fact_cluster_id": candidate["provisional_fact_cluster_id"],
            "provisional_expected_decision": candidate["provisional_expected_decision"],
            "provisional_scenario_type": candidate["provisional_scenario_type"],
            "provisional_risk_labels": list(candidate["provisional_risk_labels"]),
            "current_kb_support": candidate["current_kb_support"],
            "missing_evidence_type": list(candidate["missing_evidence_type"]),
            "policy_rule_ids": list(candidate["policy_rule_ids"]),
            "overlap_decision": candidate["overlap_decision"],
            "dev50_overlap_status": candidate["dev50_overlap_status"],
            "internal_overlap_status": candidate["internal_overlap_status"],
        }
        row.update(_pending_pass1_fields())
        prepared.append(row)

    independence_units = {row["independence_unit_id"] for row in prepared}
    expected_units = int(config["expected_independence_unit_count"])
    if len(independence_units) != expected_units:
        raise ValueError(
            f"独立单元数量应为 {expected_units}，实际为 {len(independence_units)}"
        )

    random.Random(int(config["pass1_shuffle_seed"])).shuffle(prepared)
    for order, row in enumerate(prepared, start=1):
        row["annotation_order"] = order
    return prepared


def _csv_value(value: Any) -> Any:
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_pass1_queue(rows: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=PASS1_QUEUE_FIELDS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: _csv_value(row.get(field, "")) for field in PASS1_QUEUE_FIELDS}
            )


def read_pass1_queue(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    for row in rows:
        for field in JSON_LIST_FIELDS:
            raw = row.get(field, "")
            parsed = json.loads(raw) if raw else []
            if not isinstance(parsed, list):
                raise ValueError(f"{field} 必须是 JSON 数组")
            row[field] = parsed
        for field in INTEGER_FIELDS:
            row[field] = int(row.get(field) or 0)
    return rows


def _write_pass1_queue_atomic(rows: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        write_pass1_queue(rows, temp_path)
        temp_path.replace(output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _write_json_atomic(payload: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _validate_iso_timestamp(
    value: str,
    candidate_id: str,
    field_name: str = "pass1_reviewed_at",
) -> None:
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{candidate_id} 的 {field_name} 不是 ISO 时间") from exc


def _validate_pass1_metadata_quality(row: dict[str, Any], candidate_id: str) -> None:
    for field in PASS1_REVIEW_FIELDS:
        value = row.get(field)
        values = value if isinstance(value, list) else [value]
        if any(GARBLED_PASS1_METADATA_RE.search(str(item or "")) for item in values):
            raise ValueError(f"{candidate_id} 的 {field} 含乱码或编码损坏")


def _validate_reaudit_report_content(
    row: dict[str, Any],
    report: dict[str, Any],
    *,
    require_clear: bool,
) -> None:
    candidate_id = str(row["candidate_id"])
    if report.get("reaudit_version") != PASS1_REAUDIT_VERSION:
        raise ValueError(f"{candidate_id} 的复审版本不一致")
    if report.get("candidate_id") != candidate_id:
        raise ValueError(f"{candidate_id} 的复审 candidate_id 不一致")
    if _normalize_text(report.get("original_question")) != _normalize_text(row.get("question")):
        raise ValueError(f"{candidate_id} 的复审原始问题不一致")
    if _normalize_text(report.get("revised_question")) != _normalize_text(
        row.get("pass1_final_question")
    ):
        raise ValueError(f"{candidate_id} 的复审修订问题不一致")
    if report.get("independence_unit_id") != row.get("independence_unit_id"):
        raise ValueError(f"{candidate_id} 的复审独立性单元不一致")

    decision = _normalize_text(report.get("reaudit_decision"))
    if require_clear and decision != "clear":
        raise ValueError(f"{candidate_id} 的修订题重叠复审尚未通过: {decision or 'missing'}")

    usage = report.get("usage")
    if not isinstance(usage, dict):
        raise ValueError(f"{candidate_id} 的复审缺少 usage 记录")
    expected_zero_usage = {
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
    }
    if any(usage.get(field) != value for field, value in expected_zero_usage.items()):
        raise ValueError(f"{candidate_id} 的复审 usage 不是零调用记录")


def _validate_pass1_reaudit(row: dict[str, Any]) -> bool:
    candidate_id = str(row["candidate_id"])
    values = [_normalize_text(row.get(field)) for field in PASS1_REAUDIT_FIELDS]
    if not any(values):
        return False
    if not all(values):
        raise ValueError(f"{candidate_id} 的首轮重叠复审字段部分填写")
    if row["pass1_overlap_reaudit_status"] != "clear":
        raise ValueError(f"{candidate_id} 的修订题重叠复审尚未通过")

    report_path = Path(row["pass1_overlap_reaudit_report_path"])
    if not report_path.is_file():
        raise ValueError(f"{candidate_id} 的复审报告文件不存在")
    actual_sha256 = _compute_sha256(report_path)
    if actual_sha256 != row["pass1_overlap_reaudit_report_sha256"]:
        raise ValueError(f"{candidate_id} 的复审报告 SHA-256 不一致")
    _validate_iso_timestamp(
        row["pass1_overlap_reaudited_at"],
        candidate_id,
        "pass1_overlap_reaudited_at",
    )
    _validate_reaudit_report_content(row, load_json(report_path), require_clear=True)
    return True


def _validate_pass1_row(
    row: dict[str, Any],
    candidate: dict[str, Any],
    anchors_by_id: dict[str, dict[str, Any]],
    sources_by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[bool, bool]:
    candidate_id = row["candidate_id"]
    missing = sorted(
        field for field in PASS1_REQUIRED_SCALAR_FIELDS if not _normalize_text(row.get(field))
    )
    if missing:
        raise ValueError(f"首轮核验未完成: {candidate_id}: {', '.join(missing)}")
    if row["pass1_annotator_role"] != config["annotator_role"]:
        raise ValueError(f"{candidate_id} 的 annotator role 只能为 author")
    _validate_iso_timestamp(row["pass1_reviewed_at"], candidate_id)
    _validate_pass1_metadata_quality(row, candidate_id)

    outcome = row["pass1_outcome"]
    if outcome not in config["allowed_outcomes"]:
        raise ValueError(f"{candidate_id} 的 pass1_outcome 不合规")
    decision = row["pass1_expected_decision"]
    if decision not in config["allowed_decisions"]:
        raise ValueError(f"{candidate_id} 的 expected decision 不合规")
    if row["pass1_current_kb_support"] not in config["allowed_kb_support"]:
        raise ValueError(f"{candidate_id} 的 current KB support 不合规")
    evidence_status = row["pass1_gold_evidence_status"]
    if evidence_status not in config["allowed_gold_evidence_status"]:
        raise ValueError(f"{candidate_id} 的 gold evidence status 不合规")
    if not row.get("pass1_risk_labels"):
        raise ValueError(f"{candidate_id} 缺少 pass1 risk labels")
    if decision in {"answer", "review_required"}:
        if not row.get("pass1_required_evidence_type") or not row.get(
            "pass1_required_claims"
        ):
            raise ValueError(f"{candidate_id} 缺少 required evidence/claims")

    if evidence_status == "page_span_located":
        gold_fields = (
            row.get("source_id"),
            row.get("source_title"),
            row.get("source_sha256"),
            row.get("page_number"),
            row.get("anchor_text_span"),
            row.get("evidence_anchor_ids"),
        )
        if not all(gold_fields):
            raise ValueError(f"{candidate_id} 缺少完整 gold evidence")
    if evidence_status == "missing_source" and not row.get(
        "pass1_missing_evidence_type"
    ):
        raise ValueError(f"{candidate_id} 缺少 missing evidence type")
    if decision == "boundary_refusal":
        if not row.get("policy_rule_ids") or not row.get("pass1_forbidden_claims"):
            raise ValueError(
                f"{candidate_id} 的 boundary refusal 缺少 policy rule 和 forbidden claims"
            )

    _validate_candidate_provenance(
        candidate,
        anchors_by_id,
        sources_by_id,
        config,
    )
    for field in (
        "question",
        "source_id",
        "source_title",
        "source_filename",
        "source_sha256",
        "page_number",
        "anchor_text_span",
        "evidence_anchor_ids",
        "independence_unit_id",
    ):
        _require_equal(candidate_id, f"首轮队列字段 {field}", row.get(field), candidate.get(field))

    final_question = _normalize_text(row["pass1_final_question"])
    original_question = _normalize_text(row["question"])
    if outcome == "accepted" and final_question != original_question:
        raise ValueError(f"{candidate_id} 的 accepted 记录不能修改问题")
    if outcome == "revise" and final_question == original_question:
        raise ValueError(f"{candidate_id} 的 revise 记录必须修改问题")
    has_reaudit_values = any(_normalize_text(row.get(field)) for field in PASS1_REAUDIT_FIELDS)
    if outcome != "revise" and has_reaudit_values:
        raise ValueError(f"{candidate_id} 的非 revise 记录不能包含重叠复审结果")
    reaudit_clear = _validate_pass1_reaudit(row) if outcome == "revise" else False
    overlap_reaudit_required = outcome == "revise" and not reaudit_clear
    promotable = outcome == "accepted" or (outcome == "revise" and reaudit_clear)
    return promotable, overlap_reaudit_required


def _has_pass1_values(row: dict[str, Any]) -> bool:
    return any(
        bool(value) if isinstance(value, list) else bool(_normalize_text(value))
        for value in (row.get(field) for field in PASS1_REVIEW_FIELDS)
    )


def validate_pass1_progress(
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    if len(rows) != int(config["expected_candidate_count"]):
        raise ValueError("首轮核验记录数量与配置不一致")
    candidate_map = {row["candidate_id"]: row for row in candidates}
    if len(candidate_map) != len(candidates):
        raise ValueError("候选 ID 存在重复")
    anchors_by_id = _anchor_map(anchors)
    sources_by_id = _manifest_source_map(manifest)

    outcomes: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    seen_ids: set[str] = set()
    promotable_count = 0
    reaudit_count = 0
    pending_count = 0
    for row in rows:
        candidate_id = row.get("candidate_id", "")
        if candidate_id in seen_ids or candidate_id not in candidate_map:
            raise ValueError(f"首轮记录 ID 重复或未知: {candidate_id}")
        seen_ids.add(candidate_id)
        if not _normalize_text(row.get("pass1_outcome")):
            if _has_pass1_values(row):
                raise ValueError(f"首轮记录部分填写但未完成: {candidate_id}")
            pending_count += 1
            continue
        promotable, needs_reaudit = _validate_pass1_row(
            row,
            candidate_map[candidate_id],
            anchors_by_id,
            sources_by_id,
            config,
        )
        outcomes[row["pass1_outcome"]] += 1
        decisions[row["pass1_expected_decision"]] += 1
        promotable_count += int(promotable)
        reaudit_count += int(needs_reaudit)

    if seen_ids != set(candidate_map):
        raise ValueError("首轮核验未覆盖全部候选")
    completed_count = len(rows) - pending_count
    return {
        "annotation_version": config["annotation_version"],
        "config_version": config["config_version"],
        "dataset_version": config["output_dataset_version"],
        "kb_version": config["kb_version"],
        "candidate_count": len(rows),
        "completed_count": completed_count,
        "pending_count": pending_count,
        "status": "pass1_pending" if pending_count else "pass1_completed",
        "outcome_distribution": dict(sorted(outcomes.items())),
        "decision_distribution": dict(sorted(decisions.items())),
        "promotable_to_pass2_count": promotable_count,
        "overlap_reaudit_required_count": reaudit_count,
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
    }


def apply_pass1_batch(
    rows: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row_ids = {row.get("candidate_id", "") for row in rows}
    decision_ids = [str(item.get("candidate_id", "")) for item in decisions]
    if len(set(decision_ids)) != len(decision_ids):
        raise ValueError("批次 candidate_id 存在重复")
    unknown_ids = sorted(set(decision_ids) - row_ids)
    if unknown_ids:
        raise ValueError(f"批次包含未知 candidate_id: {', '.join(unknown_ids)}")

    allowed_fields = {"candidate_id", *PASS1_REVIEW_FIELDS}
    updated = copy.deepcopy(rows)
    updated_by_id = {row["candidate_id"]: row for row in updated}
    for decision in decisions:
        candidate_id = str(decision["candidate_id"])
        target = updated_by_id[candidate_id]
        if _has_pass1_values(target):
            raise ValueError(f"首轮记录已完成或已部分填写，不能覆盖: {candidate_id}")
        unexpected_fields = sorted(set(decision) - allowed_fields)
        if unexpected_fields:
            raise ValueError(
                f"{candidate_id} 包含未知首轮字段: {', '.join(unexpected_fields)}"
            )
        for field in PASS1_REVIEW_FIELDS:
            target[field] = copy.deepcopy(decision.get(field, [] if field in JSON_LIST_FIELDS else ""))

    summary = validate_pass1_progress(
        updated,
        candidates,
        anchors,
        manifest,
        config,
    )
    return updated, summary


def apply_pass1_correction(
    rows: list[dict[str, Any]],
    correction: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply one exact-match metadata correction without rewriting review history."""
    required = {
        "correction_id",
        "annotation_version",
        "dataset_version",
        "kb_version",
        "parent_queue_sha256",
        "candidate_id",
        "field",
        "expected_old_value",
        "corrected_value",
        "reason",
    }
    missing = sorted(required - set(correction))
    if missing:
        raise ValueError(f"首轮纠错记录缺少字段: {', '.join(missing)}")
    unexpected = sorted(set(correction) - required)
    if unexpected:
        raise ValueError(f"首轮纠错记录包含未知字段: {', '.join(unexpected)}")
    if correction["annotation_version"] != config["annotation_version"]:
        raise ValueError("首轮纠错 annotation_version 与配置不一致")
    if correction["dataset_version"] != config["output_dataset_version"]:
        raise ValueError("首轮纠错 dataset_version 与配置不一致")
    if correction["kb_version"] != config["kb_version"]:
        raise ValueError("首轮纠错 kb_version 与配置不一致")

    correction_id = _normalize_text(correction["correction_id"])
    candidate_id = _normalize_text(correction["candidate_id"])
    field = _normalize_text(correction["field"])
    if not correction_id or not candidate_id:
        raise ValueError("首轮纠错 correction_id 和 candidate_id 不能为空")
    if field not in PASS1_CORRECTION_FIELDS:
        raise ValueError(f"不允许纠正字段: {field}")
    if not _normalize_text(correction["reason"]):
        raise ValueError(f"{correction_id} 的纠错原因不能为空")
    if not _normalize_text(correction["corrected_value"]):
        raise ValueError(f"{correction_id} 的纠正值不能为空")
    if GARBLED_PASS1_METADATA_RE.search(str(correction["corrected_value"])):
        raise ValueError(f"{correction_id} 的纠正值含乱码或编码损坏")

    matches = [row for row in rows if row.get("candidate_id") == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"纠错 candidate_id 必须唯一存在于首轮队列: {candidate_id}")
    target = matches[0]
    if not _normalize_text(target.get("pass1_outcome")):
        raise ValueError(f"{candidate_id} 尚未完成首轮核验，不能应用元数据纠错")
    if target.get(field) != correction["expected_old_value"]:
        raise ValueError(f"{candidate_id} 的 {field} 当前值不一致，拒绝覆盖")
    if correction["corrected_value"] == correction["expected_old_value"]:
        raise ValueError(f"{candidate_id} 的纠正值与旧值相同")

    updated = copy.deepcopy(rows)
    updated_target = next(row for row in updated if row["candidate_id"] == candidate_id)
    updated_target[field] = correction["corrected_value"]
    _validate_pass1_metadata_quality(updated_target, candidate_id)
    return updated


def apply_pass1_reaudit(
    rows: list[dict[str, Any]],
    report: dict[str, Any],
    report_path: str | Path,
    report_sha256: str,
    candidates: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report_path = Path(report_path)
    candidate_id = _normalize_text(report.get("candidate_id"))
    matches = [row for row in rows if row.get("candidate_id") == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"复审 candidate_id 必须唯一存在于首轮队列: {candidate_id}")
    target = matches[0]
    if target.get("pass1_outcome") != "revise":
        raise ValueError(f"{candidate_id} 不是待复审的 revise 记录")
    if any(_normalize_text(target.get(field)) for field in PASS1_REAUDIT_FIELDS):
        raise ValueError(f"{candidate_id} 的重叠复审结果已存在，不能覆盖")
    if not report_path.is_file():
        raise ValueError(f"{candidate_id} 的复审报告文件不存在")
    actual_sha256 = _compute_sha256(report_path)
    if actual_sha256 != report_sha256:
        raise ValueError(f"{candidate_id} 的复审报告 SHA-256 不一致")
    if load_json(report_path) != report:
        raise ValueError(f"{candidate_id} 的复审报告内存内容与文件不一致")
    _validate_reaudit_report_content(target, report, require_clear=True)

    updated = copy.deepcopy(rows)
    updated_target = next(row for row in updated if row["candidate_id"] == candidate_id)
    updated_target.update(
        {
            "pass1_overlap_reaudit_status": "clear",
            "pass1_overlap_reaudit_report_path": report_path.as_posix(),
            "pass1_overlap_reaudit_report_sha256": report_sha256,
            "pass1_overlap_reaudited_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
        }
    )
    summary = validate_pass1_progress(
        updated,
        candidates,
        anchors,
        manifest,
        config,
    )
    return updated, summary


def validate_completed_pass1(
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    summary = validate_pass1_progress(rows, candidates, anchors, manifest, config)
    if summary["pending_count"]:
        raise ValueError(f"首轮核验未完成: 仍有 {summary['pending_count']} 条 pending")
    return summary


def summarize_pass1_queue(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    completed = sum(bool(_normalize_text(row.get("pass1_outcome"))) for row in rows)
    return {
        "annotation_version": config["annotation_version"],
        "config_version": config["config_version"],
        "dataset_version": config["output_dataset_version"],
        "kb_version": config["kb_version"],
        "candidate_count": len(rows),
        "independence_unit_count": len(
            {row["independence_unit_id"] for row in rows}
        ),
        "pending_count": len(rows) - completed,
        "completed_count": completed,
        "status": "pass1_pending" if completed < len(rows) else "pass1_completed",
        "candidate_role_distribution": dict(
            sorted(Counter(row["candidate_role"] for row in rows).items())
        ),
        "provisional_decision_distribution": dict(
            sorted(
                Counter(
                    row["provisional_expected_decision"] for row in rows
                ).items()
            )
        ),
        "source_distribution": dict(
            sorted(Counter(row["source_id"] for row in rows).items())
        ),
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
    }


def ensure_prepare_output_is_safe(path: str | Path) -> None:
    output_path = Path(path)
    if not output_path.exists():
        return
    reviewed_ids = [
        row["candidate_id"]
        for row in read_pass1_queue(output_path)
        if _has_pass1_values(row)
    ]
    if reviewed_ids:
        raise ValueError(
            "现有首轮队列已包含人工核验结果，prepare-pass1 不能覆盖: "
            + ", ".join(reviewed_ids[:5])
        )


def _load_pass1_batch(path: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    payload = load_json(path)
    required = {
        "batch_id",
        "annotation_version",
        "dataset_version",
        "kb_version",
        "records",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"首轮批次缺少字段: {', '.join(missing)}")
    if payload["annotation_version"] != config["annotation_version"]:
        raise ValueError("首轮批次 annotation_version 与配置不一致")
    if payload["dataset_version"] != config["output_dataset_version"]:
        raise ValueError("首轮批次 dataset_version 与配置不一致")
    if payload["kb_version"] != config["kb_version"]:
        raise ValueError("首轮批次 kb_version 与配置不一致")
    if not isinstance(payload["records"], list) or not payload["records"]:
        raise ValueError("首轮批次 records 必须是非空数组")
    return payload


def _default_paths() -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    benchmark_dir = repo_root / "revision" / "benchmark" / "benchmark_v1"
    return {
        "config": Path(__file__).resolve().parent
        / "configs"
        / "benchmark_annotation_v0_1.json",
        "candidates": benchmark_dir / "benchmark_candidates_v0_2_deduplicated.jsonl",
        "anchors": benchmark_dir / "evidence_anchor_pool_v0_1.jsonl",
        "manifest": repo_root / "data" / "guidelines" / "source_manifest.json",
        "output": benchmark_dir / "annotation_pass1_queue_v0_1.csv",
        "summary": benchmark_dir / "annotation_pass1_summary_v0_1.json",
        "progress_summary": benchmark_dir / "annotation_pass1_progress_v0_1.json",
    }


def run_prepare_pass1(
    config_path: str | Path,
    candidate_path: str | Path,
    anchor_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
) -> dict[str, Any]:
    ensure_prepare_output_is_safe(output_path)
    config = load_annotation_config(config_path)
    rows = build_pass1_queue(
        load_jsonl(candidate_path),
        load_jsonl(anchor_path),
        load_json(manifest_path),
        config,
    )
    write_pass1_queue(rows, output_path)
    summary = summarize_pass1_queue(rows, config)
    summary.update(
        {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "input_artifacts": {
                "candidates_sha256": _compute_sha256(candidate_path),
                "anchors_sha256": _compute_sha256(anchor_path),
                "source_manifest_sha256": _compute_sha256(manifest_path),
            },
            "output_artifact": {
                "path": Path(output_path).as_posix(),
                "sha256": _compute_sha256(output_path),
            },
            "workflow_boundary": (
                "B3.1 only: pending two-pass author verification; not verified, "
                "not frozen, and not expert-validated"
            ),
        }
    )
    summary_output = Path(summary_path)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def run_apply_pass1_batch(
    config_path: str | Path,
    candidate_path: str | Path,
    anchor_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    batch_path: str | Path,
    progress_summary_path: str | Path,
) -> dict[str, Any]:
    config = load_annotation_config(config_path)
    batch = _load_pass1_batch(batch_path, config)
    progress_path = Path(progress_summary_path)
    previous_progress = load_json(progress_path) if progress_path.exists() else {}
    applied_batches = list(previous_progress.get("applied_batches", []))
    batch_id = _normalize_text(batch["batch_id"])
    if not batch_id:
        raise ValueError("首轮批次 batch_id 不能为空")
    if batch_id in applied_batches:
        raise ValueError(f"首轮批次已应用，不能重复执行: {batch_id}")

    previous_queue_sha256 = _compute_sha256(output_path)
    updated_rows, summary = apply_pass1_batch(
        read_pass1_queue(output_path),
        batch["records"],
        load_jsonl(candidate_path),
        load_jsonl(anchor_path),
        load_json(manifest_path),
        config,
    )
    _write_pass1_queue_atomic(updated_rows, output_path)
    applied_batches.append(batch_id)
    summary = {**previous_progress, **summary}
    summary.update(
        {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "applied_batches": applied_batches,
            "latest_batch": {
                "batch_id": batch_id,
                "path": Path(batch_path).as_posix(),
                "sha256": _compute_sha256(batch_path),
                "record_count": len(batch["records"]),
            },
            "previous_queue_sha256": previous_queue_sha256,
            "current_queue_sha256": _compute_sha256(output_path),
            "workflow_boundary": (
                "B3.2 first-pass author verification in progress; not frozen and "
                "not expert-validated"
            ),
        }
    )
    _write_json_atomic(summary, progress_path)
    return summary


def run_apply_pass1_correction(
    config_path: str | Path,
    output_path: str | Path,
    correction_path: str | Path,
    progress_summary_path: str | Path,
) -> dict[str, Any]:
    config = load_annotation_config(config_path)
    correction_path = Path(correction_path)
    correction = load_json(correction_path)
    progress_path = Path(progress_summary_path)
    previous_progress = load_json(progress_path) if progress_path.exists() else {}
    applied_corrections = list(previous_progress.get("applied_corrections", []))
    correction_id = _normalize_text(correction.get("correction_id"))
    if any(
        item.get("correction_id") == correction_id
        for item in applied_corrections
    ):
        raise ValueError(f"首轮纠错已应用，不能重复执行: {correction_id}")

    previous_queue_sha256 = _compute_sha256(output_path)
    parent_queue_sha256 = _normalize_text(correction.get("parent_queue_sha256"))
    if parent_queue_sha256 != previous_queue_sha256:
        raise ValueError(f"{correction_id or '首轮纠错'} 的父队列 SHA-256 与当前队列不一致")

    updated_rows = apply_pass1_correction(
        read_pass1_queue(output_path),
        correction,
        config,
    )
    _write_pass1_queue_atomic(updated_rows, output_path)
    audit_record = {
        "correction_id": correction_id,
        "candidate_id": _normalize_text(correction["candidate_id"]),
        "field": _normalize_text(correction["field"]),
        "path": correction_path.as_posix(),
        "sha256": _compute_sha256(correction_path),
    }
    applied_corrections.append(audit_record)
    summary = dict(previous_progress)
    summary.update(
        {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "applied_corrections": applied_corrections,
            "latest_correction": audit_record,
            "previous_queue_sha256": previous_queue_sha256,
            "current_queue_sha256": _compute_sha256(output_path),
            "workflow_boundary": (
                "B3.2 first-pass author verification in progress; metadata "
                "corrections are exact-match and audit logged; not frozen and "
                "not expert-validated"
            ),
        }
    )
    _write_json_atomic(summary, progress_path)
    return summary


def run_apply_pass1_reaudit(
    config_path: str | Path,
    candidate_path: str | Path,
    anchor_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    progress_summary_path: str | Path,
) -> dict[str, Any]:
    config = load_annotation_config(config_path)
    report_path = Path(report_path)
    report = load_json(report_path)
    progress_path = Path(progress_summary_path)
    previous_progress = load_json(progress_path) if progress_path.exists() else {}
    applied_reaudits = list(previous_progress.get("applied_reaudits", []))
    candidate_id = _normalize_text(report.get("candidate_id"))
    if any(item.get("candidate_id") == candidate_id for item in applied_reaudits):
        raise ValueError(f"修订题重叠复审已应用，不能重复执行: {candidate_id}")

    previous_queue_sha256 = _compute_sha256(output_path)
    parent_queue_sha256 = _normalize_text(
        (report.get("parent_artifacts") or {}).get("queue_sha256")
    )
    if parent_queue_sha256 != previous_queue_sha256:
        raise ValueError(f"{candidate_id} 的复审父队列 SHA-256 与当前队列不一致")
    report_sha256 = _compute_sha256(report_path)
    updated_rows, summary = apply_pass1_reaudit(
        read_pass1_queue(output_path),
        report,
        report_path,
        report_sha256,
        load_jsonl(candidate_path),
        load_jsonl(anchor_path),
        load_json(manifest_path),
        config,
    )
    _write_pass1_queue_atomic(updated_rows, output_path)
    applied_reaudits.append(
        {
            "candidate_id": candidate_id,
            "status": "clear",
            "path": report_path.as_posix(),
            "sha256": report_sha256,
        }
    )
    summary = {**previous_progress, **summary}
    summary.update(
        {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "applied_reaudits": applied_reaudits,
            "latest_reaudit": applied_reaudits[-1],
            "previous_queue_sha256": previous_queue_sha256,
            "current_queue_sha256": _compute_sha256(output_path),
            "workflow_boundary": (
                "B3.2 first-pass author verification in progress; revised questions "
                "are promotable only after a clear overlap re-audit; not frozen and "
                "not expert-validated"
            ),
        }
    )
    _write_json_atomic(summary, progress_path)
    return summary


def run_validate_pass1_progress(
    config_path: str | Path,
    candidate_path: str | Path,
    anchor_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    return validate_pass1_progress(
        read_pass1_queue(output_path),
        load_jsonl(candidate_path),
        load_jsonl(anchor_path),
        load_json(manifest_path),
        load_annotation_config(config_path),
    )


def main() -> int:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(
        description="Prepare, incrementally apply, or validate Benchmark-v1 pass 1."
    )
    parser.add_argument(
        "--mode",
        choices=[
            "prepare-pass1",
            "apply-pass1-batch",
            "apply-pass1-correction",
            "apply-pass1-reaudit",
            "validate-pass1-progress",
        ],
        default="prepare-pass1",
    )
    for name in ("config", "candidates", "anchors", "manifest", "output", "summary"):
        parser.add_argument(f"--{name}", type=Path, default=defaults[name])
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--correction", type=Path)
    parser.add_argument("--reaudit-report", type=Path)
    parser.add_argument(
        "--progress-summary",
        type=Path,
        default=defaults["progress_summary"],
    )
    args = parser.parse_args()
    if args.mode == "prepare-pass1":
        summary = run_prepare_pass1(
            args.config,
            args.candidates,
            args.anchors,
            args.manifest,
            args.output,
            args.summary,
        )
    elif args.mode == "apply-pass1-batch":
        if args.batch is None:
            parser.error("apply-pass1-batch 需要 --batch")
        summary = run_apply_pass1_batch(
            args.config,
            args.candidates,
            args.anchors,
            args.manifest,
            args.output,
            args.batch,
            args.progress_summary,
        )
    elif args.mode == "apply-pass1-correction":
        if args.correction is None:
            parser.error("apply-pass1-correction 需要 --correction")
        summary = run_apply_pass1_correction(
            args.config,
            args.output,
            args.correction,
            args.progress_summary,
        )
    elif args.mode == "apply-pass1-reaudit":
        if args.reaudit_report is None:
            parser.error("apply-pass1-reaudit 需要 --reaudit-report")
        summary = run_apply_pass1_reaudit(
            args.config,
            args.candidates,
            args.anchors,
            args.manifest,
            args.output,
            args.reaudit_report,
            args.progress_summary,
        )
    else:
        summary = run_validate_pass1_progress(
            args.config,
            args.candidates,
            args.anchors,
            args.manifest,
            args.output,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
