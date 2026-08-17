from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

SCALAR_COMPARISON_FIELDS = (
    ("expected_decision", "first_pass_expected_decision", "pass2_expected_decision"),
    (
        "current_kb_support",
        "first_pass_current_kb_support",
        "pass2_current_kb_support",
    ),
    (
        "gold_evidence_status",
        "first_pass_gold_evidence_status",
        "pass2_gold_evidence_status",
    ),
)

LIST_COMPARISON_FIELDS = (
    (
        "required_evidence_type",
        "first_pass_required_evidence_type",
        "pass2_required_evidence_type",
    ),
    ("required_claims", "first_pass_required_claims", "pass2_required_claims"),
    ("allowed_claims", "first_pass_allowed_claims", "pass2_allowed_claims"),
    (
        "forbidden_claims",
        "first_pass_forbidden_claims",
        "pass2_forbidden_claims",
    ),
    (
        "missing_evidence_type",
        "first_pass_missing_evidence_type",
        "pass2_missing_evidence_type",
    ),
    (
        "missing_information",
        "first_pass_missing_information",
        "pass2_missing_information",
    ),
    ("risk_labels", "first_pass_risk_labels", "pass2_risk_labels"),
)

COMPARISON_FIELDS = tuple(
    item[0] for item in SCALAR_COMPARISON_FIELDS + LIST_COMPARISON_FIELDS
)

RESOLUTION_EMPTY_FIELDS = (
    "resolution_reviewer_id",
    "resolution_annotator_role",
    "resolution_reviewed_at",
    "resolution_status",
    "resolution_final_decision",
    "resolution_final_kb_support",
    "resolution_final_gold_evidence_status",
    "resolution_final_required_evidence_type",
    "resolution_final_required_claims",
    "resolution_final_allowed_claims",
    "resolution_final_forbidden_claims",
    "resolution_final_missing_evidence_type",
    "resolution_final_missing_information",
    "resolution_final_risk_labels",
    "resolution_reason",
)

LINKAGE_METADATA_FIELDS = (
    "question",
    "dataset_split",
    "selection_version",
    "source_id",
    "page_number",
)

PASS2_VISIBLE_FIELDS = (
    "pass2_order",
    "pass2_item_id",
    "pass2_annotation_version",
    "dataset_version",
    "kb_version",
    "protocol_version",
    "question",
    "source_id",
    "source_title",
    "source_filename",
    "source_sha256",
    "source_type",
    "source_year",
    "jurisdiction",
    "page_number",
    "anchor_text_span",
    "policy_rule_ids",
    "policy_evidence_text",
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


def zero_usage() -> dict[str, int | float]:
    return {
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    content = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _json_list(value: Any, *, field_name: str) -> list[Any]:
    if isinstance(value, list):
        parsed = value
    else:
        try:
            parsed = json.loads(str(value or "[]"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} 必须是 JSON 数组") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{field_name} 必须是 JSON 数组")
    return parsed


def _normalize_list(value: Any, *, field_name: str) -> list[str]:
    parsed = _json_list(value, field_name=field_name)
    normalized = [
        _text(item)
        if isinstance(item, (str, int, float, bool)) or item is None
        else json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for item in parsed
    ]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} 不允许重复项")
    return sorted(normalized)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def _assert_clean_text(row: dict[str, Any], *, context: str) -> None:
    for key, value in row.items():
        text = str(value or "")
        if "\ufffd" in text or "???" in text:
            raise ValueError(f"{context} 存在乱码字段 {key}")


def _index_unique(
    rows: list[dict[str, Any]],
    *,
    id_field: str,
    context: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = _text(row.get(id_field))
        if not item_id:
            raise ValueError(f"{context} 的 {id_field} 不能为空")
        if item_id in indexed:
            raise ValueError(f"{context} 存在重复 {id_field}: {item_id}")
        indexed[item_id] = row
    return indexed


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "config_version",
        "resolution_version",
        "dataset_version",
        "kb_version",
        "protocol_version",
        "expected_selection_count",
        "expected_second_pass_count",
        "expected_resolution_count",
        "expected_full_agreement_count",
        "allowed_decisions",
        "allowed_kb_support",
        "allowed_gold_evidence_status",
        "external_model_calls",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Benchmark120 裁决配置缺少字段: {missing}")
    if int(config["external_model_calls"]) != 0:
        raise ValueError("双轮关联与分歧检测不允许外部模型调用")


def _validate_progress(
    progress: dict[str, Any],
    config: dict[str, Any],
    *,
    pass2_queue_sha256: str,
) -> None:
    expected_count = int(config["expected_second_pass_count"])
    if (
        _text(progress.get("status")) != "complete"
        or int(progress.get("pending_count", -1)) != 0
        or int(progress.get("completed_count", -1)) != expected_count
    ):
        raise ValueError("第二轮尚未完整完成，禁止打开私有映射")
    if int(progress.get("candidate_count", -1)) != expected_count:
        raise ValueError("第二轮进度中的候选数量与配置不一致")
    if int(progress.get("promotable_count", -1)) != expected_count:
        raise ValueError("第二轮存在未通过复核的记录，禁止生成裁决队列")
    if _text(progress.get("queue_sha256")) != pass2_queue_sha256:
        raise ValueError("第二轮进度记录的队列哈希与输入文件不一致")
    for field in ("dataset_version", "kb_version"):
        if _text(progress.get(field)) != _text(config[field]):
            raise ValueError(f"第二轮进度的 {field} 与配置不一致")


def _validate_allowed_values(
    selection_row: dict[str, Any],
    pass2_row: dict[str, Any],
    config: dict[str, Any],
) -> None:
    candidate_id = _text(selection_row["candidate_id"])
    checks = (
        (
            selection_row.get("expected_decision"),
            config["allowed_decisions"],
            "第一轮决策",
        ),
        (
            pass2_row.get("pass2_expected_decision"),
            config["allowed_decisions"],
            "第二轮决策",
        ),
        (
            selection_row.get("current_kb_support"),
            config["allowed_kb_support"],
            "第一轮知识库支持状态",
        ),
        (
            pass2_row.get("pass2_current_kb_support"),
            config["allowed_kb_support"],
            "第二轮知识库支持状态",
        ),
        (
            selection_row.get("gold_evidence_status"),
            config["allowed_gold_evidence_status"],
            "第一轮证据状态",
        ),
        (
            pass2_row.get("pass2_gold_evidence_status"),
            config["allowed_gold_evidence_status"],
            "第二轮证据状态",
        ),
    )
    for value, allowed, label in checks:
        if _text(value) not in set(allowed):
            raise ValueError(f"{candidate_id} 的{label}非法: {value}")


def _validate_linkage_against_selection(
    selection_row: dict[str, Any],
    linkage_row: dict[str, Any],
) -> None:
    candidate_id = _text(selection_row["candidate_id"])
    expected_hash = _canonical_sha256(selection_row)
    if _text(linkage_row.get("selection_row_sha256")) != expected_hash:
        raise ValueError(f"{candidate_id} 的 selection_row_sha256 校验失败")
    for field in LINKAGE_METADATA_FIELDS:
        if _text(linkage_row.get(field)) != _text(selection_row.get(field)):
            raise ValueError(f"{candidate_id} 的 linkage 字段漂移: {field}")
    for field, linkage_field, _ in SCALAR_COMPARISON_FIELDS:
        if _text(linkage_row.get(linkage_field)) != _text(selection_row.get(field)):
            raise ValueError(f"{candidate_id} 的第一轮标量字段漂移: {field}")
    for field, linkage_field, _ in LIST_COMPARISON_FIELDS:
        linkage_value = _normalize_list(
            linkage_row.get(linkage_field), field_name=linkage_field
        )
        selection_value = _normalize_list(
            selection_row.get(field), field_name=field
        )
        if linkage_value != selection_value:
            raise ValueError(f"{candidate_id} 的第一轮数组字段漂移: {field}")


def _validate_pass2_row(
    selection_row: dict[str, Any],
    linkage_row: dict[str, Any],
    pass2_row: dict[str, Any],
    config: dict[str, Any],
) -> None:
    item_id = _text(pass2_row.get("pass2_item_id"))
    if _text(pass2_row.get("pass2_outcome")) != "accepted":
        raise ValueError(f"第二轮记录未被接受: {item_id}")
    question = _text(selection_row.get("question"))
    if (
        _text(linkage_row.get("question")) != question
        or _text(pass2_row.get("question")) != question
        or _text(pass2_row.get("pass2_final_question")) != question
    ):
        raise ValueError(f"第二轮问题文本漂移: {item_id}")
    for field in ("source_id", "page_number"):
        if (
            _text(pass2_row.get(field)) != _text(selection_row.get(field))
            or _text(linkage_row.get(field)) != _text(selection_row.get(field))
        ):
            raise ValueError(f"{item_id} 的来源定位字段漂移: {field}")
    for field in ("dataset_version", "kb_version", "protocol_version"):
        if _text(pass2_row.get(field)) != _text(config[field]):
            raise ValueError(f"{item_id} 的 {field} 与配置不一致")
    _validate_allowed_values(selection_row, pass2_row, config)
    _assert_clean_text(pass2_row, context=f"第二轮记录 {item_id}")


def _compare_fields(
    linkage_row: dict[str, Any],
    pass2_row: dict[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    disagreements: list[str] = []
    comparison: dict[str, dict[str, Any]] = {}
    for field, first_key, second_key in SCALAR_COMPARISON_FIELDS:
        first = _text(linkage_row.get(first_key))
        second = _text(pass2_row.get(second_key))
        agrees = first == second
        comparison[field] = {
            "first_pass": first,
            "second_pass": second,
            "agrees": agrees,
        }
        if not agrees:
            disagreements.append(field)
    for field, first_key, second_key in LIST_COMPARISON_FIELDS:
        first = _normalize_list(linkage_row.get(first_key), field_name=first_key)
        second = _normalize_list(pass2_row.get(second_key), field_name=second_key)
        agrees = first == second
        comparison[field] = {
            "first_pass_normalized": first,
            "second_pass_normalized": second,
            "agrees": agrees,
        }
        if not agrees:
            disagreements.append(field)
    return disagreements, comparison


def _make_resolution_row(
    selection_row: dict[str, Any],
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
        "protocol_version": config["protocol_version"],
        "candidate_id": selection_row["candidate_id"],
        "pass2_item_id": pass2_row["pass2_item_id"],
        "question": selection_row["question"],
        "dataset_split": selection_row["dataset_split"],
        "selection_row_sha256": linkage_row["selection_row_sha256"],
        "source_id": selection_row["source_id"],
        "source_title": selection_row.get("source_title", ""),
        "source_filename": selection_row.get("source_filename", ""),
        "source_sha256": selection_row.get("source_sha256", ""),
        "page_number": selection_row["page_number"],
        "anchor_text_span": selection_row.get("anchor_text_span", ""),
    }
    for _, first_key, _ in SCALAR_COMPARISON_FIELDS + LIST_COMPARISON_FIELDS:
        row[first_key] = _csv_value(linkage_row.get(first_key, ""))
    for field in PASS2_VISIBLE_FIELDS:
        if field not in row:
            row[field] = _csv_value(pass2_row.get(field, ""))
    row["disagreement_fields"] = json.dumps(
        disagreement_fields, ensure_ascii=False, separators=(",", ":")
    )
    for field in RESOLUTION_EMPTY_FIELDS:
        row[field] = ""
    _assert_clean_text(row, context=f"待裁决记录 {row['candidate_id']}")
    return row


def _validate_expected_distribution(
    actual: Counter[str],
    config: dict[str, Any],
    config_key: str,
) -> None:
    if config_key not in config:
        return
    expected = {str(key): int(value) for key, value in config[config_key].items()}
    if dict(sorted(actual.items())) != dict(sorted(expected.items())):
        raise ValueError(
            f"{config_key} 与实际结果不一致: "
            f"expected={expected}, actual={dict(sorted(actual.items()))}"
        )


def build_resolution_artifacts(
    selection_rows: list[dict[str, Any]],
    linkage: dict[str, Any],
    pass2_rows: list[dict[str, Any]],
    progress: dict[str, Any],
    config: dict[str, Any],
    *,
    selection_sha256: str,
    linkage_sha256: str,
    pass2_queue_sha256: str,
    progress_sha256: str,
) -> dict[str, Any]:
    """关联 Benchmark120 的双轮审核，并仅输出真实分歧。"""
    _validate_config(config)
    if len(selection_rows) != int(config["expected_selection_count"]):
        raise ValueError("Benchmark120 选择草案数量与配置不一致")
    expected_count = int(config["expected_second_pass_count"])
    if len(pass2_rows) != expected_count:
        raise ValueError("Benchmark120 第二轮记录数量与配置不一致")
    linkage_records = list(linkage.get("records", []))
    if int(linkage.get("record_count", -1)) != len(linkage_records):
        raise ValueError("私有 linkage 的 record_count 与记录数不一致")
    _validate_progress(
        progress,
        config,
        pass2_queue_sha256=pass2_queue_sha256,
    )

    selection_by_id = _index_unique(
        selection_rows,
        id_field="candidate_id",
        context="Benchmark120 选择草案",
    )
    linkage_by_item = _index_unique(
        linkage_records,
        id_field="pass2_item_id",
        context="Benchmark120 私有 linkage",
    )
    pass2_by_item = _index_unique(
        pass2_rows,
        id_field="pass2_item_id",
        context="Benchmark120 第二轮队列",
    )
    if len(linkage_records) != expected_count:
        raise ValueError("私有 linkage 数量与配置不一致")
    if set(linkage_by_item) != set(pass2_by_item):
        raise ValueError("私有 linkage 与第二轮队列的 pass2_item_id 集合不一致")

    candidate_ids = [_text(row.get("candidate_id")) for row in linkage_records]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("私有 linkage 存在重复 candidate_id")
    if any(candidate_id not in selection_by_id for candidate_id in candidate_ids):
        raise ValueError("私有 linkage 引用了选择草案中不存在的 candidate_id")

    linked_comparison: list[dict[str, Any]] = []
    resolution_queue: list[dict[str, Any]] = []
    disagreement_fields_counter: Counter[str] = Counter()
    pass1_decisions: Counter[str] = Counter()
    pass2_decisions: Counter[str] = Counter()
    decision_transitions: Counter[str] = Counter()
    support_transitions: Counter[str] = Counter()
    evidence_transitions: Counter[str] = Counter()

    ordered_rows = sorted(pass2_rows, key=lambda row: int(row["pass2_order"]))
    for pass2_row in ordered_rows:
        item_id = _text(pass2_row["pass2_item_id"])
        linkage_row = linkage_by_item[item_id]
        candidate_id = _text(linkage_row["candidate_id"])
        selection_row = selection_by_id[candidate_id]
        if selection_row.get("requires_second_pass") is not True:
            raise ValueError(f"{candidate_id} 未标记为 requires_second_pass")
        _validate_linkage_against_selection(selection_row, linkage_row)
        _validate_pass2_row(selection_row, linkage_row, pass2_row, config)

        disagreements, comparison = _compare_fields(linkage_row, pass2_row)
        linked_comparison.append(
            {
                "pass2_order": int(pass2_row["pass2_order"]),
                "pass2_item_id": item_id,
                "candidate_id": candidate_id,
                "question": selection_row["question"],
                "source_id": selection_row["source_id"],
                "page_number": selection_row["page_number"],
                "selection_row_sha256": linkage_row["selection_row_sha256"],
                "disagreement_fields": disagreements,
                "comparison": comparison,
            }
        )
        disagreement_fields_counter.update(disagreements)

        first_decision = _text(linkage_row["first_pass_expected_decision"])
        second_decision = _text(pass2_row["pass2_expected_decision"])
        first_support = _text(linkage_row["first_pass_current_kb_support"])
        second_support = _text(pass2_row["pass2_current_kb_support"])
        first_evidence = _text(linkage_row["first_pass_gold_evidence_status"])
        second_evidence = _text(pass2_row["pass2_gold_evidence_status"])
        pass1_decisions[first_decision] += 1
        pass2_decisions[second_decision] += 1
        decision_transitions[f"{first_decision}->{second_decision}"] += 1
        support_transitions[f"{first_support}->{second_support}"] += 1
        evidence_transitions[f"{first_evidence}->{second_evidence}"] += 1

        if disagreements:
            resolution_queue.append(
                _make_resolution_row(
                    selection_row,
                    linkage_row,
                    pass2_row,
                    resolution_order=len(resolution_queue) + 1,
                    disagreement_fields=disagreements,
                    config=config,
                )
            )

    agreement_count = len(linked_comparison) - len(resolution_queue)
    if len(resolution_queue) != int(config["expected_resolution_count"]):
        raise ValueError("真实分歧数量与配置不一致")
    if agreement_count != int(config["expected_full_agreement_count"]):
        raise ValueError("双轮完全一致数量与配置不一致")

    distributions = {
        "expected_disagreement_field_distribution": disagreement_fields_counter,
        "expected_pass1_decision_distribution": pass1_decisions,
        "expected_pass2_decision_distribution": pass2_decisions,
        "expected_decision_transition_distribution": decision_transitions,
        "expected_support_transition_distribution": support_transitions,
        "expected_evidence_transition_distribution": evidence_transitions,
    }
    for config_key, actual in distributions.items():
        _validate_expected_distribution(actual, config, config_key)

    summary = {
        "status": "resolution_queue_ready_author_adjudication_pending",
        "config_version": config["config_version"],
        "resolution_version": config["resolution_version"],
        "dataset_version": config["dataset_version"],
        "kb_version": config["kb_version"],
        "protocol_version": config["protocol_version"],
        "selection_count": len(selection_rows),
        "linked_count": len(linked_comparison),
        "full_agreement_count": agreement_count,
        "resolution_candidate_count": len(resolution_queue),
        "disagreement_field_distribution": dict(
            sorted(disagreement_fields_counter.items())
        ),
        "pass1_decision_distribution": dict(sorted(pass1_decisions.items())),
        "pass2_decision_distribution": dict(sorted(pass2_decisions.items())),
        "decision_transition_distribution": dict(sorted(decision_transitions.items())),
        "support_transition_distribution": dict(sorted(support_transitions.items())),
        "evidence_transition_distribution": dict(sorted(evidence_transitions.items())),
        "input_sha256": {
            "selection_draft": selection_sha256,
            "linkage": linkage_sha256,
            "pass2_queue": pass2_queue_sha256,
            "progress": progress_sha256,
        },
        "gold_promotion_performed": False,
        "freeze_performed": False,
        "usage": zero_usage(),
        "workflow_boundary": (
            "Linkage and disagreement detection only; all resolution fields remain "
            "blank; Benchmark-v1 is not Gold-promoted or frozen."
        ),
    }
    return {
        "linked_comparison": linked_comparison,
        "resolution_queue": resolution_queue,
        "summary": summary,
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 根节点必须是对象: {path}")
    return payload


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL 行必须是对象: {path}")
    return rows


def _load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def _atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding=encoding, newline="\n")
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    _atomic_write_text(path, content)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("不能写入空的 Benchmark120 待裁决队列")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write_text(path, buffer.getvalue(), encoding="utf-8-sig")


def _verify_input_hashes(config: dict[str, Any]) -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative_path, expected in sorted(config["input_sha256"].items()):
        path = ROOT / relative_path
        if not path.exists():
            raise ValueError(f"上游资产缺失: {relative_path}")
        digest = file_sha256(path)
        actual[relative_path] = digest
        if digest.lower() != _text(expected).lower():
            raise ValueError(
                f"上游资产哈希漂移: {relative_path}; "
                f"expected={expected}, actual={digest}"
            )
    return actual


def _resolution_guide(summary: dict[str, Any]) -> str:
    return f"""# Benchmark-120 双轮分歧裁决指南

## 当前范围

- 已关联记录：{summary['linked_count']} 条。
- 双轮完全一致：{summary['full_agreement_count']} 条。
- 待作者裁决：{summary['resolution_candidate_count']} 条。
- 当前状态：`{summary['status']}`。

## 裁决纪律

1. 同时阅读第一轮与第二轮字段、原始证据片段和差异字段。
2. 最终结论只能来自已定位证据或明确的医疗安全政策，不得补入背景记忆。
3. 逐条填写 `resolution_*` 字段，并保留具体裁决理由。
4. 完成全部裁决并通过一致性审计前，不得执行 Gold promotion 或 Test freeze。

## 当前边界

- Gold promotion：未执行。
- Benchmark freeze：未执行。
- 外部模型调用：0。
"""


def run(config_path: str | Path) -> dict[str, Any]:
    config = _load_json(config_path)
    _validate_config(config)
    actual_input_hashes = _verify_input_hashes(config)
    input_paths = config["input_paths"]

    selection_path = ROOT / input_paths["selection_draft"]
    linkage_path = ROOT / input_paths["linkage"]
    pass2_queue_path = ROOT / input_paths["pass2_queue"]
    progress_path = ROOT / input_paths["progress"]
    artifacts = build_resolution_artifacts(
        _load_jsonl(selection_path),
        _load_json(linkage_path),
        _load_csv(pass2_queue_path),
        _load_json(progress_path),
        config,
        selection_sha256=actual_input_hashes[input_paths["selection_draft"]],
        linkage_sha256=actual_input_hashes[input_paths["linkage"]],
        pass2_queue_sha256=actual_input_hashes[input_paths["pass2_queue"]],
        progress_sha256=actual_input_hashes[input_paths["progress"]],
    )

    output_dir = ROOT / config["output_dir"]
    outputs = config["outputs"]
    linked_path = output_dir / outputs["linked_comparison"]
    queue_path = output_dir / outputs["resolution_queue"]
    summary_path = output_dir / outputs["summary"]
    guide_path = output_dir / outputs["resolution_guide"]
    _write_jsonl(linked_path, artifacts["linked_comparison"])
    _write_csv(queue_path, artifacts["resolution_queue"])
    _atomic_write_text(guide_path, _resolution_guide(artifacts["summary"]))

    summary = dict(artifacts["summary"])
    summary["input_sha256_by_path"] = actual_input_hashes
    summary["output_sha256"] = {
        "linked_comparison": file_sha256(linked_path),
        "resolution_queue": file_sha256(queue_path),
        "resolution_guide": file_sha256(guide_path),
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Link Benchmark120 author passes and build a disagreement queue."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "configs"
            / "benchmark120_second_pass_resolution_v0_1.json"
        ),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
