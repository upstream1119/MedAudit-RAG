from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

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

VISIBLE_SOURCE_FIELDS = (
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
)

BLINDED_FORBIDDEN_FIELDS = {
    "candidate_id",
    "dataset_split",
    "expected_decision",
    "current_kb_support",
    "gold_evidence_status",
    "required_evidence_type",
    "required_claims",
    "allowed_claims",
    "forbidden_claims",
    "missing_evidence_type",
    "missing_information",
    "risk_labels",
    "selection_version",
    "origin_pool",
    "fact_cluster_id",
    "evidence_anchor_group_id",
    "evidence_anchor_ids",
    "independence_unit_id",
    "source_row_sha256",
    "annotation_pass_count",
    "requires_second_pass",
    "candidate_status",
    "split_status",
    "freeze_status",
    "evidence_scope",
    "age_scope",
    "applicability_conditions",
    "supported_claim_types",
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
        return value
    try:
        parsed = json.loads(str(value or "[]"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} 必须是 JSON 数组") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{field_name} 必须是 JSON 数组")
    return parsed


def _clean_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _assert_clean_text(value: Any, *, context: str) -> None:
    text = json.dumps(value, ensure_ascii=False)
    if "�" in text or "???" in text:
        raise ValueError(f"{context} 包含乱码")


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "config_version",
        "annotation_version",
        "dataset_version",
        "kb_version",
        "protocol_version",
        "blind_seed",
        "expected_selection_count",
        "expected_second_pass_count",
        "expected_pending_candidate_status",
        "expected_split_distribution",
        "expected_first_pass_decision_distribution",
        "batch_sizes",
        "policy_rules",
        "external_model_calls",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"第二轮队列配置缺少字段: {', '.join(missing)}")
    if int(config["external_model_calls"]) != 0:
        raise ValueError("第二轮作者盲审队列禁止调用外部模型")
    batch_sizes = [int(value) for value in config["batch_sizes"]]
    if any(value <= 0 for value in batch_sizes):
        raise ValueError("第二轮批次大小必须为正整数")
    if sum(batch_sizes) != int(config["expected_second_pass_count"]):
        raise ValueError("第二轮批次大小之和与待二审数量不一致")


def _blind_id(candidate_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}|{candidate_id}".encode("utf-8")).hexdigest()
    return f"B120-P2-{digest[:16]}"


def _policy_evidence(
    policy_rule_ids: list[Any],
    *,
    policy_text: str,
    policy_rules: dict[str, str],
) -> str:
    evidence: list[str] = []
    for raw_rule_id in policy_rule_ids:
        rule_id = _text(raw_rule_id)
        if rule_id not in policy_rules:
            raise ValueError(f"未知安全政策规则: {rule_id}")
        rule_text = _text(policy_rules[rule_id])
        if not rule_text or rule_text not in policy_text:
            raise ValueError(f"政策原文无法验证: {rule_id}")
        evidence.append(rule_text)
    return "\n".join(evidence)


def _validate_pending_row(row: dict[str, Any], config: dict[str, Any]) -> None:
    candidate_id = _text(row.get("candidate_id"))
    if not candidate_id:
        raise ValueError("待二审记录缺少 candidate_id")
    if int(row.get("annotation_pass_count", 0)) != 1:
        raise ValueError(f"待二审记录不是单轮状态: {candidate_id}")
    if row.get("requires_second_pass") is not True:
        raise ValueError(f"待二审标记不一致: {candidate_id}")
    if _text(row.get("candidate_status")) != config["expected_pending_candidate_status"]:
        raise ValueError(f"待二审候选状态不一致: {candidate_id}")
    if _text(row.get("freeze_status")) != "draft":
        raise ValueError(f"待二审记录已被错误冻结: {candidate_id}")
    if _text(row.get("gold_status")):
        raise ValueError(f"待二审记录已被错误晋升 Gold: {candidate_id}")

    required_evidence_fields = (
        "question",
        "source_id",
        "source_title",
        "source_filename",
        "source_sha256",
        "page_number",
        "anchor_text_span",
    )
    missing = [field for field in required_evidence_fields if not _text(row.get(field))]
    if missing:
        raise ValueError(
            f"待二审记录缺少证据字段: {candidate_id}; {', '.join(missing)}"
        )
    try:
        page_number = int(row["page_number"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"待二审记录页码非法: {candidate_id}") from exc
    if page_number <= 0:
        raise ValueError(f"待二审记录页码非法: {candidate_id}")
    _assert_clean_text(row, context=f"待二审记录 {candidate_id}")


def _make_visible_row(
    row: dict[str, Any],
    *,
    order: int,
    item_id: str,
    policy_text: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    visible: dict[str, Any] = {
        "pass2_order": order,
        "pass2_item_id": item_id,
        "pass2_annotation_version": config["annotation_version"],
        "dataset_version": config["dataset_version"],
        "kb_version": config["kb_version"],
        "protocol_version": config["protocol_version"],
    }
    for field in VISIBLE_SOURCE_FIELDS:
        visible[field] = row.get(field, "")

    policy_rule_ids = _json_list(
        row.get("policy_rule_ids", []),
        field_name="policy_rule_ids",
    )
    visible["policy_rule_ids"] = json.dumps(
        policy_rule_ids,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    visible["policy_evidence_text"] = _policy_evidence(
        policy_rule_ids,
        policy_text=policy_text,
        policy_rules=config["policy_rules"],
    )
    for field in PASS2_EMPTY_FIELDS:
        visible[field] = ""

    leaked = sorted(set(visible) & BLINDED_FORBIDDEN_FIELDS)
    if leaked:
        raise ValueError(f"评审可见队列泄露第一轮字段: {', '.join(leaked)}")
    if _text(row["candidate_id"]) in json.dumps(visible, ensure_ascii=False):
        raise ValueError("评审可见队列泄露 candidate_id")
    _assert_clean_text(visible, context=f"盲化记录 {item_id}")
    return visible


def _make_linkage_row(
    row: dict[str, Any],
    *,
    item_id: str,
) -> dict[str, Any]:
    return {
        "pass2_item_id": item_id,
        "candidate_id": row["candidate_id"],
        "question": row["question"],
        "dataset_split": row["dataset_split"],
        "origin_pool": row.get("origin_pool", ""),
        "selection_version": row.get("selection_version", ""),
        "source_id": row["source_id"],
        "page_number": row["page_number"],
        "source_row_sha256": row.get("source_row_sha256", ""),
        "selection_row_sha256": _canonical_sha256(row),
        "fact_cluster_id": row.get("fact_cluster_id", ""),
        "evidence_anchor_group_id": row.get("evidence_anchor_group_id", ""),
        "evidence_anchor_ids": _clean_copy(row.get("evidence_anchor_ids", [])),
        "independence_unit_id": row.get("independence_unit_id", ""),
        "first_pass_expected_decision": row["expected_decision"],
        "first_pass_current_kb_support": row.get("current_kb_support", ""),
        "first_pass_gold_evidence_status": row.get("gold_evidence_status", ""),
        "first_pass_required_evidence_type": _clean_copy(
            row.get("required_evidence_type", [])
        ),
        "first_pass_required_claims": _clean_copy(row.get("required_claims", [])),
        "first_pass_allowed_claims": _clean_copy(row.get("allowed_claims", [])),
        "first_pass_forbidden_claims": _clean_copy(row.get("forbidden_claims", [])),
        "first_pass_missing_evidence_type": _clean_copy(
            row.get("missing_evidence_type", [])
        ),
        "first_pass_missing_information": _clean_copy(
            row.get("missing_information", [])
        ),
        "first_pass_risk_labels": _clean_copy(row.get("risk_labels", [])),
    }


def _build_batches(
    review_queue: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    start = 0
    for batch_index, size in enumerate(config["batch_sizes"], start=1):
        size = int(size)
        records = _clean_copy(review_queue[start : start + size])
        first_order = start + 1
        last_order = start + size
        batches.append(
            {
                "annotation_version": config["annotation_version"],
                "batch_id": f"phase7-b3.6j-benchmark120-pass2-batch{batch_index:02d}",
                "batch_scope": f"pass2_order {first_order}-{last_order}",
                "dataset_version": config["dataset_version"],
                "kb_version": config["kb_version"],
                "protocol_version": config["protocol_version"],
                "record_count": len(records),
                "review_mode": "same-author row-level first-pass-label-blinded second review",
                "external_model_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost": 0,
                "records": records,
            }
        )
        start += size
    if start != len(review_queue):
        raise ValueError("第二轮批次未完整覆盖评审队列")
    return batches


def build_second_pass_artifacts(
    selection_rows: list[dict[str, Any]],
    policy_text: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """构建与第一轮逐条结论隔离的同一作者第二轮复核队列。"""
    _validate_config(config)
    if len(selection_rows) != int(config["expected_selection_count"]):
        raise ValueError("Benchmark 选择草案数量与配置不一致")

    candidate_ids = [_text(row.get("candidate_id")) for row in selection_rows]
    if not all(candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Benchmark 选择草案存在空或重复 candidate_id")

    for rule_id, rule_text in config["policy_rules"].items():
        if not _text(rule_id) or _text(rule_text) not in policy_text:
            raise ValueError(f"政策原文无法验证: {rule_id}")

    pending = [row for row in selection_rows if row.get("requires_second_pass") is True]
    if len(pending) != int(config["expected_second_pass_count"]):
        raise ValueError("Benchmark 待二审数量与配置不一致")
    for row in pending:
        _validate_pending_row(row, config)

    split_distribution = Counter(_text(row["dataset_split"]) for row in pending)
    if dict(sorted(split_distribution.items())) != dict(
        sorted(config["expected_split_distribution"].items())
    ):
        raise ValueError("待二审 split 分布与配置不一致")
    decision_distribution = Counter(_text(row["expected_decision"]) for row in pending)
    if dict(sorted(decision_distribution.items())) != dict(
        sorted(config["expected_first_pass_decision_distribution"].items())
    ):
        raise ValueError("待二审第一轮决策分布与配置不一致")

    ordered = sorted(pending, key=lambda row: _text(row["candidate_id"]))
    random.Random(int(config["blind_seed"])).shuffle(ordered)
    review_queue: list[dict[str, Any]] = []
    linkage_records: list[dict[str, Any]] = []
    for order, row in enumerate(ordered, start=1):
        item_id = _blind_id(_text(row["candidate_id"]), int(config["blind_seed"]))
        review_queue.append(
            _make_visible_row(
                row,
                order=order,
                item_id=item_id,
                policy_text=policy_text,
                config=config,
            )
        )
        linkage_records.append(_make_linkage_row(row, item_id=item_id))

    review_ids = [row["pass2_item_id"] for row in review_queue]
    linkage_ids = [row["pass2_item_id"] for row in linkage_records]
    if len(review_ids) != len(set(review_ids)) or set(review_ids) != set(linkage_ids):
        raise ValueError("第二轮 blind ID 与 linkage 映射不一致")

    batches = _build_batches(review_queue, config)
    summary = {
        "status": "second_pass_queue_ready_review_pending",
        "selection_count": len(selection_rows),
        "second_pass_count": len(review_queue),
        "source_count": len({_text(row["source_id"]) for row in pending}),
        "split_distribution": dict(sorted(split_distribution.items())),
        "first_pass_decision_distribution": dict(
            sorted(decision_distribution.items())
        ),
        "batch_sizes": [batch["record_count"] for batch in batches],
        "review_mode": (
            "same-author row-level first-pass-label-blinded second review; "
            "not independent expert review"
        ),
        "gold_promotion_performed": False,
        "freeze_performed": False,
        "usage": {
            "external_model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost": 0,
        },
        "workflow_boundary": (
            "Queue generation only; second review remains pending; "
            "Benchmark-v1 is not Gold-promoted or frozen."
        ),
    }
    return {
        "review_queue": review_queue,
        "linkage_records": linkage_records,
        "batches": batches,
        "summary": summary,
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 根节点必须是对象: {path}")
    return payload


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL 行必须是对象: {path}")
    return rows


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("不能写入空的第二轮队列")
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


def _review_guide(config: dict[str, Any], summary: dict[str, Any]) -> str:
    return f"""# Benchmark-120 第二轮作者盲审指南

## 当前范围

- 待复审记录：{summary['second_pass_count']} 条。
- 批次：{' / '.join(str(value) for value in summary['batch_sizes'])}。
- 评审模式：同一作者在看不到逐条第一轮结论、理由和 split 的条件下进行第二遍审核。
- 该流程不是独立专家评审、临床验证或医生共识。

## 评审纪律

1. 复审期间只打开 reviewer-visible CSV 或当前批次 JSON。
2. 不得打开 linkage manifest、选择草案或选择审计文件。
3. 仅依据当前问题、来源、页码、证据片段和已显示的安全政策作出判断。
4. 证据不足时保留 `review_required` 或 `insufficient_evidence`，不得补写背景记忆。
5. 每批完成后单独校验，再进行后续关联与分歧裁决。

## 当前边界

- 状态：`second_pass_queue_ready_review_pending`
- Gold promotion：未执行
- Benchmark freeze：未执行
- 外部模型调用：0
- 配置版本：`{config['config_version']}`
"""


def run(config_path: str | Path) -> dict[str, Any]:
    config = _load_json(config_path)
    _validate_config(config)
    actual_input_hashes = _verify_input_hashes(config)

    selection_path = ROOT / config["input_paths"]["selection_draft"]
    audit_path = ROOT / config["input_paths"]["selection_audit"]
    policy_path = ROOT / config["input_paths"]["medical_safety_policy"]
    selection_audit = _load_json(audit_path)
    if selection_audit.get("status") != "draft_selection_ready_for_second_pass":
        raise ValueError("Benchmark120 选择审计状态不允许生成第二轮队列")
    if selection_audit.get("gold_promotion_performed") is not False:
        raise ValueError("选择审计错误标记了 Gold promotion")
    if selection_audit.get("freeze_performed") is not False:
        raise ValueError("选择审计错误标记了 freeze")

    artifacts = build_second_pass_artifacts(
        _load_jsonl(selection_path),
        policy_path.read_text(encoding="utf-8"),
        config,
    )
    output_dir = ROOT / config["output_dir"]
    queue_path = output_dir / config["outputs"]["review_queue"]
    linkage_path = output_dir / config["outputs"]["linkage"]
    audit_output_path = output_dir / config["outputs"]["generation_audit"]
    guide_path = output_dir / config["outputs"]["review_guide"]
    batch_dir = output_dir / config["outputs"]["batch_dir"]

    _write_csv(queue_path, artifacts["review_queue"])
    _write_json(
        linkage_path,
        {
            "linkage_version": config["annotation_version"],
            "review_access": "do_not_open_during_second_pass_review",
            "record_count": len(artifacts["linkage_records"]),
            "records": artifacts["linkage_records"],
        },
    )
    batch_paths: list[Path] = []
    for batch_index, batch in enumerate(artifacts["batches"], start=1):
        batch_path = batch_dir / f"batch{batch_index:02d}_template_v0_1.json"
        _write_json(batch_path, batch)
        batch_paths.append(batch_path)
    _atomic_write_text(guide_path, _review_guide(config, artifacts["summary"]))

    output_hashes = {
        "review_queue": file_sha256(queue_path),
        "linkage": file_sha256(linkage_path),
        "review_guide": file_sha256(guide_path),
        "batches": {
            path.name: file_sha256(path)
            for path in batch_paths
        },
    }
    generation_audit = {
        "config_version": config["config_version"],
        "annotation_version": config["annotation_version"],
        "dataset_version": config["dataset_version"],
        "kb_version": config["kb_version"],
        "protocol_version": config["protocol_version"],
        "input_sha256": actual_input_hashes,
        "output_sha256": output_hashes,
        "blinding_checks": {
            "review_queue_row_count": len(artifacts["review_queue"]),
            "linkage_row_count": len(artifacts["linkage_records"]),
            "one_to_one_mapping": True,
            "forbidden_visible_field_count": 0,
            "blank_review_field_count": sum(
                all(row[field] == "" for field in PASS2_EMPTY_FIELDS)
                for row in artifacts["review_queue"]
            ),
        },
        **artifacts["summary"],
    }
    _write_json(audit_output_path, generation_audit)
    return generation_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Benchmark-120 blinded second-pass review queue."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "configs"
            / "benchmark120_second_pass_queue_v0_1.json"
        ),
    )
    args = parser.parse_args()
    report = run(args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
