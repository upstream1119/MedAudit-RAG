from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

ADJUDICATION_TO_GOLD_FIELDS = {
    "resolution_final_decision": "expected_decision",
    "resolution_final_kb_support": "current_kb_support",
    "resolution_final_gold_evidence_status": "gold_evidence_status",
    "resolution_final_required_evidence_type": "required_evidence_type",
    "resolution_final_required_claims": "required_claims",
    "resolution_final_allowed_claims": "allowed_claims",
    "resolution_final_forbidden_claims": "forbidden_claims",
    "resolution_final_missing_evidence_type": "missing_evidence_type",
    "resolution_final_missing_information": "missing_information",
    "resolution_final_risk_labels": "risk_labels",
}

LIST_RESOLUTION_FIELDS = {
    "resolution_final_required_evidence_type",
    "resolution_final_required_claims",
    "resolution_final_allowed_claims",
    "resolution_final_forbidden_claims",
    "resolution_final_missing_evidence_type",
    "resolution_final_missing_information",
    "resolution_final_risk_labels",
}

IMMUTABLE_PROVENANCE_FIELDS = (
    "candidate_id",
    "question",
    "source_id",
    "source_title",
    "source_filename",
    "source_sha256",
    "page_number",
    "anchor_text_span",
    "dataset_split",
)


def zero_usage() -> dict[str, int | float]:
    return {
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
    }


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _canonical_value(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return _text(value)


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _canonical_jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def compute_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index_unique(
    rows: list[dict[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate_id = _text(row.get("candidate_id"))
        if not candidate_id:
            raise ValueError(f"{label} 含空 candidate_id")
        if candidate_id in indexed:
            raise ValueError(f"{label} 含重复 candidate_id: {candidate_id}")
        indexed[candidate_id] = row
    return indexed


def _check_input_hashes(
    config: dict[str, Any], observed_input_sha256: dict[str, str]
) -> None:
    expected = {
        _text(key): _text(value).lower()
        for key, value in dict(config.get("expected_input_sha256", {})).items()
    }
    observed = {
        _text(key): _text(value).lower()
        for key, value in dict(observed_input_sha256).items()
    }
    if expected != observed:
        raise ValueError(
            f"input hash mismatch: expected={expected}, observed={observed}"
        )


def _parse_json_list(value: Any, *, field: str, candidate_id: str) -> list[Any]:
    if isinstance(value, list):
        parsed = value
    else:
        try:
            parsed = json.loads(_text(value))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{candidate_id} {field} 不是合法 JSON 数组") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{candidate_id} {field} 不是合法 JSON 数组")
    if any(not _text(item) for item in parsed):
        raise ValueError(f"{candidate_id} {field} 包含空值")
    canonical = [json.dumps(item, ensure_ascii=False, sort_keys=True) for item in parsed]
    if len(canonical) != len(set(canonical)):
        raise ValueError(f"{candidate_id} {field} 包含重复值")
    return parsed


def _validate_adjudication_row(
    selection_row: dict[str, Any], adjudication_row: dict[str, Any]
) -> None:
    candidate_id = _text(selection_row.get("candidate_id"))
    for field in IMMUTABLE_PROVENANCE_FIELDS:
        if _canonical_value(selection_row.get(field)) != _canonical_value(
            adjudication_row.get(field)
        ):
            raise ValueError(f"{candidate_id} {field} 发生来源或证据漂移")
    required_text_fields = (
        "resolution_reviewer_id",
        "resolution_annotator_role",
        "resolution_reviewed_at",
        "resolution_status",
        "resolution_final_decision",
        "resolution_final_kb_support",
        "resolution_final_gold_evidence_status",
        "resolution_reason",
    )
    missing = [
        field for field in required_text_fields if not _text(adjudication_row.get(field))
    ]
    if missing:
        raise ValueError(f"{candidate_id} 裁决字段不完整: {', '.join(missing)}")
    if _text(adjudication_row.get("resolution_status")) != "accepted":
        raise ValueError(f"{candidate_id} resolution_status 必须为 accepted")
    for field in LIST_RESOLUTION_FIELDS:
        _parse_json_list(
            adjudication_row.get(field), field=field, candidate_id=candidate_id
        )


def _promote_row(
    selection_row: dict[str, Any],
    adjudication_row: dict[str, Any] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    row = deepcopy(selection_row)
    if adjudication_row is not None:
        _validate_adjudication_row(selection_row, adjudication_row)
        for source_field, target_field in ADJUDICATION_TO_GOLD_FIELDS.items():
            value = adjudication_row.get(source_field)
            if source_field in LIST_RESOLUTION_FIELDS:
                value = _parse_json_list(
                    value,
                    field=source_field,
                    candidate_id=_text(selection_row.get("candidate_id")),
                )
            row[target_field] = value
        row["adjudication_reviewer_id"] = _text(
            adjudication_row.get("resolution_reviewer_id")
        )
        row["adjudication_annotator_role"] = _text(
            adjudication_row.get("resolution_annotator_role")
        )
        row["adjudication_reviewed_at"] = _text(
            adjudication_row.get("resolution_reviewed_at")
        )
        row["adjudication_reason"] = _text(
            adjudication_row.get("resolution_reason")
        )
        row["adjudication_status"] = "accepted"
    else:
        row["adjudication_status"] = "not_required"

    row["dataset_version"] = config["gold_version"]
    row["kb_version"] = config["kb_version"]
    row["freeze_version"] = config["freeze_version"]
    row["freeze_status"] = "frozen"
    row["split_status"] = "frozen"
    row["gold_promotion_status"] = "promoted"
    row["clinically_validated"] = False
    _canonicalize_boundary_policy(row)
    return row


def _canonicalize_boundary_policy(row: dict[str, Any]) -> None:
    if _text(row.get("expected_decision")) != "boundary_refusal":
        row["boundary_policy_normalization_status"] = "not_applicable"
        return
    evidence_types = row.get("required_evidence_type", [])
    if not isinstance(evidence_types, list):
        evidence_types = _parse_json_list(
            evidence_types,
            field="required_evidence_type",
            candidate_id=_text(row.get("candidate_id")),
        )
    normalized_types = [_text(value) for value in evidence_types]
    if (
        _text(row.get("current_kb_support")) == "policy_rule"
        and _text(row.get("gold_evidence_status")) == "page_span_located"
        and "medical_safety_policy" in normalized_types
        and "safety_policy" not in normalized_types
    ):
        row["gold_evidence_status"] = "policy_rule"
        row["required_evidence_type"] = [
            "safety_policy" if value == "medical_safety_policy" else value
            for value in normalized_types
        ]
        row["boundary_policy_normalization_status"] = "canonicalized"
        return
    if (
        _text(row.get("current_kb_support")) == "policy_rule"
        and _text(row.get("gold_evidence_status")) == "policy_rule"
        and "safety_policy" in normalized_types
    ):
        row["boundary_policy_normalization_status"] = "already_canonical"
        return
    row["boundary_policy_normalization_status"] = "unresolved"


def _check_boundary_policy(row: dict[str, Any]) -> None:
    if _text(row.get("expected_decision")) != "boundary_refusal":
        return
    evidence_types = row.get("required_evidence_type", [])
    if not isinstance(evidence_types, list):
        evidence_types = _parse_json_list(
            evidence_types,
            field="required_evidence_type",
            candidate_id=_text(row.get("candidate_id")),
        )
    if (
        _text(row.get("current_kb_support")) != "policy_rule"
        or _text(row.get("gold_evidence_status")) != "policy_rule"
        or "safety_policy" not in {_text(value) for value in evidence_types}
    ):
        raise ValueError(
            f"{row.get('candidate_id')} boundary_refusal 必须保持 policy_rule 依据"
        )


def _check_cross_split_leakage(
    validation_rows: list[dict[str, Any]], pilot_test_rows: list[dict[str, Any]]
) -> None:
    def values(rows: list[dict[str, Any]], field: str) -> set[str]:
        return {_canonical_value(row.get(field)) for row in rows if _text(row.get(field))}

    for field in (
        "independence_unit_id",
        "fact_cluster_id",
        "evidence_anchor_group_id",
    ):
        overlap = values(validation_rows, field) & values(pilot_test_rows, field)
        if overlap:
            raise ValueError(f"validation/pilot_test 存在 {field} 泄漏: {sorted(overlap)}")

    validation_questions = {
        " ".join(_text(row.get("question")).lower().split()) for row in validation_rows
    }
    pilot_questions = {
        " ".join(_text(row.get("question")).lower().split()) for row in pilot_test_rows
    }
    question_overlap = validation_questions & pilot_questions
    if question_overlap:
        raise ValueError("validation/pilot_test 存在相同问题文本泄漏")

    validation_sources = {
        (_text(row.get("source_id")), _text(row.get("page_number")))
        for row in validation_rows
    }
    pilot_sources = {
        (_text(row.get("source_id")), _text(row.get("page_number")))
        for row in pilot_test_rows
    }
    source_overlap = validation_sources & pilot_sources
    if source_overlap:
        raise ValueError(
            f"validation/pilot_test 存在相同 source/page 证据泄漏: {sorted(source_overlap)}"
        )


def build_gold_freeze(
    selection_rows: list[dict[str, Any]],
    adjudication_rows: list[dict[str, Any]],
    split_payload: dict[str, Any],
    config: dict[str, Any],
    observed_input_sha256: dict[str, str],
) -> dict[str, Any]:
    _check_input_hashes(config, observed_input_sha256)
    if int(config.get("external_model_calls", 0)) != 0:
        raise ValueError("Gold freeze 阶段禁止外部模型调用")

    expected_selection = int(config["expected_selection_count"])
    if len(selection_rows) != expected_selection:
        raise ValueError(
            f"selection 数量不符: expected={expected_selection}, actual={len(selection_rows)}"
        )
    selection_index = _index_unique(selection_rows, label="selection")
    adjudication_index = _index_unique(adjudication_rows, label="adjudication")
    required_adjudication_ids = {
        candidate_id
        for candidate_id, row in selection_index.items()
        if bool(row.get("requires_second_pass"))
    }
    expected_adjudication = int(config["expected_adjudication_count"])
    if (
        len(adjudication_rows) != expected_adjudication
        or set(adjudication_index) != required_adjudication_ids
    ):
        raise ValueError(
            "第二轮裁决未完整覆盖 required_second_pass / adjudication coverage drift"
        )

    gold_rows = [
        _promote_row(
            selection_row,
            adjudication_index.get(candidate_id),
            config,
        )
        for candidate_id, selection_row in selection_index.items()
    ]
    gold_index = {row["candidate_id"]: row for row in gold_rows}
    for row in gold_rows:
        _check_boundary_policy(row)
    boundary_policy_normalized_candidate_ids = [
        row["candidate_id"]
        for row in gold_rows
        if row.get("boundary_policy_normalization_status") == "canonicalized"
    ]

    validation_ids = [
        _text(value) for value in split_payload.get("validation_candidate_ids", [])
    ]
    pilot_test_ids = [
        _text(value) for value in split_payload.get("pilot_test_candidate_ids", [])
    ]
    if len(validation_ids) != len(set(validation_ids)) or len(pilot_test_ids) != len(
        set(pilot_test_ids)
    ):
        raise ValueError("split 中含重复 candidate_id")
    if set(validation_ids) & set(pilot_test_ids):
        raise ValueError("validation/pilot_test candidate_id 泄漏")
    if set(validation_ids) | set(pilot_test_ids) != set(gold_index):
        raise ValueError("split 未完整且唯一覆盖 selection")

    expected_split_counts = dict(config["expected_split_counts"])
    if len(validation_ids) != int(expected_split_counts["validation"]):
        raise ValueError("validation split 数量不符")
    if len(pilot_test_ids) != int(expected_split_counts["pilot_test"]):
        raise ValueError("pilot_test split 数量不符")

    validation_rows = [deepcopy(gold_index[value]) for value in validation_ids]
    pilot_test_rows = [deepcopy(gold_index[value]) for value in pilot_test_ids]
    for row in validation_rows:
        if _text(row.get("dataset_split")) != "validation":
            raise ValueError(f"{row['candidate_id']} dataset_split 与冻结拆分不一致")
    for row in pilot_test_rows:
        if _text(row.get("dataset_split")) != "pilot_test":
            raise ValueError(f"{row['candidate_id']} dataset_split 与冻结拆分不一致")
    _check_cross_split_leakage(validation_rows, pilot_test_rows)

    summary = {
        "status": "guideline_grounded_gold_frozen",
        "config_version": config["config_version"],
        "gold_version": config["gold_version"],
        "freeze_version": config["freeze_version"],
        "kb_version": config["kb_version"],
        "protocol_version": config["protocol_version"],
        "selection_count": len(gold_rows),
        "adjudication_count": len(adjudication_rows),
        "split_counts": {
            "validation": len(validation_rows),
            "pilot_test": len(pilot_test_rows),
        },
        "decision_distribution": dict(
            sorted(Counter(_text(row.get("expected_decision")) for row in gold_rows).items())
        ),
        "gold_evidence_status_distribution": dict(
            sorted(Counter(_text(row.get("gold_evidence_status")) for row in gold_rows).items())
        ),
        "boundary_policy_normalization_count": len(
            boundary_policy_normalized_candidate_ids
        ),
        "boundary_policy_normalized_candidate_ids": (
            boundary_policy_normalized_candidate_ids
        ),
        "input_sha256": dict(observed_input_sha256),
        "gold_promotion_performed": True,
        "freeze_performed": True,
        "clinically_validated": False,
        "usage": zero_usage(),
        "workflow_boundary": (
            "Guideline-grounded and author-adjudicated benchmark only; no independent "
            "clinical expert validation or real-world clinical effectiveness claim."
        ),
    }
    return {
        "gold_rows": gold_rows,
        "validation_rows": validation_rows,
        "pilot_test_rows": pilot_test_rows,
        "summary": summary,
    }


def _summary_markdown(summary: dict[str, Any]) -> str:
    return f"""# Benchmark120 Gold Freeze 摘要

## 冻结结果

- 状态：`{summary['status']}`
- Gold 总数：{summary['selection_count']}
- Validation：{summary['split_counts']['validation']}
- Pilot Test：{summary['split_counts']['pilot_test']}
- 作者裁决：{summary['adjudication_count']}
- 旧边界政策枚举规范化：{summary['boundary_policy_normalization_count']}
- 外部模型调用：0

## 研究边界

该版本是基于公开权威资料、可定位证据和作者裁决形成的 guideline-grounded benchmark。
`clinically_validated=false`：未经过独立临床专家验证，不构成临床有效性或真实世界安全性证明。
"""


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    temporary.replace(path)


def write_frozen_outputs(
    result: dict[str, Any], output_dir: str | Path
) -> dict[str, dict[str, str]]:
    output_dir = Path(output_dir)
    gold_bytes = _canonical_jsonl_bytes(result["gold_rows"])
    validation_bytes = _canonical_jsonl_bytes(result["validation_rows"])
    pilot_bytes = _canonical_jsonl_bytes(result["pilot_test_rows"])
    data_hashes = {
        "benchmark120_gold_v1_0.jsonl": _sha256_bytes(gold_bytes),
        "benchmark120_validation40_v1_0.jsonl": _sha256_bytes(validation_bytes),
        "benchmark120_pilot_test80_v1_0.jsonl": _sha256_bytes(pilot_bytes),
    }
    audit_payload = {
        **result["summary"],
        "frozen_data_sha256": data_hashes,
        "immutable_outputs": True,
    }
    audit_bytes = _canonical_json_bytes(audit_payload)
    summary_bytes = _summary_markdown(result["summary"]).encode("utf-8")
    manifest_payload = {
        "status": result["summary"]["status"],
        "gold_version": result["summary"]["gold_version"],
        "freeze_version": result["summary"]["freeze_version"],
        "clinically_validated": False,
        "files_sha256": {
            **data_hashes,
            "benchmark120_gold_promotion_audit_v1_0.json": _sha256_bytes(audit_bytes),
            "benchmark120_gold_summary_v1_0.md": _sha256_bytes(summary_bytes),
        },
        "usage": zero_usage(),
    }
    manifest_bytes = _canonical_json_bytes(manifest_payload)
    contents = {
        "benchmark120_gold_v1_0.jsonl": gold_bytes,
        "benchmark120_validation40_v1_0.jsonl": validation_bytes,
        "benchmark120_pilot_test80_v1_0.jsonl": pilot_bytes,
        "benchmark120_gold_promotion_audit_v1_0.json": audit_bytes,
        "benchmark120_freeze_manifest_v1_0.json": manifest_bytes,
        "benchmark120_gold_summary_v1_0.md": summary_bytes,
    }

    # Preflight every path before writing so a mismatch cannot leave a partial freeze.
    for name, content in contents.items():
        path = output_dir / name
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"冻结资产 immutable，拒绝覆盖: {path}")
    for name, content in contents.items():
        path = output_dir / name
        if not path.exists():
            _atomic_write_bytes(path, content)
    return {
        name: {"path": str(output_dir / name), "sha256": _sha256_bytes(content)}
        for name, content in contents.items()
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


def _load_adjudication(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl(path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"裁决资产仅支持 CSV 或 JSONL: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"裁决 CSV 为空: {path}")
    return rows


def validate_adjudication_audit(
    audit: dict[str, Any],
    config: dict[str, Any],
    *,
    adjudication_sha256: str,
) -> None:
    if _text(audit.get("status")) != "author_adjudication_complete":
        raise ValueError("裁决审计状态不是 author_adjudication_complete")
    if int(audit.get("adjudicated_count", -1)) != int(
        config["expected_adjudication_count"]
    ):
        raise ValueError("裁决审计数量与 freeze 配置不一致")
    if _text(audit.get("adjudication_csv_sha256")).lower() != _text(
        adjudication_sha256
    ).lower():
        raise ValueError("裁决审计记录的 CSV hash 与实际裁决资产不一致")
    for field in (
        "gold_promotion_performed",
        "freeze_performed",
        "clinically_validated",
    ):
        if audit.get(field) is not False:
            raise ValueError(f"裁决审计状态非法: {field} 必须为 false")


def run(config_path: str | Path) -> dict[str, Any]:
    config = _load_json(config_path)
    input_paths = dict(config["input_paths"])
    observed_hashes: dict[str, str] = {}
    resolved_paths: dict[str, Path] = {}
    for key, relative_path in input_paths.items():
        path = ROOT / relative_path
        if not path.exists():
            raise ValueError(f"冻结上游资产缺失: {relative_path}")
        resolved_paths[key] = path
        observed_hashes[key] = compute_sha256(path)
    adjudication_audit = _load_json(resolved_paths["adjudication_audit"])
    validate_adjudication_audit(
        adjudication_audit,
        config,
        adjudication_sha256=observed_hashes["adjudication"],
    )
    result = build_gold_freeze(
        _load_jsonl(resolved_paths["selection"]),
        _load_adjudication(resolved_paths["adjudication"]),
        _load_json(resolved_paths["split"]),
        config,
        observed_hashes,
    )
    outputs = write_frozen_outputs(result, ROOT / config["output_dir"])
    return {**result["summary"], "outputs": outputs}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote an author-adjudicated Benchmark120 and freeze exact splits."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
