from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

RESOLUTION_AUTHOR_FIELDS = (
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

RESOLUTION_LIST_FIELDS = (
    "resolution_final_required_evidence_type",
    "resolution_final_required_claims",
    "resolution_final_allowed_claims",
    "resolution_final_forbidden_claims",
    "resolution_final_missing_evidence_type",
    "resolution_final_missing_information",
    "resolution_final_risk_labels",
)

REQUIRED_NONEMPTY_LIST_FIELDS = (
    "resolution_final_required_evidence_type",
    "resolution_final_required_claims",
    "resolution_final_allowed_claims",
    "resolution_final_forbidden_claims",
    "resolution_final_risk_labels",
)

REQUIRED_CONFIG_FIELDS = {
    "config_version",
    "adjudication_version",
    "dataset_version",
    "kb_version",
    "protocol_version",
    "expected_candidate_count",
    "batch_sizes",
    "allowed_resolution_status",
    "allowed_decisions",
    "allowed_kb_support",
    "allowed_gold_evidence_status",
    "expected_input_sha256",
}


def zero_usage() -> dict[str, int | float]:
    return {
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
    }


def compute_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("分歧裁决配置根节点必须是 JSON 对象")
    missing = sorted(REQUIRED_CONFIG_FIELDS - set(config))
    if missing:
        raise ValueError(f"分歧裁决配置缺少字段: {', '.join(missing)}")
    if int(config.get("external_model_calls", 0)) != 0:
        raise ValueError("分歧裁决阶段禁止配置外部模型调用")
    return config


def _check_input_hashes(
    config: dict[str, Any], observed_input_sha256: dict[str, str]
) -> None:
    expected = {
        str(key): str(value).lower()
        for key, value in dict(config["expected_input_sha256"]).items()
    }
    observed = {
        str(key): str(value).lower()
        for key, value in dict(observed_input_sha256).items()
    }
    if observed != expected:
        raise ValueError(
            f"input hash mismatch: expected={expected}, observed={observed}"
        )


def _text(value: Any) -> str:
    return str(value or "").strip()


def _serialized(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value or "")


def _index_unique(
    rows: list[dict[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    candidate_ids = [_text(row.get("candidate_id")) for row in rows]
    if not all(candidate_ids):
        raise ValueError(f"{label} 含空 candidate_id")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"{label} 含重复 candidate_id")
    return {candidate_id: row for candidate_id, row in zip(candidate_ids, rows)}


def _parse_json_list(
    value: Any,
    *,
    field: str,
    candidate_id: str,
    require_nonempty: bool,
) -> list[Any]:
    if isinstance(value, list):
        parsed = value
    else:
        try:
            parsed = json.loads(str(value))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{candidate_id} {field} 不是合法 JSON 数组") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{candidate_id} {field} 不是合法 JSON 数组")
    if require_nonempty and not parsed:
        raise ValueError(f"{candidate_id} {field} 必须是非空 JSON 数组")
    if any(not _text(item) for item in parsed):
        raise ValueError(f"{candidate_id} {field} 包含空值")
    if len([json.dumps(item, ensure_ascii=False, sort_keys=True) for item in parsed]) != len(
        set(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in parsed)
    ):
        raise ValueError(f"{candidate_id} {field} 包含重复值")
    return parsed


def prepare_adjudication_batches(
    resolution_rows: list[dict[str, Any]],
    config: dict[str, Any],
    observed_input_sha256: dict[str, str],
) -> list[dict[str, Any]]:
    _check_input_hashes(config, observed_input_sha256)
    expected = int(config["expected_candidate_count"])
    if len(resolution_rows) != expected:
        raise ValueError(
            f"分歧裁决必须完整覆盖 {expected} 条；当前为 {len(resolution_rows)} 条"
        )
    indexed = _index_unique(resolution_rows, label="resolution queue")
    rows: list[dict[str, Any]] = []
    for candidate_id, source in indexed.items():
        prefilled = [
            field
            for field in RESOLUTION_AUTHOR_FIELDS
            if _text(source.get(field))
        ]
        if prefilled:
            raise ValueError(
                f"{candidate_id} 作者裁决字段存在预填非空值: {', '.join(prefilled)}"
            )
        row = dict(source)
        for field in RESOLUTION_AUTHOR_FIELDS:
            row[field] = ""
        rows.append(row)

    def sort_key(row: dict[str, Any]) -> tuple[int, str]:
        try:
            order = int(_text(row.get("resolution_order")))
        except ValueError as exc:
            raise ValueError(
                f"{row.get('candidate_id', '')} resolution_order 非整数"
            ) from exc
        return order, _text(row.get("candidate_id"))

    rows.sort(key=sort_key)
    batch_sizes = [int(value) for value in config["batch_sizes"]]
    if any(value <= 0 for value in batch_sizes):
        raise ValueError("batch_sizes 必须全部大于 0")
    if sum(batch_sizes) != expected:
        raise ValueError("batch_sizes 总数必须等于待裁决数量")
    batches: list[dict[str, Any]] = []
    offset = 0
    for number, size in enumerate(batch_sizes, start=1):
        batches.append(
            {
                "batch_id": f"batch_{number:02d}",
                "config_version": config["config_version"],
                "adjudication_version": config["adjudication_version"],
                "dataset_version": config["dataset_version"],
                "kb_version": config["kb_version"],
                "protocol_version": config["protocol_version"],
                "input_sha256": dict(observed_input_sha256),
                "rows": [dict(row) for row in rows[offset : offset + size]],
            }
        )
        offset += size
    return batches


def _validate_resolution_row(
    row: dict[str, Any], config: dict[str, Any]
) -> None:
    candidate_id = _text(row.get("candidate_id"))
    required_text = (
        "resolution_reviewer_id",
        "resolution_annotator_role",
        "resolution_reviewed_at",
        "resolution_status",
        "resolution_final_decision",
        "resolution_final_kb_support",
        "resolution_final_gold_evidence_status",
        "resolution_reason",
    )
    missing = [field for field in required_text if not _text(row.get(field))]
    if missing:
        raise ValueError(f"{candidate_id} 裁决缺少字段: {', '.join(missing)}")
    if _text(row["resolution_status"]) not in set(
        config["allowed_resolution_status"]
    ):
        raise ValueError(f"{candidate_id} resolution_status 非法")
    if _text(row["resolution_final_decision"]) not in set(
        config["allowed_decisions"]
    ):
        raise ValueError(f"{candidate_id} resolution_final_decision 非法")
    if _text(row["resolution_final_kb_support"]) not in set(
        config["allowed_kb_support"]
    ):
        raise ValueError(f"{candidate_id} resolution_final_kb_support 非法")
    if _text(row["resolution_final_gold_evidence_status"]) not in set(
        config["allowed_gold_evidence_status"]
    ):
        raise ValueError(f"{candidate_id} resolution_final_gold_evidence_status 非法")
    parsed_lists = {
        field: _parse_json_list(
            row.get(field),
            field=field,
            candidate_id=candidate_id,
            require_nonempty=field in REQUIRED_NONEMPTY_LIST_FIELDS,
        )
        for field in RESOLUTION_LIST_FIELDS
    }
    if _text(row["resolution_final_decision"]) == "boundary_refusal":
        required_evidence = {_text(value) for value in parsed_lists[
            "resolution_final_required_evidence_type"
        ]}
        if (
            _text(row["resolution_final_kb_support"]) != "policy_rule"
            or _text(row["resolution_final_gold_evidence_status"]) != "policy_rule"
            or "safety_policy" not in required_evidence
        ):
            raise ValueError(
                f"{candidate_id} boundary_refusal 必须使用 policy_rule 支持、"
                "policy_rule 证据状态和 safety_policy 证据类型"
            )


def validate_adjudication_batch(
    author_rows: list[dict[str, Any]],
    prepared_rows: list[dict[str, Any]],
    config: dict[str, Any],
    observed_input_sha256: dict[str, str],
) -> list[dict[str, Any]]:
    _check_input_hashes(config, observed_input_sha256)
    prepared = _index_unique(prepared_rows, label="prepared adjudication rows")
    authors = _index_unique(author_rows, label="author adjudication rows")
    unknown = sorted(set(authors) - set(prepared))
    if unknown:
        raise ValueError(f"裁决包含未知 candidate_id: {', '.join(unknown)}")
    validated: list[dict[str, Any]] = []
    for candidate_id, row in authors.items():
        source = prepared[candidate_id]
        drifted = [
            field
            for field, value in source.items()
            if field not in RESOLUTION_AUTHOR_FIELDS
            and _serialized(row.get(field, "")) != _serialized(value)
        ]
        if drifted:
            raise ValueError(
                f"{candidate_id} 裁决不可变字段发生漂移: {', '.join(drifted)}"
            )
        _validate_resolution_row(row, config)
        validated.append(dict(row))
    return sorted(
        validated,
        key=lambda row: (int(_text(row["resolution_order"])), row["candidate_id"]),
    )


def finalize_adjudications(
    author_rows: list[dict[str, Any]],
    prepared_rows: list[dict[str, Any]],
    config: dict[str, Any],
    observed_input_sha256: dict[str, str],
) -> dict[str, Any]:
    expected = int(config["expected_candidate_count"])
    prepared_ids = [_text(row.get("candidate_id")) for row in prepared_rows]
    author_ids = [_text(row.get("candidate_id")) for row in author_rows]
    if (
        len(prepared_ids) != expected
        or len(set(prepared_ids)) != expected
        or len(author_ids) != expected
        or len(set(author_ids)) != expected
        or set(author_ids) != set(prepared_ids)
    ):
        raise ValueError(
            f"分歧裁决必须完整覆盖全部 {expected} 条且 exactly once"
        )
    validated = validate_adjudication_batch(
        author_rows, prepared_rows, config, observed_input_sha256
    )
    decisions = Counter(row["resolution_final_decision"] for row in validated)
    support = Counter(row["resolution_final_kb_support"] for row in validated)
    return {
        "adjudication_rows": validated,
        "summary": {
            "status": "author_adjudication_complete",
            "config_version": config["config_version"],
            "adjudication_version": config["adjudication_version"],
            "dataset_version": config["dataset_version"],
            "kb_version": config["kb_version"],
            "protocol_version": config["protocol_version"],
            "input_sha256": dict(observed_input_sha256),
            "adjudicated_count": len(validated),
            "decision_distribution": dict(sorted(decisions.items())),
            "kb_support_distribution": dict(sorted(support.items())),
            "gold_promotion_performed": False,
            "freeze_performed": False,
            "clinically_validated": False,
            "usage": zero_usage(),
            "workflow_boundary": (
                "同一作者分歧裁决完成；尚未执行 Gold promotion 或冻结，"
                "不构成独立专家验证或临床验证。"
            ),
        },
    }


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("不能写出空裁决 CSV")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return "\ufeff" + buffer.getvalue()


def _atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def write_prepared_batches(
    batches: list[dict[str, Any]], output_dir: str | Path
) -> dict[str, str]:
    if not batches:
        raise ValueError("没有可写出的裁决批次")
    output_path = Path(output_dir)
    batch_dir = output_path / "benchmark120_disagreement_adjudication_batches_v0_1"
    for existing_path in sorted(batch_dir.glob("batch_*.csv")):
        if any(
            _text(row.get(field))
            for row in read_csv(existing_path)
            for field in RESOLUTION_AUTHOR_FIELDS
        ):
            raise FileExistsError(f"拒绝覆盖已有作者裁决结论: {existing_path}")
    batch_assets: list[dict[str, Any]] = []
    for batch in batches:
        path = batch_dir / f"{batch['batch_id']}.csv"
        _atomic_write_text(path, _csv_text(batch["rows"]), encoding="utf-8")
        batch_assets.append(
            {
                "batch_id": batch["batch_id"],
                "row_count": len(batch["rows"]),
                "path": str(path),
                "sha256": compute_sha256(path),
            }
        )
    manifest = {
        "status": "pending_author_confirmation",
        "config_version": batches[0]["config_version"],
        "adjudication_version": batches[0]["adjudication_version"],
        "dataset_version": batches[0]["dataset_version"],
        "kb_version": batches[0]["kb_version"],
        "protocol_version": batches[0]["protocol_version"],
        "input_sha256": dict(batches[0]["input_sha256"]),
        "batch_count": len(batches),
        "candidate_count": sum(len(batch["rows"]) for batch in batches),
        "author_adjudicated_count": 0,
        "gold_promotion_performed": False,
        "freeze_performed": False,
        "clinically_validated": False,
        "batches": batch_assets,
        "usage": zero_usage(),
    }
    manifest_path = output_path / "benchmark120_disagreement_adjudication_manifest_v0_1.json"
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    guide_path = output_path / "benchmark120_disagreement_adjudication_guide_v0_1.md"
    _atomic_write_text(
        guide_path,
        "\n".join(
            [
                "# Benchmark-120 双轮分歧作者裁决指南 v0.1",
                "",
                "- 共 31 条，按 8/8/8/7 分为四批。",
                "- 必须逐条对照问题、来源、页码、证据跨度和两轮字段。",
                "- AI 建议只作为独立草案，不得自动写入作者裁决字段。",
                "- `resolution_reviewer_id=PYH` 只能在作者明确确认后写入。",
                "- `boundary_refusal` 必须绑定 `policy_rule` 与 `safety_policy`。",
                "- 31/31 完整通过校验后才可汇总；本步骤不执行 Gold 晋升或冻结。",
                "- 同一作者裁决不构成独立专家验证或临床验证。",
                "",
            ]
        ),
    )
    return {
        "batch_dir": str(batch_dir),
        "manifest": str(manifest_path),
        "guide": str(guide_path),
    }


def read_batch_dir(path: str | Path) -> list[dict[str, str]]:
    files = sorted(Path(path).glob("batch_*.csv"))
    if not files:
        raise FileNotFoundError(f"未找到作者裁决批次: {path}")
    rows: list[dict[str, str]] = []
    for file_path in files:
        rows.extend(read_csv(file_path))
    return rows


def write_final_adjudication(
    result: dict[str, Any], output_dir: str | Path
) -> dict[str, str]:
    output_path = Path(output_dir)
    final_path = output_path / "benchmark120_disagreement_adjudication_final_v0_1.csv"
    audit_path = output_path / "benchmark120_disagreement_adjudication_audit_v0_1.json"
    summary_path = output_path / "benchmark120_disagreement_adjudication_summary_v0_1.md"
    _atomic_write_text(
        final_path,
        _csv_text(result["adjudication_rows"]),
        encoding="utf-8",
    )
    summary = dict(result["summary"])
    summary["adjudication_csv_sha256"] = compute_sha256(final_path)
    _atomic_write_text(
        audit_path,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(
        summary_path,
        "\n".join(
            [
                "# Benchmark-120 双轮分歧作者裁决结果 v0.1",
                "",
                f"- 状态：`{summary['status']}`",
                f"- 已裁决：{summary['adjudicated_count']}",
                f"- 最终决策分布：`{summary['decision_distribution']}`",
                "- Gold 晋升：否",
                "- Benchmark 冻结：否",
                "- 临床验证：否",
                "- 外部模型/API 调用：0",
                "",
                summary["workflow_boundary"],
                "",
            ]
        ),
    )
    return {
        "adjudication": str(final_path),
        "audit": str(audit_path),
        "summary": str(summary_path),
    }


def _resolve_inputs(
    config: dict[str, Any], repo_root: Path
) -> tuple[list[dict[str, str]], dict[str, str]]:
    paths = {
        label: repo_root / str(value)
        for label, value in dict(config.get("input_paths", {})).items()
    }
    if set(paths) != {"resolution_queue", "resolution_summary"}:
        raise ValueError("分歧裁决配置必须锁定 resolution_queue 与 resolution_summary")
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"锁定输入不存在: {path}")
    observed = {label: compute_sha256(path) for label, path in paths.items()}
    _check_input_hashes(config, observed)
    summary = json.loads(paths["resolution_summary"].read_text(encoding="utf-8"))
    if summary.get("status") != "resolution_queue_ready_author_adjudication_pending":
        raise ValueError("上游 resolution summary 状态不允许进入作者裁决")
    if int(summary.get("resolution_candidate_count", -1)) != int(
        config["expected_candidate_count"]
    ):
        raise ValueError("上游 resolution candidate count 与配置不一致")
    return read_csv(paths["resolution_queue"]), observed


def _prepare_from_config(
    config: dict[str, Any], repo_root: Path
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows, observed = _resolve_inputs(config, repo_root)
    return prepare_adjudication_batches(rows, config, observed), observed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="准备、校验或汇总 Benchmark-120 双轮分歧作者裁决包"
    )
    parser.add_argument(
        "--config",
        default=(
            Path(__file__).resolve().parent
            / "configs"
            / "benchmark120_disagreement_adjudication_v0_1.json"
        ),
    )
    parser.add_argument(
        "--mode", choices=["prepare", "validate-batch", "finalize"], required=True
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--batch-file")
    parser.add_argument("--batch-dir")
    args = parser.parse_args()

    config = load_config(args.config)
    batches, observed = _prepare_from_config(config, ROOT)
    prepared_rows = [row for batch in batches for row in batch["rows"]]
    target_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / str(config["output_dir"])
    )
    if args.mode == "prepare":
        print(
            json.dumps(
                write_prepared_batches(batches, target_dir),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.mode == "validate-batch":
        if not args.batch_file:
            parser.error("validate-batch 需要 --batch-file")
        validated = validate_adjudication_batch(
            read_csv(args.batch_file), prepared_rows, config, observed
        )
        print(json.dumps({"validated_count": len(validated)}, ensure_ascii=False))
        return
    batch_dir = (
        Path(args.batch_dir)
        if args.batch_dir
        else target_dir / "benchmark120_disagreement_adjudication_batches_v0_1"
    )
    result = finalize_adjudications(
        read_batch_dir(batch_dir), prepared_rows, config, observed
    )
    print(
        json.dumps(
            write_final_adjudication(result, target_dir),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
