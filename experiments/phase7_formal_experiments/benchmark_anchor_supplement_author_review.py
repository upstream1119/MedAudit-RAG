from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


AUTHOR_REVIEW_FIELDS = [
    "author_outcome",
    "author_final_question",
    "author_final_decision",
    "author_final_risk_labels",
    "author_allowed_answer_scope",
    "author_forbidden_claims",
    "author_reason",
    "reviewer_id",
    "reviewed_at",
]
PROMOTABLE_FIELDS = [
    "author_final_question",
    "author_final_decision",
    "author_final_risk_labels",
    "author_allowed_answer_scope",
    "author_forbidden_claims",
]
REQUIRED_CONFIG_FIELDS = {
    "config_version",
    "review_version",
    "dataset_version",
    "kb_version",
    "expected_candidate_count",
    "batch_sizes",
    "expected_input_sha256",
    "allowed_author_outcomes",
    "allowed_final_decisions",
}
QUEUE_IMMUTABLE_FIELDS = [
    "candidate_id",
    "question",
    "provisional_expected_decision",
    "challenge_type",
    "source_id",
    "source_title",
    "page_number",
    "anchor_text_span",
    "evidence_scope",
    "age_scope",
    "applicability_conditions",
    "evidence_anchor_ids",
    "independence_unit_id",
]
CANDIDATE_CONTEXT_FIELDS = [
    "provisional_risk_labels",
    "current_kb_support",
    "policy_rule_ids",
    "supported_claim_types",
    "scope_check",
    "source_filename",
    "source_sha256",
]
INPUT_PATH_FIELDS = {
    "candidates": "candidate_path",
    "review_queue": "review_queue_path",
    "audit": "audit_path",
    "summary": "summary_path",
}


def zero_usage() -> dict[str, int | float]:
    return {
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
    }


def compute_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("不能写出空作者审核文件")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("作者审核配置根节点必须是 JSON 对象")
    missing = sorted(REQUIRED_CONFIG_FIELDS - set(config))
    if missing:
        raise ValueError(f"作者审核配置缺少字段: {', '.join(missing)}")
    return config


def _check_input_hashes(
    config: dict[str, Any], observed_input_sha256: dict[str, str]
) -> None:
    expected = dict(config["expected_input_sha256"])
    if observed_input_sha256 != expected:
        raise ValueError(
            "input hash mismatch: "
            f"expected={expected}, observed={observed_input_sha256}"
        )


def _index_unique(
    rows: list[dict[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    candidate_ids = [str(row.get("candidate_id", "")).strip() for row in rows]
    if not all(candidate_ids):
        raise ValueError(f"{label} 含空 candidate_id")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"{label} 含重复 candidate_id")
    return {candidate_id: row for candidate_id, row in zip(candidate_ids, rows)}


def _serialized(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _prepare_rows(
    candidate_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    expected = int(config["expected_candidate_count"])
    if len(candidate_rows) != expected or len(review_rows) != expected:
        raise ValueError(
            f"author review requires all {expected} candidates; "
            f"got candidates={len(candidate_rows)}, review_queue={len(review_rows)}"
        )
    candidates = _index_unique(candidate_rows, label="candidate rows")
    reviews = _index_unique(review_rows, label="review queue")
    if set(candidates) != set(reviews):
        raise ValueError("candidate rows 与 review queue 的 candidate_id 不一致")

    prepared: list[dict[str, Any]] = []
    for candidate_id, candidate in candidates.items():
        review = dict(reviews[candidate_id])
        if candidate.get("annotation_status") != "pending_author_review":
            raise ValueError(f"{candidate_id} 不是待作者审核状态")
        if candidate.get("freeze_status") != "draft":
            raise ValueError(f"{candidate_id} 不是 draft 状态")
        for field in QUEUE_IMMUTABLE_FIELDS:
            if _serialized(review.get(field, "")) != _serialized(
                candidate.get(field, "")
            ):
                raise ValueError(
                    f"{candidate_id} review queue immutable field drift: {field}"
                )
        missing_context = [
            field for field in CANDIDATE_CONTEXT_FIELDS if field not in candidate
        ]
        if missing_context:
            raise ValueError(
                f"{candidate_id} 候选缺少审核上下文字段: "
                f"{', '.join(missing_context)}"
            )
        for field in CANDIDATE_CONTEXT_FIELDS:
            review[field] = _serialized(candidate[field])
        if any(str(review.get(field, "")).strip() for field in AUTHOR_REVIEW_FIELDS):
            raise ValueError(f"{candidate_id} review queue 作者字段必须为空")
        for field in AUTHOR_REVIEW_FIELDS:
            review[field] = ""
        prepared.append(review)
    return sorted(
        prepared,
        key=lambda row: (
            str(row["source_id"]),
            int(row["page_number"]),
            str(row["candidate_id"]),
        ),
    )


def prepare_author_batches(
    candidate_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    observed_input_sha256: dict[str, str],
) -> list[dict[str, Any]]:
    _check_input_hashes(config, observed_input_sha256)
    rows = _prepare_rows(candidate_rows, review_rows, config)
    batch_sizes = [int(size) for size in config["batch_sizes"]]
    if any(size <= 0 for size in batch_sizes):
        raise ValueError("batch_sizes 必须全部大于 0")
    if sum(batch_sizes) != len(rows):
        raise ValueError("batch_sizes 总数必须等于候选题数量")

    batches: list[dict[str, Any]] = []
    offset = 0
    for batch_number, batch_size in enumerate(batch_sizes, start=1):
        batches.append(
            {
                "batch_id": f"batch_{batch_number:02d}",
                "config_version": config["config_version"],
                "review_version": config["review_version"],
                "dataset_version": config["dataset_version"],
                "kb_version": config["kb_version"],
                "input_sha256": dict(observed_input_sha256),
                "rows": [dict(row) for row in rows[offset : offset + batch_size]],
            }
        )
        offset += batch_size
    return batches


def _parse_nonempty_json_list(value: Any, *, field: str, candidate_id: str) -> list:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{candidate_id} {field} 不是合法 JSON") from exc
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"{candidate_id} {field} 必须是非空列表")
    if not all(str(item).strip() for item in parsed):
        raise ValueError(f"{candidate_id} {field} 不能包含空值")
    return parsed


def _validate_accepted_row(
    row: dict[str, Any], config: dict[str, Any]
) -> None:
    candidate_id = str(row["candidate_id"])
    required = [
        "author_final_question",
        "author_final_decision",
        "author_final_risk_labels",
        "author_allowed_answer_scope",
        "author_forbidden_claims",
    ]
    missing = [field for field in required if not str(row.get(field, "")).strip()]
    if missing:
        raise ValueError(
            f"{candidate_id} accepted 结论缺少字段: {', '.join(missing)}"
        )
    if str(row["author_final_decision"]) not in set(
        config["allowed_final_decisions"]
    ):
        raise ValueError(
            f"{candidate_id} author_final_decision 非法: "
            f"{row['author_final_decision']}"
        )
    _parse_nonempty_json_list(
        row["author_final_risk_labels"],
        field="author_final_risk_labels",
        candidate_id=candidate_id,
    )
    _parse_nonempty_json_list(
        row["author_forbidden_claims"],
        field="author_forbidden_claims",
        candidate_id=candidate_id,
    )


def _validate_nonaccepted_row(row: dict[str, Any], *, outcome: str) -> None:
    candidate_id = str(row["candidate_id"])
    if outcome == "revision_required":
        if not str(row.get("author_final_question", "")).strip():
            raise ValueError(
                f"{candidate_id} revision_required 缺少修改后问题草案"
            )
        forbidden = PROMOTABLE_FIELDS[1:]
    else:
        forbidden = PROMOTABLE_FIELDS
    populated = [field for field in forbidden if str(row.get(field, "")).strip()]
    if populated:
        raise ValueError(
            f"{candidate_id} {outcome} 不得携带可晋升字段: "
            f"{', '.join(populated)}"
        )


def validate_author_batch(
    author_rows: list[dict[str, Any]],
    prepared_rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    observed_input_sha256: dict[str, str],
) -> list[dict[str, Any]]:
    _check_input_hashes(config, observed_input_sha256)
    prepared = _index_unique(prepared_rows, label="prepared review rows")
    authors = _index_unique(author_rows, label="author review rows")
    unknown = sorted(set(authors) - set(prepared))
    if unknown:
        raise ValueError(f"作者审核包含未知 candidate_id: {', '.join(unknown)}")

    allowed_outcomes = set(config["allowed_author_outcomes"])
    validated: list[dict[str, Any]] = []
    for candidate_id, row in authors.items():
        source = prepared[candidate_id]
        drifted = [
            field
            for field, value in source.items()
            if field not in AUTHOR_REVIEW_FIELDS
            and _serialized(row.get(field, "")) != _serialized(value)
        ]
        if drifted:
            raise ValueError(
                f"{candidate_id} author review immutable field drift: "
                f"{', '.join(drifted)}"
            )
        reviewer_id = str(row.get("reviewer_id", "")).strip()
        reviewed_at = str(row.get("reviewed_at", "")).strip()
        outcome = str(row.get("author_outcome", "")).strip()
        reason = str(row.get("author_reason", "")).strip()
        if not reviewer_id or not reviewed_at:
            raise ValueError(f"{candidate_id} 缺少审核人或审核时间")
        if outcome not in allowed_outcomes:
            raise ValueError(f"{candidate_id} author_outcome 非法: {outcome}")
        if not reason:
            raise ValueError(f"{candidate_id} 缺少作者审核理由")
        if outcome == "accepted":
            _validate_accepted_row(row, config)
        else:
            _validate_nonaccepted_row(row, outcome=outcome)
        validated.append(dict(row))
    return sorted(
        validated,
        key=lambda row: (
            str(row["source_id"]),
            int(row["page_number"]),
            str(row["candidate_id"]),
        ),
    )


def finalize_author_reviews(
    author_rows: list[dict[str, Any]],
    prepared_rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    observed_input_sha256: dict[str, str],
) -> dict[str, Any]:
    expected = int(config["expected_candidate_count"])
    prepared_ids = [str(row.get("candidate_id", "")) for row in prepared_rows]
    author_ids = [str(row.get("candidate_id", "")) for row in author_rows]
    if (
        len(prepared_rows) != expected
        or len(set(prepared_ids)) != expected
        or len(author_rows) != expected
        or len(set(author_ids)) != expected
        or set(author_ids) != set(prepared_ids)
    ):
        raise ValueError(
            f"author review must cover all {expected} candidates exactly once"
        )
    validated = validate_author_batch(
        author_rows,
        prepared_rows,
        config,
        observed_input_sha256=observed_input_sha256,
    )
    outcomes = Counter(row["author_outcome"] for row in validated)
    final_decisions = Counter(
        row["author_final_decision"]
        for row in validated
        if row["author_outcome"] == "accepted"
    )
    return {
        "review_rows": validated,
        "summary": {
            "status": "author_review_complete",
            "config_version": config["config_version"],
            "review_version": config["review_version"],
            "dataset_version": config["dataset_version"],
            "kb_version": config["kb_version"],
            "input_sha256": dict(observed_input_sha256),
            "author_reviewed_count": len(validated),
            "outcome_counts": dict(sorted(outcomes.items())),
            "accepted_decision_counts": dict(sorted(final_decisions.items())),
            "candidate_merge_performed": False,
            "gold_promotion_performed": False,
            "freeze_performed": False,
            "usage": zero_usage(),
            "medical_boundary": (
                "作者审核完成不等于 benchmark 合并、gold label 晋升、独立专家验证"
                "或临床验证；通过项必须在后续独立步骤中重新执行范围、独立性和"
                "分布审计。"
            ),
        },
    }


def write_prepared_author_batches(
    batches: list[dict[str, Any]], output_dir: str | Path
) -> dict[str, str]:
    if not batches:
        raise ValueError("没有可写出的作者审核批次")
    output_path = Path(output_dir)
    batch_dir = output_path / "supplement_author_review_batches_v0_2"
    existing_batch_paths = sorted(batch_dir.glob("batch_*.csv"))
    for path in existing_batch_paths:
        existing = read_csv(path)
        if any(
            str(row.get(field, "")).strip()
            for row in existing
            for field in AUTHOR_REVIEW_FIELDS
        ):
            raise FileExistsError(f"拒绝覆盖已有作者审核结论: {path}")

    batch_files: list[dict[str, Any]] = []
    for batch in batches:
        path = batch_dir / f"{batch['batch_id']}.csv"
        _write_csv(batch["rows"], path)
        batch_files.append(
            {
                "batch_id": batch["batch_id"],
                "row_count": len(batch["rows"]),
                "path": str(path),
                "sha256": compute_sha256(path),
            }
        )

    candidate_count = sum(len(batch["rows"]) for batch in batches)
    manifest = {
        "status": "pending_author_review",
        "config_version": batches[0]["config_version"],
        "review_version": batches[0]["review_version"],
        "dataset_version": batches[0]["dataset_version"],
        "kb_version": batches[0]["kb_version"],
        "input_sha256": dict(batches[0]["input_sha256"]),
        "batch_count": len(batches),
        "candidate_count": candidate_count,
        "author_reviewed_count": 0,
        "candidate_merge_performed": False,
        "gold_promotion_performed": False,
        "freeze_performed": False,
        "batches": batch_files,
        "usage": zero_usage(),
        "medical_boundary": (
            "审核包只用于作者逐条确认候选题；空白作者字段不代表确认，"
            "也不得直接用于 benchmark 合并、gold 晋升或冻结。"
        ),
    }
    manifest_path = output_path / "supplement_author_review_manifest_v0_2.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    batch_shape = "/".join(str(len(batch["rows"])) for batch in batches)
    guide_path = output_path / "supplement_author_review_guide_v0_2.md"
    guide_path.write_text(
        "\n".join(
            [
                "# Benchmark-v1 补充候选作者审核指南 v0.2",
                "",
                f"- 共 {candidate_count} 条，按 {batch_shape} 分为 {len(batches)} 批。",
                "- 必须逐条核对问题、来源、页码、证据片段和适用范围。",
                "- `accepted` 必须填写最终问题、决策、风险标签、允许回答范围和禁止主张。",
                "- `revision_required` 只记录修改后问题草案，完成再次审核前不可晋升。",
                "- `rejected` 只保留拒绝理由，不得携带可晋升字段。",
                "- 未全部完成前禁止汇总；本步骤不执行候选合并、Gold 晋升或冻结。",
                "- 同一作者审核不构成独立专家验证或临床验证。",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    return {
        "batch_dir": str(batch_dir),
        "manifest": str(manifest_path),
        "guide": str(guide_path),
    }


def _resolve_inputs(
    config: dict[str, Any], repo_root: Path
) -> tuple[dict[str, Path], dict[str, str]]:
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for label, config_field in INPUT_PATH_FIELDS.items():
        if config_field not in config:
            raise ValueError(f"作者审核配置缺少字段: {config_field}")
        path = repo_root / str(config[config_field])
        if not path.exists():
            raise FileNotFoundError(f"锁定输入不存在: {path}")
        paths[label] = path
        hashes[label] = compute_sha256(path)
    _check_input_hashes(config, hashes)
    return paths, hashes


def _prepare_from_sources(
    config: dict[str, Any], repo_root: Path
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    paths, hashes = _resolve_inputs(config, repo_root)
    candidates = _read_jsonl(paths["candidates"])
    review_rows = read_csv(paths["review_queue"])
    batches = prepare_author_batches(
        candidates,
        review_rows,
        config,
        observed_input_sha256=hashes,
    )
    return batches, hashes


def prepare_from_config(
    config_path: str | Path, *, output_dir: str | Path | None = None
) -> dict[str, str]:
    config = load_config(config_path)
    repo_root = Path(__file__).resolve().parents[2]
    batches, _ = _prepare_from_sources(config, repo_root)
    target_dir = (
        Path(output_dir)
        if output_dir is not None
        else repo_root / str(config["output_dir"])
    )
    return write_prepared_author_batches(batches, target_dir)


def read_author_batch_dir(path: str | Path) -> list[dict[str, str]]:
    batch_dir = Path(path)
    files = sorted(batch_dir.glob("batch_*.csv"))
    if not files:
        raise FileNotFoundError(f"未找到作者审核批次: {batch_dir}")
    rows: list[dict[str, str]] = []
    for path in files:
        rows.extend(read_csv(path))
    return rows


def write_final_author_review(
    result: dict[str, Any], output_dir: str | Path
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    review_path = (
        output_path / "benchmark_anchor_supplement_author_review_final_v0_2.csv"
    )
    audit_path = (
        output_path / "benchmark_anchor_supplement_author_review_audit_v0_2.json"
    )
    summary_path = (
        output_path / "benchmark_anchor_supplement_author_review_summary_v0_2.md"
    )
    _write_csv(result["review_rows"], review_path)
    summary = dict(result["summary"])
    summary["author_review_csv_sha256"] = compute_sha256(review_path)
    audit_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    summary_path.write_text(
        "\n".join(
            [
                "# Benchmark-v1 补充候选作者审核结果 v0.2",
                "",
                f"- 审核状态：`{summary['status']}`",
                f"- 已审核：{summary['author_reviewed_count']}",
                f"- 审核结果分布：`{summary['outcome_counts']}`",
                f"- 接受项决策分布：`{summary['accepted_decision_counts']}`",
                "- 候选合并：否",
                "- Gold 晋升：否",
                "- Benchmark 冻结：否",
                "- 外部模型/API 调用：0",
                "",
                summary["medical_boundary"],
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    return {
        "review": str(review_path),
        "audit": str(audit_path),
        "summary": str(summary_path),
    }


def _prepared_rows_from_batches(
    batches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [row for batch in batches for row in batch["rows"]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="准备、校验或汇总 Benchmark-v1 补充候选作者审核包"
    )
    parser.add_argument(
        "--config",
        default=(
            Path(__file__).resolve().parent
            / "configs"
            / "benchmark_anchor_supplement_author_review_v0_2.json"
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
    repo_root = Path(__file__).resolve().parents[2]
    target_dir = (
        Path(args.output_dir)
        if args.output_dir
        else repo_root / str(config["output_dir"])
    )
    batches, hashes = _prepare_from_sources(config, repo_root)
    prepared_rows = _prepared_rows_from_batches(batches)

    if args.mode == "prepare":
        outputs = write_prepared_author_batches(batches, target_dir)
        print(json.dumps(outputs, ensure_ascii=False, indent=2))
        return
    if args.mode == "validate-batch":
        if not args.batch_file:
            parser.error("validate-batch 需要 --batch-file")
        validated = validate_author_batch(
            read_csv(args.batch_file),
            prepared_rows,
            config,
            observed_input_sha256=hashes,
        )
        print(json.dumps({"validated_count": len(validated)}, ensure_ascii=False))
        return

    batch_dir = (
        Path(args.batch_dir)
        if args.batch_dir
        else target_dir / "supplement_author_review_batches_v0_2"
    )
    result = finalize_author_reviews(
        read_author_batch_dir(batch_dir),
        prepared_rows,
        config,
        observed_input_sha256=hashes,
    )
    outputs = write_final_author_review(result, target_dir)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
