from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


AUTHOR_REVIEW_FIELDS = [
    "reviewer_id",
    "author_reviewed_at",
    "author_review_outcome",
    "author_review_reason",
    "verified_text_span",
    "supported_claim_types",
    "evidence_scope",
    "age_scope",
    "applicability_conditions",
    "scope_check",
]
ASSISTANT_REVIEW_FIELDS = [
    "assistant_review_outcome",
    "assistant_review_reason",
    "assistant_verified_text_span",
    "assistant_supported_claim_types",
    "assistant_evidence_scope",
    "assistant_age_scope",
    "assistant_applicability_conditions",
    "assistant_scope_check",
]
REQUIRED_CONFIG_FIELDS = {
    "config_version",
    "review_version",
    "dataset_version",
    "kb_version",
    "expected_candidate_count",
    "batch_size",
    "expected_parent_queue_sha256",
    "expected_assistant_draft_sha256",
    "min_verified_text_chars",
    "allowed_author_outcomes",
    "required_scope_check",
}
HTML_BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


def zero_usage() -> dict[str, int | float]:
    return {
        "external_api_calls": 0,
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


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("作者核验配置根节点必须是 JSON 对象")
    missing = sorted(REQUIRED_CONFIG_FIELDS - set(config))
    if missing:
        raise ValueError(f"作者核验配置缺少字段: {', '.join(missing)}")
    if int(config["expected_candidate_count"]) <= 0:
        raise ValueError("expected_candidate_count 必须大于 0")
    if int(config["batch_size"]) <= 0:
        raise ValueError("batch_size 必须大于 0")
    if int(config["min_verified_text_chars"]) <= 0:
        raise ValueError("min_verified_text_chars 必须大于 0")
    return config


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("不能写出空作者核验文件")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _normalized_text(value: Any) -> str:
    text = HTML_BREAK_RE.sub(" ", str(value or ""))
    return WHITESPACE_RE.sub("", text).lower()


def _check_hashes(
    config: dict[str, Any],
    *,
    parent_queue_sha256: str,
    assistant_draft_sha256: str,
) -> None:
    expected_parent = str(config["expected_parent_queue_sha256"])
    expected_draft = str(config["expected_assistant_draft_sha256"])
    if parent_queue_sha256 != expected_parent:
        raise ValueError(
            "parent queue hash mismatch: "
            f"expected {expected_parent}, got {parent_queue_sha256}"
        )
    if assistant_draft_sha256 != expected_draft:
        raise ValueError(
            "assistant draft hash mismatch: "
            f"expected {expected_draft}, got {assistant_draft_sha256}"
        )


def _index_unique_rows(
    rows: list[dict[str, Any]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    candidate_ids = [str(row.get("candidate_id", "")).strip() for row in rows]
    if not all(candidate_ids):
        raise ValueError(f"{label} 含空 candidate_id")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError(f"{label} 含重复 candidate_id")
    return {str(row["candidate_id"]): row for row in rows}


def _validate_source_inputs(
    parent_rows: list[dict[str, Any]],
    assistant_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    expected = int(config["expected_candidate_count"])
    if len(parent_rows) != expected or len(assistant_rows) != expected:
        raise ValueError(
            f"author review requires all {expected} candidates; "
            f"got parent={len(parent_rows)}, assistant={len(assistant_rows)}"
        )
    parents = _index_unique_rows(parent_rows, label="parent queue")
    drafts = _index_unique_rows(assistant_rows, label="assistant draft")
    if set(parents) != set(drafts):
        raise ValueError("parent queue 与 assistant draft 的 candidate_id 不一致")

    for candidate_id, parent in parents.items():
        draft = drafts[candidate_id]
        if str(parent.get("review_status", "")) != "pending_author_review":
            raise ValueError(f"{candidate_id} 不是待作者核验状态")
        if any(str(parent.get(field, "")).strip() for field in AUTHOR_REVIEW_FIELDS):
            raise ValueError(f"{candidate_id} parent queue 作者字段必须为空")
        if any(str(draft.get(field, "")).strip() for field in AUTHOR_REVIEW_FIELDS):
            raise ValueError(f"{candidate_id} AI 草稿不得预填作者字段")
        for field, value in parent.items():
            if field in AUTHOR_REVIEW_FIELDS:
                continue
            if str(draft.get(field, "")) != str(value):
                raise ValueError(f"{candidate_id} assistant draft immutable field drift: {field}")
        missing_assistant = [
            field
            for field in ("assistant_review_outcome", "assistant_review_reason")
            if not str(draft.get(field, "")).strip()
        ]
        if missing_assistant:
            raise ValueError(
                f"{candidate_id} AI 草稿缺少字段: {', '.join(missing_assistant)}"
            )
    return parents, drafts


def prepare_author_batches(
    parent_rows: list[dict[str, Any]],
    assistant_rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    parent_queue_sha256: str,
    assistant_draft_sha256: str,
) -> list[dict[str, Any]]:
    _check_hashes(
        config,
        parent_queue_sha256=parent_queue_sha256,
        assistant_draft_sha256=assistant_draft_sha256,
    )
    _, drafts = _validate_source_inputs(parent_rows, assistant_rows, config)
    ordered = sorted(
        (dict(row) for row in drafts.values()),
        key=lambda row: (
            str(row["source_id"]),
            int(row["page_number"]),
            str(row["candidate_id"]),
        ),
    )
    batch_size = int(config["batch_size"])
    batches: list[dict[str, Any]] = []
    for offset in range(0, len(ordered), batch_size):
        batch_number = len(batches) + 1
        batches.append(
            {
                "batch_id": f"batch_{batch_number:02d}",
                "config_version": config["config_version"],
                "review_version": config["review_version"],
                "dataset_version": config["dataset_version"],
                "kb_version": config["kb_version"],
                "parent_queue_sha256": parent_queue_sha256,
                "assistant_draft_sha256": assistant_draft_sha256,
                "rows": [dict(row) for row in ordered[offset : offset + batch_size]],
            }
        )
    return batches


def _validate_accepted_author_row(
    row: dict[str, Any],
    assistant_row: dict[str, Any],
    config: dict[str, Any],
) -> None:
    candidate_id = str(row["candidate_id"])
    verified_text = str(row.get("verified_text_span", "")).strip()
    if len(verified_text) < int(config["min_verified_text_chars"]):
        raise ValueError(f"{candidate_id} 作者确认的证据跨度过短")
    compact_verified = _normalized_text(verified_text)
    traceable_sources = (
        _normalized_text(assistant_row.get("candidate_text")),
        _normalized_text(assistant_row.get("context_text")),
    )
    if not compact_verified or not any(
        compact_verified in source for source in traceable_sources
    ):
        raise ValueError(f"{candidate_id} 作者确认的证据跨度不可追溯")

    required = [
        "supported_claim_types",
        "evidence_scope",
        "age_scope",
        "applicability_conditions",
        "scope_check",
    ]
    missing = [field for field in required if not str(row.get(field, "")).strip()]
    if missing:
        raise ValueError(
            f"{candidate_id} 接受结论缺少字段: {', '.join(missing)}"
        )
    try:
        claim_types = json.loads(str(row["supported_claim_types"]))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{candidate_id} supported_claim_types 不是合法 JSON") from exc
    if not isinstance(claim_types, list) or not claim_types:
        raise ValueError(f"{candidate_id} supported_claim_types 必须是非空列表")
    if str(row["scope_check"]) != str(config["required_scope_check"]):
        raise ValueError(f"{candidate_id} 作者确认范围超出证据可支持边界")


def validate_author_batch(
    author_rows: list[dict[str, Any]],
    assistant_rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    parent_queue_sha256: str,
    assistant_draft_sha256: str,
) -> list[dict[str, Any]]:
    _check_hashes(
        config,
        parent_queue_sha256=parent_queue_sha256,
        assistant_draft_sha256=assistant_draft_sha256,
    )
    drafts = _index_unique_rows(assistant_rows, label="assistant draft")
    author_index = _index_unique_rows(author_rows, label="author review batch")
    unknown = sorted(set(author_index) - set(drafts))
    if unknown:
        raise ValueError(f"作者核验批次包含未知 candidate_id: {', '.join(unknown)}")

    allowed_outcomes = set(config["allowed_author_outcomes"])
    validated: list[dict[str, Any]] = []
    for candidate_id, row in author_index.items():
        assistant_row = drafts[candidate_id]
        drifted = [
            field
            for field, value in assistant_row.items()
            if field not in AUTHOR_REVIEW_FIELDS
            and str(row.get(field, "")) != str(value)
        ]
        if drifted:
            raise ValueError(
                f"{candidate_id} author review immutable field drift: "
                f"{', '.join(drifted)}"
            )
        reviewer_id = str(row.get("reviewer_id", "")).strip()
        reviewed_at = str(row.get("author_reviewed_at", "")).strip()
        outcome = str(row.get("author_review_outcome", "")).strip()
        reason = str(row.get("author_review_reason", "")).strip()
        if not reviewer_id or not reviewed_at:
            raise ValueError(f"{candidate_id} 缺少作者身份或核验时间")
        if outcome not in allowed_outcomes:
            raise ValueError(f"{candidate_id} author_review_outcome 非法: {outcome}")
        if outcome == "rejected" and not reason:
            raise ValueError(f"{candidate_id} 缺少明确拒绝理由")
        if outcome == "accepted":
            if not reason:
                raise ValueError(f"{candidate_id} 缺少作者接受理由")
            _validate_accepted_author_row(row, assistant_row, config)
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
    parent_rows: list[dict[str, Any]],
    assistant_rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    parent_queue_sha256: str,
    assistant_draft_sha256: str,
) -> dict[str, Any]:
    _validate_source_inputs(parent_rows, assistant_rows, config)
    expected = int(config["expected_candidate_count"])
    author_ids = [str(row.get("candidate_id", "")) for row in author_rows]
    assistant_ids = {str(row["candidate_id"]) for row in assistant_rows}
    if (
        len(author_rows) != expected
        or len(set(author_ids)) != expected
        or set(author_ids) != assistant_ids
    ):
        raise ValueError(
            f"author review must cover all {expected} candidates exactly once"
        )
    validated = validate_author_batch(
        author_rows,
        assistant_rows,
        config,
        parent_queue_sha256=parent_queue_sha256,
        assistant_draft_sha256=assistant_draft_sha256,
    )
    outcomes = Counter(row["author_review_outcome"] for row in validated)
    return {
        "review_rows": validated,
        "summary": {
            "status": "author_review_complete",
            "config_version": config["config_version"],
            "review_version": config["review_version"],
            "dataset_version": config["dataset_version"],
            "kb_version": config["kb_version"],
            "parent_queue_sha256": parent_queue_sha256,
            "assistant_draft_sha256": assistant_draft_sha256,
            "author_reviewed_count": len(validated),
            "outcome_counts": dict(sorted(outcomes.items())),
            "anchor_promotion_performed": False,
            "usage": zero_usage(),
            "medical_boundary": (
                "作者核验完成不等于 gold evidence 晋升、独立专家验证或临床验证；"
                "证据晋升必须在后续独立步骤中再次执行去重和范围门禁。"
            ),
        },
    }


def write_prepared_author_batches(
    batches: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, str]:
    if not batches:
        raise ValueError("没有可写出的作者核验批次")
    output_path = Path(output_dir)
    batch_dir = output_path / "anchor_expansion_author_review_batches_v0_2"
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_files: list[dict[str, Any]] = []
    for batch in batches:
        path = batch_dir / f"{batch['batch_id']}.csv"
        if path.exists():
            existing = read_csv(path)
            if any(
                str(row.get(field, "")).strip()
                for row in existing
                for field in AUTHOR_REVIEW_FIELDS
            ):
                raise FileExistsError(f"拒绝覆盖已有作者核验结论: {path}")
        _write_csv(batch["rows"], path)
        batch_files.append(
            {
                "batch_id": batch["batch_id"],
                "row_count": len(batch["rows"]),
                "path": str(path),
                "sha256": compute_sha256(path),
            }
        )

    manifest = {
        "status": "pending_author_review",
        "config_version": batches[0]["config_version"],
        "review_version": batches[0]["review_version"],
        "dataset_version": batches[0]["dataset_version"],
        "kb_version": batches[0]["kb_version"],
        "parent_queue_sha256": batches[0]["parent_queue_sha256"],
        "assistant_draft_sha256": batches[0]["assistant_draft_sha256"],
        "batch_count": len(batches),
        "candidate_count": sum(len(batch["rows"]) for batch in batches),
        "author_confirmed_count": 0,
        "batches": batch_files,
        "usage": zero_usage(),
        "medical_boundary": (
            "工作包仅供作者逐条核验；空白作者字段不得解释为确认，"
            "也不得直接用于 gold evidence 晋升。"
        ),
    }
    manifest_path = output_path / "anchor_expansion_author_review_manifest_v0_2.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    guide_path = output_path / "anchor_expansion_author_review_guide_v0_2.md"
    guide_path.write_text(
        "\n".join(
            [
                "# 第二批证据锚点作者核验指南 v0.2",
                "",
                "- 共 58 条，按 10/10/10/10/10/8 分为 6 批。",
                "- 必须逐条核对来源、页码、候选原文和 AI 辅助草稿。",
                "- `accepted` 仅表示作者确认窄范围证据可追溯；必须填写全部范围字段。",
                "- `rejected` 必须填写明确理由，不得为满足配额而接受。",
                "- AI 草稿不是作者意见，不得预填或批量代签。",
                "- 58 条未全部完成前禁止汇总；本步骤不执行证据锚点晋升。",
                "- 该流程不构成独立专家验证或临床验证。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "batch_dir": str(batch_dir),
        "manifest": str(manifest_path),
        "guide": str(guide_path),
    }


def read_author_batch_dir(path: str | Path) -> list[dict[str, str]]:
    batch_dir = Path(path)
    files = sorted(batch_dir.glob("batch_*.csv"))
    if not files:
        raise FileNotFoundError(f"未找到作者核验批次: {batch_dir}")
    rows: list[dict[str, str]] = []
    for file_path in files:
        rows.extend(read_csv(file_path))
    return rows


def write_final_author_review(
    result: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    review_path = output_path / "anchor_expansion_author_review_v0_2.csv"
    summary_path = output_path / "anchor_expansion_author_review_audit_v0_2.json"
    report_path = output_path / "anchor_expansion_author_review_summary_v0_2.md"
    _write_csv(result["review_rows"], review_path)
    summary = dict(result["summary"])
    summary["author_review_csv_sha256"] = compute_sha256(review_path)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        "\n".join(
            [
                "# 第二批证据锚点作者核验摘要 v0.2",
                "",
                f"- 状态：`{summary['status']}`",
                f"- 已核验：{summary['author_reviewed_count']} 条",
                f"- 结论分布：`{json.dumps(summary['outcome_counts'], ensure_ascii=False)}`",
                f"- 作者核验 CSV SHA-256：`{summary['author_review_csv_sha256']}`",
                "- 外部 API / input tokens / output tokens / 估算费用：0 / 0 / 0 / 0",
                "- 本步骤未执行证据锚点晋升，也不构成独立专家或临床验证。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "review_csv": str(review_path),
        "summary_json": str(summary_path),
        "summary_md": str(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="准备、校验或汇总第二批证据锚点作者核验工作包"
    )
    parser.add_argument(
        "--mode",
        choices=("prepare", "validate-batch", "finalize"),
        default="prepare",
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path(
            "revision/benchmark/benchmark_v1/anchor_expansion_review_queue_v0_2.csv"
        ),
    )
    parser.add_argument(
        "--drafts",
        type=Path,
        default=Path(
            "revision/benchmark/benchmark_v1/anchor_expansion_assistant_draft_v0_2.csv"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "experiments/phase7_formal_experiments/configs/"
            "benchmark_anchor_expansion_author_review_v0_2.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("revision/benchmark/benchmark_v1"),
    )
    parser.add_argument("--review-batch", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    parent_rows = read_csv(args.queue)
    assistant_rows = read_csv(args.drafts)
    parent_hash = compute_sha256(args.queue)
    draft_hash = compute_sha256(args.drafts)

    if args.mode == "prepare":
        batches = prepare_author_batches(
            parent_rows,
            assistant_rows,
            config,
            parent_queue_sha256=parent_hash,
            assistant_draft_sha256=draft_hash,
        )
        result = write_prepared_author_batches(batches, args.output_dir)
    elif args.mode == "validate-batch":
        if args.review_batch is None:
            raise ValueError("validate-batch 模式必须提供 --review-batch")
        validated = validate_author_batch(
            read_csv(args.review_batch),
            assistant_rows,
            config,
            parent_queue_sha256=parent_hash,
            assistant_draft_sha256=draft_hash,
        )
        result = {
            "status": "author_batch_valid",
            "validated_count": len(validated),
            "usage": zero_usage(),
        }
    else:
        batch_dir = (
            args.output_dir / "anchor_expansion_author_review_batches_v0_2"
        )
        author_rows = read_author_batch_dir(batch_dir)
        finalized = finalize_author_reviews(
            author_rows,
            parent_rows,
            assistant_rows,
            config,
            parent_queue_sha256=parent_hash,
            assistant_draft_sha256=draft_hash,
        )
        result = write_final_author_review(finalized, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
