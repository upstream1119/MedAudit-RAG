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
PROVENANCE_FIELDS = [
    "candidate_id",
    "source_id",
    "source_title",
    "source_filename",
    "source_sha256",
    "page_number",
    "block_type",
    "granularity",
    "candidate_text",
    "context_candidate_id",
    "context_text",
    "matched_topics",
    "matched_terms",
    "review_status",
    "dev50_overlap_anchor_ids",
    "selection_rank_within_source",
    "parser_version",
    "chunker_version",
    "candidate_config_version",
    "review_config_version",
]
REQUIRED_CONFIG_FIELDS = {
    "config_version",
    "dataset_version",
    "kb_version",
    "expected_queue_size",
    "batch_size",
    "min_verified_text_chars",
    "min_alnum_ratio",
    "allowed_assistant_outcomes",
}
WHITESPACE_RE = re.compile(r"\s+")
HTML_BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)


def compute_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("核验配置根节点必须是 JSON 对象")
    missing = sorted(REQUIRED_CONFIG_FIELDS - set(config))
    if missing:
        raise ValueError(f"核验配置缺少字段: {', '.join(missing)}")
    if int(config["batch_size"]) <= 0:
        raise ValueError("batch_size 必须大于 0")
    return config


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalized_text(value: Any) -> str:
    text = HTML_BREAK_RE.sub(" ", str(value or ""))
    return WHITESPACE_RE.sub("", text).lower()


def _alnum_ratio(value: str) -> float:
    if not value:
        return 0.0
    return sum(character.isalnum() for character in value) / len(value)


def _ensure_pristine_parent_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    expected = int(config["expected_queue_size"])
    if len(rows) != expected:
        raise ValueError(f"候选队列数量应为 {expected}，实际为 {len(rows)}")
    candidate_ids = [str(row.get("candidate_id", "")) for row in rows]
    if not all(candidate_ids) or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("候选队列含空或重复 candidate_id")
    for row in rows:
        if row.get("review_status") != "pending_author_review":
            raise ValueError(f"{row.get('candidate_id')} 不是待作者核验状态")
        if any(str(row.get(field, "")).strip() for field in AUTHOR_REVIEW_FIELDS):
            raise ValueError(f"{row.get('candidate_id')} 作者字段必须保持为空")


def _blank_assistant_fields(row: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(row)
    prepared.update({field: "" for field in ASSISTANT_REVIEW_FIELDS})
    return prepared


def prepare_review_batches(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    parent_queue_sha256: str,
) -> list[dict[str, Any]]:
    _ensure_pristine_parent_rows(rows, config)
    ordered = sorted(
        rows,
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
                "parent_queue_sha256": parent_queue_sha256,
                "config_version": config["config_version"],
                "dataset_version": config["dataset_version"],
                "kb_version": config["kb_version"],
                "rows": [
                    _blank_assistant_fields(row)
                    for row in ordered[offset : offset + batch_size]
                ],
            }
        )
    return batches


def _validate_provenance(
    draft: dict[str, Any],
    parent: dict[str, Any],
) -> None:
    drifted = [
        field
        for field in [*PROVENANCE_FIELDS, *AUTHOR_REVIEW_FIELDS]
        if str(draft.get(field, "")) != str(parent.get(field, ""))
    ]
    if drifted:
        raise ValueError(
            f"{parent['candidate_id']} 来源字段漂移: {', '.join(drifted)}"
        )


def _validate_accepted_draft(
    draft: dict[str, Any],
    parent: dict[str, Any],
    config: dict[str, Any],
) -> None:
    verified_text = str(draft.get("assistant_verified_text_span", "")).strip()
    compact_verified = _normalized_text(verified_text)
    if (
        len(verified_text) < int(config["min_verified_text_chars"])
        or _alnum_ratio(verified_text) < float(config["min_alnum_ratio"])
    ):
        raise ValueError(f"{parent['candidate_id']} 核验原文过短或不可读")
    traceable_sources = (
        _normalized_text(parent.get("candidate_text")),
        _normalized_text(parent.get("context_text")),
    )
    if not any(compact_verified in source for source in traceable_sources):
        raise ValueError(f"{parent['candidate_id']} 核验原文不可追溯")
    required = [
        "assistant_supported_claim_types",
        "assistant_evidence_scope",
        "assistant_age_scope",
        "assistant_applicability_conditions",
        "assistant_scope_check",
    ]
    missing = [field for field in required if not str(draft.get(field, "")).strip()]
    if missing:
        raise ValueError(
            f"{parent['candidate_id']} 接受草稿缺少字段: {', '.join(missing)}"
        )
    try:
        claim_types = json.loads(str(draft["assistant_supported_claim_types"]))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{parent['candidate_id']} claim types 不是合法 JSON") from exc
    if not isinstance(claim_types, list) or not claim_types:
        raise ValueError(f"{parent['candidate_id']} claim types 必须是非空列表")
    if draft["assistant_scope_check"] != "within_can_support":
        raise ValueError(f"{parent['candidate_id']} 接受草稿超出证据支持范围")


def validate_assistant_drafts(
    drafts: list[dict[str, Any]],
    parent_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    _ensure_pristine_parent_rows(parent_rows, config)
    parents = {row["candidate_id"]: row for row in parent_rows}
    draft_ids = [str(row.get("candidate_id", "")) for row in drafts]
    if len(drafts) != len(parent_rows) or set(draft_ids) != set(parents):
        raise ValueError("审核草稿必须完整覆盖父队列且不得增删候选")
    if len(set(draft_ids)) != len(draft_ids):
        raise ValueError("审核草稿含重复 candidate_id")

    allowed = set(config["allowed_assistant_outcomes"])
    validated: list[dict[str, Any]] = []
    for draft in drafts:
        parent = parents[draft["candidate_id"]]
        _validate_provenance(draft, parent)
        outcome = str(draft.get("assistant_review_outcome", "")).strip()
        reason = str(draft.get("assistant_review_reason", "")).strip()
        if outcome not in allowed:
            raise ValueError(f"{parent['candidate_id']} 审核结论无效: {outcome}")
        if not reason:
            raise ValueError(f"{parent['candidate_id']} 缺少审核理由")
        if outcome == "accepted_draft":
            _validate_accepted_draft(draft, parent, config)
        validated.append(dict(draft))
    return sorted(
        validated,
        key=lambda row: (
            str(row["source_id"]),
            int(row["page_number"]),
            str(row["candidate_id"]),
        ),
    )


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = list(rows[0]) if rows else [*PROVENANCE_FIELDS, *AUTHOR_REVIEW_FIELDS]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_prepared_batches(
    batches: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    batch_dir = output_path / "anchor_expansion_review_batches_v0_2"
    batch_dir.mkdir(parents=True, exist_ok=True)
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
    manifest = {
        "config_version": batches[0]["config_version"] if batches else "",
        "dataset_version": batches[0]["dataset_version"] if batches else "",
        "kb_version": batches[0]["kb_version"] if batches else "",
        "parent_queue_sha256": batches[0]["parent_queue_sha256"] if batches else "",
        "batch_count": len(batches),
        "candidate_count": sum(len(batch["rows"]) for batch in batches),
        "batches": batch_files,
        "external_api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
    }
    manifest_path = output_path / "anchor_expansion_review_batch_manifest_v0_2.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"batch_dir": str(batch_dir), "manifest": str(manifest_path)}


def write_review_outputs(
    drafts: list[dict[str, Any]],
    *,
    parent_queue_sha256: str,
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    draft_path = output_path / "anchor_expansion_assistant_draft_v0_2.csv"
    summary_path = output_path / "anchor_expansion_review_audit_v0_2.json"
    report_path = output_path / "anchor_expansion_review_summary_v0_2.md"
    _write_csv(drafts, draft_path)
    outcomes = Counter(row["assistant_review_outcome"] for row in drafts)
    summary = {
        "config_version": config["config_version"],
        "dataset_version": config["dataset_version"],
        "kb_version": config["kb_version"],
        "parent_queue_sha256": parent_queue_sha256,
        "assistant_reviewed_count": len(drafts),
        "author_confirmed_count": 0,
        "outcome_counts": dict(sorted(outcomes.items())),
        "draft_csv_sha256": compute_sha256(draft_path),
        "external_api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
        "medical_boundary": (
            "AI 辅助核验草稿不是作者确认、gold evidence、专家验证或临床验证。"
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_lines = [
        "# Benchmark-v1 第二批证据锚点 AI 辅助核验摘要 v0.2",
        "",
        f"- AI 辅助核验：{len(drafts)} 条",
        "- 作者确认：0 条",
        f"- 父队列 SHA-256：`{parent_queue_sha256}`",
        f"- 草稿 SHA-256：`{summary['draft_csv_sha256']}`",
        "- 外部 API / Token / 成本：0 / 0 / 0",
        "- 边界：该文件只是核验草稿，不得直接作为 gold evidence。",
        "",
        "## 草稿结论分布",
        "",
    ]
    report_lines.extend(
        f"- `{outcome}`：{count}"
        for outcome, count in sorted(outcomes.items())
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {
        "draft_csv": str(draft_path),
        "summary_json": str(summary_path),
        "summary_md": str(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="准备或校验 Benchmark-v1 第二批证据锚点 AI 辅助核验草稿"
    )
    parser.add_argument("--mode", choices=("prepare", "validate"), default="prepare")
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path(
            "revision/benchmark/benchmark_v1/"
            "anchor_expansion_review_queue_v0_2.csv"
        ),
    )
    parser.add_argument(
        "--drafts",
        type=Path,
        default=Path(
            "revision/benchmark/benchmark_v1/"
            "anchor_expansion_assistant_draft_v0_2.csv"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "experiments/phase7_formal_experiments/configs/"
            "benchmark_anchor_expansion_review_v0_2.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("revision/benchmark/benchmark_v1"),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    parent_rows = read_csv(args.queue)
    parent_hash = compute_sha256(args.queue)
    if args.mode == "prepare":
        batches = prepare_review_batches(parent_rows, config, parent_hash)
        result = write_prepared_batches(batches, args.output_dir)
    else:
        drafts = read_csv(args.drafts)
        validated = validate_assistant_drafts(drafts, parent_rows, config)
        result = write_review_outputs(
            validated,
            parent_queue_sha256=parent_hash,
            config=config,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
