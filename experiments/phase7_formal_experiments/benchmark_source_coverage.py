from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from backend.prepare_kb_sources import compute_sha256, load_manifest


REQUIRED_SOURCE_FIELDS = {
    "source_id",
    "title",
    "year",
    "publisher",
    "jurisdiction",
    "source_type",
    "authority_level",
    "filename",
    "status",
    "included_in_kb",
    "evidence_types",
    "can_support",
    "cannot_support",
    "content_check",
    "file_size",
    "sha256",
}
ALLOWED_STATUSES = {"approved", "indexed"}
COVERAGE_FIELDS = [
    "source_id",
    "title",
    "year",
    "publisher",
    "jurisdiction",
    "source_type",
    "authority_level",
    "filename",
    "status",
    "included_in_kb",
    "evidence_types",
    "can_support",
    "cannot_support",
    "content_check_status",
    "recorded_file_size",
    "actual_file_size",
    "file_size_matches",
    "recorded_sha256",
    "actual_sha256",
    "file_exists",
    "hash_matches",
    "candidate_anchor_count",
    "verified_anchor_count",
    "coverage_status",
    "scope_notes",
]


def validate_source_entry(source: dict[str, Any]) -> None:
    """校验正式资料记录的必要字段和准入状态。"""
    missing_fields = sorted(
        field
        for field in REQUIRED_SOURCE_FIELDS
        if field not in source or source[field] is None
    )
    if missing_fields:
        source_id = source.get("source_id", "<unknown>")
        raise ValueError(
            f"{source_id} 缺少必要字段: {', '.join(missing_fields)}"
        )
    if source["included_in_kb"] is not True:
        raise ValueError(f"{source['source_id']} 未标记 included_in_kb=true")
    if source["status"] not in ALLOWED_STATUSES:
        raise ValueError(
            f"{source['source_id']} 无效准入状态: {source['status']}"
        )


def build_source_coverage(
    manifest_path: str | Path,
    formal_dir: str | Path,
    *,
    expected_count: int,
) -> list[dict[str, Any]]:
    """构建正式知识库资料的确定性覆盖记录。"""
    manifest = load_manifest(manifest_path)
    included_sources = sorted(
        (
            source
            for source in manifest["sources"]
            if source.get("included_in_kb") is True
        ),
        key=lambda source: source["source_id"],
    )
    if len(included_sources) != expected_count:
        raise ValueError(
            f"正式知识库来源数量不匹配: expected={expected_count}, "
            f"actual={len(included_sources)}"
        )

    formal_path = Path(formal_dir)
    rows: list[dict[str, Any]] = []
    for source in included_sources:
        validate_source_entry(source)
        pdf_path = formal_path / source["filename"]
        if not pdf_path.is_file():
            raise FileNotFoundError(
                f"{source['source_id']} 对应 PDF 不存在: {pdf_path}"
            )
        actual_sha256 = compute_sha256(pdf_path)
        if actual_sha256 != source["sha256"]:
            raise ValueError(
                f"{source['source_id']} SHA-256 不匹配: "
                f"expected={source['sha256']}, actual={actual_sha256}"
            )
        actual_file_size = pdf_path.stat().st_size
        rows.append(
            {
                "source_id": source["source_id"],
                "title": source["title"],
                "year": source["year"],
                "publisher": source["publisher"],
                "jurisdiction": source["jurisdiction"],
                "source_type": source["source_type"],
                "authority_level": source["authority_level"],
                "filename": source["filename"],
                "status": source["status"],
                "included_in_kb": source["included_in_kb"],
                "evidence_types": source["evidence_types"],
                "can_support": source["can_support"],
                "cannot_support": source["cannot_support"],
                "content_check_status": source["content_check"].get("status"),
                "recorded_file_size": source["file_size"],
                "actual_file_size": actual_file_size,
                "file_size_matches": actual_file_size == source["file_size"],
                "recorded_sha256": source["sha256"],
                "file_exists": True,
                "actual_sha256": actual_sha256,
                "hash_matches": True,
                "candidate_anchor_count": 0,
                "verified_anchor_count": 0,
                "coverage_status": "pending_candidate_extraction",
                "scope_notes": "",
            }
        )
    return rows


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_coverage_outputs(
    rows: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    source_manifest_sha256: str,
    expected_count: int,
) -> dict[str, str]:
    """稳定写出 JSONL、CSV 和 Markdown 覆盖审计产物。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_path / "source_coverage_matrix_v0_1.jsonl"
    csv_path = output_path / "source_coverage_matrix_v0_1.csv"
    report_path = output_path / "source_coverage_audit_v0_1.md"

    jsonl_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=COVERAGE_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: _csv_value(row.get(field)) for field in COVERAGE_FIELDS}
            )

    report_lines = [
        "# Benchmark-v1 Source Coverage Audit v0.1",
        "",
        f"- 正式来源数量：{len(rows)}/{expected_count}",
        f"- source manifest SHA-256：`{source_manifest_sha256}`",
        "- 准入状态：仅接受 `approved` / `indexed`",
        "- 外部 API 调用：0",
        "- 当前阶段：仅完成资料身份与文件完整性校验，尚未抽取或核验证据锚点",
        "",
        "## 资料明细",
        "",
        "| source_id | status | file | SHA-256 | coverage_status |",
        "|---|---|---|---|---|",
    ]
    report_lines.extend(
        (
            f"| {row['source_id']} | {row['status']} | {row['filename']} | "
            f"{'match' if row['hash_matches'] else 'mismatch'} | "
            f"{row['coverage_status']} |"
        )
        for row in rows
    )
    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    return {
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "report": str(report_path),
    }


def run_source_coverage_audit(
    manifest_path: str | Path,
    formal_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_count: int,
) -> dict[str, Any]:
    """校验正式资料并写出可复现的覆盖矩阵。"""
    rows = build_source_coverage(
        manifest_path,
        formal_dir,
        expected_count=expected_count,
    )
    source_manifest_sha256 = compute_sha256(manifest_path)
    output_files = write_coverage_outputs(
        rows,
        output_dir,
        source_manifest_sha256=source_manifest_sha256,
        expected_count=expected_count,
    )
    return {
        "source_count": len(rows),
        "expected_count": expected_count,
        "all_hashes_match": all(row["hash_matches"] for row in rows),
        "all_file_sizes_match": all(
            row["file_size_matches"] for row in rows
        ),
        "manifest_sha256": source_manifest_sha256,
        "output_files": output_files,
        "external_api_calls": 0,
        "estimated_cost": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="构建 Benchmark-v1 正式资料覆盖矩阵"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/guidelines/source_manifest.json"),
    )
    parser.add_argument(
        "--formal-dir",
        type=Path,
        default=Path("data/guidelines"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("revision/benchmark/benchmark_v1"),
    )
    parser.add_argument("--expected-count", type=int, default=22)
    args = parser.parse_args()
    summary = run_source_coverage_audit(
        args.manifest,
        args.formal_dir,
        args.output_dir,
        expected_count=args.expected_count,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
