from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from backend.prepare_kb_sources import compute_sha256


ALLOWED_SOURCE_STATUSES = {"approved", "indexed"}
REQUIRED_CONFIG_FIELDS = {
    "config_version",
    "parser_version",
    "chunker_version",
    "granularities",
    "min_text_chars",
    "max_text_chars",
    "title_only_max_chars",
    "scope_limited_evidence_types",
    "topic_terms",
}
PICTURE_PLACEHOLDER_RE = re.compile(
    r"picture\s*\[[^\]]*\]\s*intentionally\s+omitted|"
    r"start\s+of\s+picture\s+text|end\s+of\s+picture\s+text",
    re.IGNORECASE,
)
REFERENCE_HEADING_RE = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*)?(?:参考文献|references?)(?:\s|$)",
    re.IGNORECASE,
)
TOC_HEADING_RE = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*)?(?:目录|contents?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
TOC_ENTRY_RE = re.compile(r"\.{3,}\s*\d+")
FRONT_MATTER_METADATA_RE = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*)?"
    r"(?:[［\[]?关键词[］\]]?|中图分类号|文献标志码|文章编号|DOI\b)",
    re.IGNORECASE,
)


ChunkLoader = Callable[[dict[str, Any]], dict[int, list[Any]]]


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 根节点必须是对象: {path}")
    return payload


def load_candidate_config(path: str | Path) -> dict[str, Any]:
    """读取并校验候选片段主题词配置。"""
    config = _load_json(path)
    missing = sorted(REQUIRED_CONFIG_FIELDS - set(config))
    if missing:
        raise ValueError(f"候选抽取配置缺少字段: {', '.join(missing)}")
    if not config["granularities"]:
        raise ValueError("候选抽取配置 granularities 不能为空")
    if not isinstance(config["topic_terms"], dict) or not config["topic_terms"]:
        raise ValueError("候选抽取配置 topic_terms 必须是非空对象")
    return config


def load_source_coverage(
    path: str | Path,
    *,
    expected_count: int,
) -> list[dict[str, Any]]:
    """读取 B1.0 覆盖矩阵并执行准入校验。"""
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return validate_coverage_rows(rows, expected_count=expected_count)


def validate_coverage_rows(
    coverage_rows: list[dict[str, Any]],
    *,
    expected_count: int,
) -> list[dict[str, Any]]:
    """只允许通过 B1.0 身份、文件和哈希校验的正式来源。"""
    if len(coverage_rows) != expected_count:
        raise ValueError(
            "B1.0 来源数量不匹配: "
            f"expected={expected_count}, actual={len(coverage_rows)}"
        )

    source_ids = [row.get("source_id") for row in coverage_rows]
    duplicate_ids = sorted(
        source_id
        for source_id, count in Counter(source_ids).items()
        if source_id and count > 1
    )
    if duplicate_ids:
        raise ValueError(f"重复 source_id: {', '.join(duplicate_ids)}")

    verified_rows: list[dict[str, Any]] = []
    for row in coverage_rows:
        source_id = row.get("source_id", "<unknown>")
        checks = {
            "source_id": bool(row.get("source_id")),
            "included_in_kb": row.get("included_in_kb") is True,
            "status": row.get("status") in ALLOWED_SOURCE_STATUSES,
            "file_exists": row.get("file_exists") is True,
            "hash_matches": row.get("hash_matches") is True,
            "file_size_matches": row.get("file_size_matches") is True,
            "coverage_status": (
                row.get("coverage_status") == "pending_candidate_extraction"
            ),
        }
        failed_checks = [name for name, passed in checks.items() if not passed]
        if failed_checks:
            raise ValueError(
                f"{source_id} 未通过 B1.0: {', '.join(failed_checks)}"
            )
        verified_rows.append(row)
    return sorted(verified_rows, key=lambda row: row["source_id"])


def normalize_candidate_text(text: str) -> str:
    """规范化空白，保留医学符号和原始语义。"""
    return re.sub(r"\s+", " ", text).strip()


def _is_scope_limited_source(
    source: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    evidence_types = set(source.get("evidence_types") or [])
    limited_types = set(config["scope_limited_evidence_types"])
    return bool(evidence_types) and evidence_types.issubset(limited_types)


def _is_title_only(text: str, config: dict[str, Any]) -> bool:
    if len(text) > int(config["title_only_max_chars"]):
        return False
    plain = re.sub(r"^\s{0,3}#{1,6}\s*", "", text).strip()
    line_count = len([line for line in plain.splitlines() if line.strip()])
    has_sentence_signal = bool(re.search(r"[。；;.!！?？：:]", plain))
    has_dose_signal = bool(
        re.search(r"\d+(?:\.\d+)?\s*(?:mg|g|ml|μg|ug)(?:\s*/\s*kg)?", plain, re.I)
    )
    return line_count <= 2 and not has_sentence_signal and not has_dose_signal


def _is_noise_or_low_information(
    text: str,
    *,
    page_number: int,
    config: dict[str, Any],
) -> bool:
    if page_number <= 0:
        return True
    if len(text) < int(config["min_text_chars"]):
        return True
    if len(text) > int(config["max_text_chars"]):
        return True
    if PICTURE_PLACEHOLDER_RE.search(text):
        return True
    if REFERENCE_HEADING_RE.search(text):
        return True
    if TOC_HEADING_RE.search(text) and len(TOC_ENTRY_RE.findall(text)) >= 2:
        return True
    if FRONT_MATTER_METADATA_RE.search(text):
        return True
    return _is_title_only(text, config)


def _match_topics(
    text: str,
    config: dict[str, Any],
) -> tuple[list[str], dict[str, list[str]]]:
    folded_text = text.casefold()
    matched_terms: dict[str, list[str]] = {}
    for topic in sorted(config["topic_terms"]):
        terms = config["topic_terms"][topic]
        hits = sorted(
            {
                str(term)
                for term in terms
                if str(term).strip() and str(term).casefold() in folded_text
            },
            key=lambda term: term.casefold(),
        )
        if hits:
            matched_terms[topic] = hits
    return sorted(matched_terms), matched_terms


def _candidate_id(source_id: str, page_number: int, text: str) -> str:
    identity = f"{source_id}|{page_number}|{text}".encode("utf-8")
    return f"CAND-{hashlib.sha256(identity).hexdigest()[:20]}"


def _chunk_page_number(chunk: Any) -> int:
    value = getattr(getattr(chunk, "metadata", None), "page_number", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _chunk_block_type(chunk: Any) -> str:
    value = getattr(getattr(chunk, "metadata", None), "block_type", "unknown")
    return str(value or "unknown")


def build_anchor_candidates(
    coverage_rows: list[dict[str, Any]],
    config: dict[str, Any],
    chunk_loader: ChunkLoader,
    *,
    expected_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """构建待人工核验的候选片段，不产生任何 gold 字段。"""
    verified_rows = validate_coverage_rows(
        coverage_rows,
        expected_count=expected_count,
    )
    candidates: list[dict[str, Any]] = []
    source_results: list[dict[str, Any]] = []

    for source in verified_rows:
        source_candidates: list[dict[str, Any]] = []
        parsed_chunk_count = 0
        try:
            chunks_by_granularity = chunk_loader(source)
            ordered_chunks: list[Any] = []
            for granularity in config["granularities"]:
                chunks = list(chunks_by_granularity.get(int(granularity), []))
                parsed_chunk_count += len(chunks)
                ordered_chunks.extend(
                    sorted(
                        chunks,
                        key=lambda chunk: (
                            _chunk_page_number(chunk),
                            _chunk_block_type(chunk),
                            normalize_candidate_text(
                                str(getattr(chunk, "content", ""))
                            ),
                        ),
                    )
                )

            if not _is_scope_limited_source(source, config):
                seen: set[tuple[str, int, str]] = set()
                for chunk in ordered_chunks:
                    raw_text = normalize_candidate_text(
                        str(getattr(chunk, "content", ""))
                    )
                    page_number = _chunk_page_number(chunk)
                    if _is_noise_or_low_information(
                        raw_text,
                        page_number=page_number,
                        config=config,
                    ):
                        continue
                    matched_topics, matched_terms = _match_topics(raw_text, config)
                    if not matched_topics:
                        continue
                    dedup_key = (source["source_id"], page_number, raw_text)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)
                    source_candidates.append(
                        {
                            "candidate_id": _candidate_id(
                                source["source_id"],
                                page_number,
                                raw_text,
                            ),
                            "source_id": source["source_id"],
                            "source_title": source["title"],
                            "source_filename": source["filename"],
                            "source_sha256": source["actual_sha256"],
                            "page_number": page_number,
                            "block_type": _chunk_block_type(chunk),
                            "granularity": int(getattr(chunk, "granularity", 0)),
                            "raw_text": raw_text,
                            "matched_topics": matched_topics,
                            "matched_terms": matched_terms,
                            "review_status": "candidate_unverified",
                            "parser_version": config["parser_version"],
                            "chunker_version": config["chunker_version"],
                            "config_version": config["config_version"],
                        }
                    )
        except Exception as exc:
            source_results.append(
                {
                    "source_id": source["source_id"],
                    "candidate_count": 0,
                    "parsed_chunk_count": parsed_chunk_count,
                    "processing_status": "parse_failed",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        source_candidates.sort(
            key=lambda candidate: (
                candidate["page_number"],
                candidate["candidate_id"],
            )
        )
        candidates.extend(source_candidates)
        if source_candidates:
            status = "candidate_ready"
        elif _is_scope_limited_source(source, config):
            status = "scope_limited"
        else:
            status = "no_candidate_after_filtering"
        source_results.append(
            {
                "source_id": source["source_id"],
                "candidate_count": len(source_candidates),
                "parsed_chunk_count": parsed_chunk_count,
                "processing_status": status,
                "failure_reason": "",
            }
        )

    candidates.sort(
        key=lambda candidate: (
            candidate["source_id"],
            candidate["page_number"],
            candidate["candidate_id"],
        )
    )
    source_results.sort(key=lambda result: result["source_id"])
    return candidates, source_results


def write_candidate_outputs(
    candidates: list[dict[str, Any]],
    source_results: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    coverage_sha256: str,
    expected_count: int,
) -> dict[str, str]:
    """稳定写出候选 JSONL 和逐来源处理报告。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_path / "anchor_candidates_v0_1.jsonl"
    report_path = output_path / "anchor_candidates_summary_v0_1.md"

    jsonl_path.write_text(
        "".join(
            json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n"
            for candidate in candidates
        ),
        encoding="utf-8",
    )
    status_counts = Counter(
        result["processing_status"] for result in source_results
    )
    report_lines = [
        "# Benchmark-v1 Anchor Candidate Summary v0.1",
        "",
        f"- 正式来源处理数量：{len(source_results)}/{expected_count}",
        f"- 候选片段数量：{len(candidates)}",
        f"- 候选默认状态：`candidate_unverified`",
        f"- B1.0 coverage SHA-256：`{coverage_sha256}`",
        f"- config version：`{config['config_version']}`",
        f"- parser version：`{config['parser_version']}`",
        f"- chunker version：`{config['chunker_version']}`",
        "- 外部 API 调用：0",
        "- 估算费用：0",
        "- 医学边界：关键词命中仅用于建立人工核验队列，不代表医学正确性，也不构成 gold evidence",
        "",
        "## 状态汇总",
        "",
    ]
    report_lines.extend(
        f"- `{status}`：{status_counts[status]}"
        for status in sorted(status_counts)
    )
    report_lines.extend(
        [
            "",
            "## 逐来源结果",
            "",
            "| source_id | parsed_chunks | candidates | status | failure_reason |",
            "|---|---:|---:|---|---|",
        ]
    )
    report_lines.extend(
        (
            f"| {result['source_id']} | {result['parsed_chunk_count']} | "
            f"{result['candidate_count']} | {result['processing_status']} | "
            f"{result['failure_reason'] or '-'} |"
        )
        for result in source_results
    )
    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    return {"jsonl": str(jsonl_path), "report": str(report_path)}


def make_pdf_chunk_loader(
    formal_dir: str | Path,
    parser: Any,
    chunker: Any,
) -> ChunkLoader:
    """创建带文件哈希复核的本地 PDF 解析函数。"""
    formal_path = Path(formal_dir)

    def load(source: dict[str, Any]) -> dict[int, list[Any]]:
        pdf_path = formal_path / source["filename"]
        if not pdf_path.is_file():
            raise FileNotFoundError(f"正式 PDF 不存在: {pdf_path}")
        actual_sha256 = compute_sha256(pdf_path)
        if actual_sha256 != source["actual_sha256"]:
            raise ValueError(
                f"PDF SHA-256 漂移: expected={source['actual_sha256']}, "
                f"actual={actual_sha256}"
            )
        blocks = parser.parse(pdf_path)
        return chunker.chunk_all_granularities(blocks)

    return load


def run_candidate_extraction(
    coverage_path: str | Path,
    formal_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    *,
    expected_count: int,
) -> dict[str, Any]:
    """对正式来源执行本地候选抽取并写出可审计产物。"""
    from app.knowledge.chunker import SemanticChunker
    from app.knowledge.parser import DualTrackMedicalParser

    coverage_rows = load_source_coverage(
        coverage_path,
        expected_count=expected_count,
    )
    config = load_candidate_config(config_path)
    candidates, source_results = build_anchor_candidates(
        coverage_rows,
        config,
        make_pdf_chunk_loader(
            formal_dir,
            DualTrackMedicalParser(),
            SemanticChunker(),
        ),
        expected_count=expected_count,
    )
    output_files = write_candidate_outputs(
        candidates,
        source_results,
        output_dir,
        config=config,
        coverage_sha256=compute_sha256(coverage_path),
        expected_count=expected_count,
    )
    status_counts = Counter(
        result["processing_status"] for result in source_results
    )
    return {
        "source_count": len(source_results),
        "expected_count": expected_count,
        "candidate_count": len(candidates),
        "status_counts": dict(sorted(status_counts.items())),
        "all_sources_processed": status_counts.get("parse_failed", 0) == 0,
        "output_files": output_files,
        "external_api_calls": 0,
        "estimated_cost": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 Benchmark-v1 正式资料中抽取待人工核验候选片段"
    )
    parser.add_argument(
        "--coverage",
        type=Path,
        default=Path(
            "revision/benchmark/benchmark_v1/source_coverage_matrix_v0_1.jsonl"
        ),
    )
    parser.add_argument(
        "--formal-dir",
        type=Path,
        default=Path("data/guidelines"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "experiments/phase7_formal_experiments/configs/"
            "benchmark_anchor_search_v0_1.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("revision/benchmark/benchmark_v1"),
    )
    parser.add_argument("--expected-count", type=int, default=22)
    args = parser.parse_args()
    summary = run_candidate_extraction(
        args.coverage,
        args.formal_dir,
        args.config,
        args.output_dir,
        expected_count=args.expected_count,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
