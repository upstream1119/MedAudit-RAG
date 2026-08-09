from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REQUIRED_CONFIG_FIELDS = {
    "config_version",
    "expected_source_count",
    "max_pending_per_source",
    "preferred_granularities",
    "min_verified_text_chars",
    "min_alnum_ratio",
    "scope_limited_evidence_types",
}
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
JSON_FIELD_TYPES = {
    "matched_topics": list,
    "matched_terms": dict,
    "dev50_overlap_anchor_ids": list,
    "supported_claim_types": list,
}
REVIEW_QUEUE_FIELDS = [
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
    *AUTHOR_REVIEW_FIELDS,
]
COVERAGE_V0_2_EXTRA_FIELDS = [
    "review_queue_count",
    "dev50_overlap_rejected_count",
    "review_config_version",
]
DEV50_ROW_RE = re.compile(
    r"^\|\s*([^|]+?)\s*\|\s*(SRC-\d+)\s*\|\s*(\d+)\s*\|"
)
WHITESPACE_RE = re.compile(r"\s+")
DOT_LEADER_RE = re.compile(r"(?:\.{3,}|…{3,})")
PROHIBITED_TEXT_RE = re.compile(
    r"picture\s*\[[^\]]*\]\s*intentionally\s+omitted|"
    r"start\s+of\s+picture\s+text|end\s+of\s+picture\s+text",
    re.IGNORECASE,
)
GARBLED_AUTHOR_METADATA_RE = re.compile(r"\?{2,}|�")
READABLE_AUTHOR_METADATA_FIELDS = (
    "author_review_reason",
    "evidence_scope",
    "age_scope",
    "applicability_conditions",
)


def _compute_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 根节点必须是对象: {path}")
    return payload


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_review_config(path: str | Path) -> dict[str, Any]:
    config = _load_json(path)
    missing = sorted(REQUIRED_CONFIG_FIELDS - set(config))
    if missing:
        raise ValueError(f"锚点核验配置缺少字段: {', '.join(missing)}")
    if int(config["max_pending_per_source"]) <= 0:
        raise ValueError("max_pending_per_source 必须大于 0")
    if not config["preferred_granularities"]:
        raise ValueError("preferred_granularities 不能为空")
    return config


def load_dev50_anchor_pairs(
    registry_path: str | Path,
) -> dict[tuple[str, int], list[str]]:
    """读取 Dev50 已使用的 source/page，作为正式 Benchmark 泄漏隔离表。"""
    pairs: dict[tuple[str, int], list[str]] = defaultdict(list)
    for line in Path(registry_path).read_text(encoding="utf-8").splitlines():
        match = DEV50_ROW_RE.match(line)
        if not match:
            continue
        anchor_id, source_id, page_number = match.groups()
        pairs[(source_id, int(page_number))].append(anchor_id.strip())
    return {
        pair: sorted(set(anchor_ids))
        for pair, anchor_ids in sorted(pairs.items())
    }


def _normalize_text(text: Any) -> str:
    return WHITESPACE_RE.sub(" ", str(text or "")).strip()


def _is_scope_limited_source(
    source: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    evidence_types = set(source.get("evidence_types") or [])
    limited_types = set(config["scope_limited_evidence_types"])
    return bool(evidence_types) and evidence_types.issubset(limited_types)


def _validate_candidate_provenance(
    candidate: dict[str, Any],
    source: dict[str, Any],
) -> None:
    expected = {
        "source_title": source.get("title"),
        "source_filename": source.get("filename"),
        "source_sha256": source.get("actual_sha256"),
    }
    mismatches = [
        field
        for field, value in expected.items()
        if candidate.get(field) != value
    ]
    if candidate.get("review_status") != "candidate_unverified":
        mismatches.append("review_status")
    try:
        page_number = int(candidate.get("page_number", 0))
    except (TypeError, ValueError):
        page_number = 0
    if page_number <= 0:
        mismatches.append("page_number")
    if mismatches:
        raise ValueError(
            f"{candidate.get('candidate_id', '<unknown>')} 候选来源校验失败: "
            f"{', '.join(sorted(set(mismatches)))}"
        )


def _candidate_rank_key(
    candidate: dict[str, Any],
    config: dict[str, Any],
) -> tuple[Any, ...]:
    preferred = [int(value) for value in config["preferred_granularities"]]
    try:
        granularity_rank = preferred.index(int(candidate.get("granularity", 0)))
    except ValueError:
        granularity_rank = len(preferred)
    text = _normalize_text(candidate.get("raw_text"))
    topic_count = len(candidate.get("matched_topics") or [])
    return (
        granularity_rank,
        abs(topic_count - 1),
        abs(len(text) - 600),
        int(candidate["page_number"]),
        candidate["candidate_id"],
    )


def _is_shortlist_noise(candidate: dict[str, Any]) -> bool:
    text = _normalize_text(candidate.get("raw_text"))
    return len(DOT_LEADER_RE.findall(text)) >= 2


def _select_diverse_candidates(
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    limit = int(config["max_pending_per_source"])
    ranked = sorted(candidates, key=lambda row: _candidate_rank_key(row, config))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_pages: set[int] = set()
    selected_topics: set[str] = set()

    def add(candidate: dict[str, Any]) -> None:
        selected.append(candidate)
        selected_ids.add(candidate["candidate_id"])
        selected_pages.add(int(candidate["page_number"]))
        selected_topics.update(candidate.get("matched_topics") or [])

    for candidate in ranked:
        topics = set(candidate.get("matched_topics") or [])
        if (
            len(selected) < limit
            and int(candidate["page_number"]) not in selected_pages
            and bool(topics - selected_topics)
        ):
            add(candidate)
    for candidate in ranked:
        if len(selected) >= limit:
            break
        if (
            candidate["candidate_id"] not in selected_ids
            and int(candidate["page_number"]) not in selected_pages
        ):
            add(candidate)
    for candidate in ranked:
        if len(selected) >= limit:
            break
        if candidate["candidate_id"] not in selected_ids:
            add(candidate)
    return selected


def _context_candidate(
    candidate: dict[str, Any],
    page_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    return max(
        page_candidates,
        key=lambda row: (
            len(_normalize_text(row.get("raw_text"))),
            int(row.get("granularity", 0)),
            row["candidate_id"],
        ),
        default=candidate,
    )


def _review_row(
    candidate: dict[str, Any],
    *,
    context: dict[str, Any],
    review_status: str,
    overlap_anchor_ids: list[str],
    selection_rank: int | str,
    config: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "candidate_id": candidate["candidate_id"],
        "source_id": candidate["source_id"],
        "source_title": candidate["source_title"],
        "source_filename": candidate["source_filename"],
        "source_sha256": candidate["source_sha256"],
        "page_number": int(candidate["page_number"]),
        "block_type": candidate.get("block_type", "unknown"),
        "granularity": int(candidate.get("granularity", 0)),
        "candidate_text": _normalize_text(candidate.get("raw_text")),
        "context_candidate_id": context["candidate_id"],
        "context_text": _normalize_text(context.get("raw_text")),
        "matched_topics": list(candidate.get("matched_topics") or []),
        "matched_terms": dict(candidate.get("matched_terms") or {}),
        "review_status": review_status,
        "dev50_overlap_anchor_ids": list(overlap_anchor_ids),
        "selection_rank_within_source": selection_rank,
        "parser_version": candidate.get("parser_version", ""),
        "chunker_version": candidate.get("chunker_version", ""),
        "candidate_config_version": candidate.get("config_version", ""),
        "review_config_version": config["config_version"],
    }
    row.update({field: "" for field in AUTHOR_REVIEW_FIELDS})
    return row


def build_review_queue(
    candidates: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    dev50_pairs: dict[tuple[str, int], list[str]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """压缩候选池，保留泄漏拒绝轨迹并生成待作者核验队列。"""
    source_map = {row["source_id"]: row for row in coverage_rows}
    if len(source_map) != len(coverage_rows):
        raise ValueError("coverage_rows 含重复 source_id")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    page_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        source_id = candidate.get("source_id")
        if source_id not in source_map:
            raise ValueError(f"候选引用未知 source_id: {source_id}")
        _validate_candidate_provenance(candidate, source_map[source_id])
        grouped[source_id].append(candidate)
        page_groups[(source_id, int(candidate["page_number"]))].append(candidate)

    queue: list[dict[str, Any]] = []
    source_results: list[dict[str, Any]] = []
    for source in sorted(coverage_rows, key=lambda row: row["source_id"]):
        source_id = source["source_id"]
        source_candidates = sorted(
            grouped.get(source_id, []),
            key=lambda row: (
                int(row["page_number"]),
                row["candidate_id"],
            ),
        )
        overlap_candidates: list[dict[str, Any]] = []
        eligible_candidates: list[dict[str, Any]] = []
        for candidate in source_candidates:
            pair = (source_id, int(candidate["page_number"]))
            if pair in dev50_pairs:
                overlap_candidates.append(candidate)
            else:
                eligible_candidates.append(candidate)

        shortlist_candidates = [
            candidate
            for candidate in eligible_candidates
            if not _is_shortlist_noise(candidate)
        ]
        selected = _select_diverse_candidates(shortlist_candidates, config)
        for rank, candidate in enumerate(selected, start=1):
            pair = (source_id, int(candidate["page_number"]))
            queue.append(
                _review_row(
                    candidate,
                    context=_context_candidate(candidate, page_groups[pair]),
                    review_status="pending_author_review",
                    overlap_anchor_ids=[],
                    selection_rank=rank,
                    config=config,
                )
            )
        for candidate in overlap_candidates:
            pair = (source_id, int(candidate["page_number"]))
            queue.append(
                _review_row(
                    candidate,
                    context=_context_candidate(candidate, page_groups[pair]),
                    review_status="rejected_dev50_overlap",
                    overlap_anchor_ids=dev50_pairs[pair],
                    selection_rank="",
                    config=config,
                )
            )

        if selected:
            processing_status = "pending_author_review"
            scope_notes = ""
        elif _is_scope_limited_source(source, config):
            processing_status = "scope_limited"
            scope_notes = (
                "资料仅支持药物身份、剂型或目录状态，不强制构造儿科临床锚点。"
            )
        elif overlap_candidates and not eligible_candidates:
            processing_status = "dev50_overlap_only"
            scope_notes = "全部候选与 Dev50 source/page 重叠。"
        else:
            processing_status = "no_candidate_after_shortlisting"
            scope_notes = "候选筛选后无可进入人工核验的片段。"
        source_results.append(
            {
                "source_id": source_id,
                "candidate_count": len(source_candidates),
                "pending_review_count": len(selected),
                "dev50_overlap_rejected_count": len(overlap_candidates),
                "processing_status": processing_status,
                "scope_notes": scope_notes,
            }
        )

    status_order = {"pending_author_review": 0, "rejected_dev50_overlap": 1}
    queue.sort(
        key=lambda row: (
            row["source_id"],
            status_order[row["review_status"]],
            int(row["page_number"]),
            row["candidate_id"],
        )
    )
    return queue, source_results


def _parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError("字段必须是 JSON 数组")
    return parsed


def _parse_json_field(value: Any, expected_type: type) -> Any:
    if isinstance(value, expected_type):
        return value
    if value in (None, ""):
        return expected_type()
    parsed = json.loads(str(value))
    if not isinstance(parsed, expected_type):
        raise ValueError(f"字段必须是 JSON {expected_type.__name__}")
    return parsed


def _text_is_traceable_and_readable(
    verified_text: str,
    review: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    text = _normalize_text(verified_text)
    if len(text) < int(config["min_verified_text_chars"]):
        return False
    if "�" in text or PROHIBITED_TEXT_RE.search(text):
        return False
    alnum_count = sum(character.isalnum() for character in text)
    if alnum_count / max(len(text), 1) < float(config["min_alnum_ratio"]):
        return False
    candidate_text = _normalize_text(review.get("candidate_text"))
    context_text = _normalize_text(review.get("context_text"))
    return text in candidate_text or text in context_text


def _author_metadata_is_readable(review: dict[str, Any]) -> bool:
    return all(
        not GARBLED_AUTHOR_METADATA_RE.search(str(review.get(field, "")))
        for field in READABLE_AUTHOR_METADATA_FIELDS
    )


def _anchor_id(source_id: str, page_number: int, text: str) -> str:
    identity = f"{source_id}|{page_number}|{_normalize_text(text)}".encode("utf-8")
    return f"ANCH-{hashlib.sha256(identity).hexdigest()[:20]}"


def _decision(
    candidate_id: str,
    status: str,
    reason: str,
) -> dict[str, str]:
    return {
        "candidate_id": candidate_id,
        "promotion_status": status,
        "reason": reason,
    }


def promote_verified_anchors(
    review_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    page_counts: dict[str, int],
    dev50_pairs: dict[tuple[str, int], list[str]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """只将逐页核验、范围明确且无泄漏的记录升级为正式锚点。"""
    source_map = {row["source_id"]: row for row in coverage_rows}
    candidate_map = {row["candidate_id"]: row for row in candidates}
    anchors: list[dict[str, Any]] = []
    decisions: list[dict[str, str]] = []
    seen_anchor_ids: set[str] = set()

    for review in sorted(
        review_rows,
        key=lambda row: (row.get("source_id", ""), row.get("candidate_id", "")),
    ):
        candidate_id = str(review.get("candidate_id", ""))
        candidate = candidate_map.get(candidate_id)
        source = source_map.get(review.get("source_id"))
        if candidate is None or source is None:
            decisions.append(
                _decision(candidate_id, "rejected_provenance_mismatch", "未知候选或来源")
            )
            continue
        try:
            page_number = int(review.get("page_number", 0))
        except (TypeError, ValueError):
            page_number = 0
        pair = (source["source_id"], page_number)
        if pair in dev50_pairs or review.get("review_status") == "rejected_dev50_overlap":
            decisions.append(
                _decision(candidate_id, "rejected_dev50_overlap", "与 Dev50 source/page 重叠")
            )
            continue

        missing_fields = [
            field
            for field in AUTHOR_REVIEW_FIELDS
            if review.get(field) in (None, "", [])
        ]
        if missing_fields:
            decisions.append(
                _decision(
                    candidate_id,
                    "rejected_incomplete_review",
                    f"人工字段不完整: {', '.join(missing_fields)}",
                )
            )
            continue
        if not _author_metadata_is_readable(review):
            decisions.append(
                _decision(
                    candidate_id,
                    "rejected_metadata_quality",
                    "人工核验元数据含乱码或编码损坏",
                )
            )
            continue
        if (
            review.get("author_review_outcome") != "accepted"
            or review.get("scope_check") != "within_can_support"
        ):
            decisions.append(
                _decision(candidate_id, "rejected_scope", "作者结论或支持范围不允许升级")
            )
            continue

        provenance_checks = {
            "source_id": review.get("source_id") == candidate.get("source_id"),
            "source_title": (
                review.get("source_title")
                == candidate.get("source_title")
                == source.get("title")
            ),
            "source_filename": (
                review.get("source_filename")
                == candidate.get("source_filename")
                == source.get("filename")
            ),
            "source_sha256": (
                review.get("source_sha256")
                == candidate.get("source_sha256")
                == source.get("actual_sha256")
            ),
            "page_number": (
                page_number == int(candidate.get("page_number", 0))
                and 0 < page_number <= int(page_counts.get(source["source_id"], 0))
            ),
        }
        failed_provenance = [
            field for field, passed in provenance_checks.items() if not passed
        ]
        if failed_provenance:
            decisions.append(
                _decision(
                    candidate_id,
                    "rejected_provenance_mismatch",
                    f"来源字段不一致: {', '.join(failed_provenance)}",
                )
            )
            continue

        verified_text = _normalize_text(review["verified_text_span"])
        try:
            supported_claim_types = _parse_list(review["supported_claim_types"])
        except (TypeError, ValueError, json.JSONDecodeError):
            supported_claim_types = []
        if (
            not supported_claim_types
            or not _text_is_traceable_and_readable(verified_text, review, config)
        ):
            decisions.append(
                _decision(candidate_id, "rejected_text_quality", "文本不可追溯、过短、乱码或声明类型为空")
            )
            continue

        anchor_id = _anchor_id(source["source_id"], page_number, verified_text)
        if anchor_id in seen_anchor_ids:
            decisions.append(
                _decision(candidate_id, "rejected_duplicate_anchor", "锚点 source/page/text 重复")
            )
            continue
        seen_anchor_ids.add(anchor_id)
        anchors.append(
            {
                "anchor_id": anchor_id,
                "candidate_id": candidate_id,
                "source_id": source["source_id"],
                "source_title": source["title"],
                "source_filename": source["filename"],
                "source_sha256": source["actual_sha256"],
                "page_number": page_number,
                "text_span": verified_text,
                "supported_claim_types": supported_claim_types,
                "evidence_scope": review["evidence_scope"],
                "age_scope": review["age_scope"],
                "applicability_conditions": review["applicability_conditions"],
                "scope_check": review["scope_check"],
                "reviewer_id": review["reviewer_id"],
                "author_reviewed_at": review["author_reviewed_at"],
                "author_review_reason": review["author_review_reason"],
                "verification_status": "author_verified_anchor",
                "parser_version": candidate.get("parser_version", ""),
                "chunker_version": candidate.get("chunker_version", ""),
                "candidate_config_version": candidate.get("config_version", ""),
                "review_config_version": config["config_version"],
            }
        )
        decisions.append(_decision(candidate_id, "promoted", "通过全部升级校验"))

    anchors.sort(key=lambda row: row["anchor_id"])
    decisions.sort(key=lambda row: row["candidate_id"])
    return anchors, decisions


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _read_review_queue(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    for row in rows:
        for field, expected_type in JSON_FIELD_TYPES.items():
            if field in row:
                row[field] = _parse_json_field(row[field], expected_type)
        row["page_number"] = int(row.get("page_number") or 0)
        row["granularity"] = int(row.get("granularity") or 0)
    return rows


def _write_review_queue(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=REVIEW_QUEUE_FIELDS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: _csv_value(row.get(field, "")) for field in REVIEW_QUEUE_FIELDS}
            )


def _write_anchor_pool(anchors: list[dict[str, Any]], path: Path) -> None:
    path.write_text(
        "".join(
            json.dumps(anchor, ensure_ascii=False, sort_keys=True) + "\n"
            for anchor in anchors
        ),
        encoding="utf-8",
    )


def _coverage_fields(rows: list[dict[str, Any]]) -> list[str]:
    base_fields = list(rows[0]) if rows else []
    return base_fields + [
        field for field in COVERAGE_V0_2_EXTRA_FIELDS if field not in base_fields
    ]


def _build_coverage_v0_2(
    coverage_rows: list[dict[str, Any]],
    source_results: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    result_map = {row["source_id"]: row for row in source_results}
    anchor_counts = Counter(anchor["source_id"] for anchor in anchors)
    updated: list[dict[str, Any]] = []
    for source in sorted(coverage_rows, key=lambda row: row["source_id"]):
        result = result_map[source["source_id"]]
        verified_count = anchor_counts[source["source_id"]]
        if verified_count:
            coverage_status = "verified_anchor_ready"
        else:
            coverage_status = result["processing_status"]
        row = dict(source)
        row.update(
            {
                "candidate_anchor_count": result["candidate_count"],
                "verified_anchor_count": verified_count,
                "coverage_status": coverage_status,
                "scope_notes": result["scope_notes"],
                "review_queue_count": result["pending_review_count"],
                "dev50_overlap_rejected_count": result[
                    "dev50_overlap_rejected_count"
                ],
                "review_config_version": config["config_version"],
            }
        )
        updated.append(row)
    return updated


def _write_coverage_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = _coverage_fields(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def _write_report(
    path: Path,
    *,
    queue: list[dict[str, Any]],
    source_results: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    decisions: list[dict[str, str]],
    config: dict[str, Any],
    candidate_sha256: str,
    stage: str,
) -> None:
    queue_counts = Counter(row["review_status"] for row in queue)
    pending_author_review_count = sum(
        1
        for row in queue
        if row["review_status"] == "pending_author_review"
        and not row.get("author_reviewed_at")
    )
    decision_counts = Counter(row["promotion_status"] for row in decisions)
    report_lines = [
        "# Benchmark-v1 Anchor Verification Report v0.1",
        "",
        f"- 当前阶段：`{stage}`",
        f"- 正式来源覆盖：{len(source_results)}/{config['expected_source_count']}",
        f"- 待作者逐页核验：{pending_author_review_count}",
        f"- Dev50 重叠自动隔离：{queue_counts.get('rejected_dev50_overlap', 0)}",
        f"- 已升级 author-verified anchors：{len(anchors)}",
        f"- 候选池 SHA-256：`{candidate_sha256}`",
        f"- review config version：`{config['config_version']}`",
        "- 外部 API 调用：0",
        "- input/output tokens：0/0",
        "- estimated cost：0",
        "- 医学边界：候选与队列均不等于 gold evidence，只有人工逐页核验且通过升级门的片段才可进入锚点池。",
        "",
        "## 人工检查点",
        "",
    ]
    if anchors:
        report_lines.append("已执行升级校验；仍需确认所有拒绝原因和来源覆盖状态。")
    else:
        report_lines.append(
            "当前锚点池为空。请逐页核对 review queue，填写全部作者字段后再运行 `promote`。"
        )
    report_lines.extend(
        [
            "",
            "## 逐来源状态",
            "",
            "| source_id | candidates | pending | dev50_overlap | status | scope_notes |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    report_lines.extend(
        (
            f"| {row['source_id']} | {row['candidate_count']} | "
            f"{row['pending_review_count']} | "
            f"{row['dev50_overlap_rejected_count']} | "
            f"{row['processing_status']} | {row['scope_notes'] or '-'} |"
        )
        for row in source_results
    )
    if decision_counts:
        report_lines.extend(["", "## 升级校验结果", ""])
        report_lines.extend(
            f"- `{status}`：{decision_counts[status]}"
            for status in sorted(decision_counts)
        )
    path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def _build_page_counts(
    coverage_rows: list[dict[str, Any]],
    formal_dir: str | Path,
) -> dict[str, int]:
    import fitz

    formal_path = Path(formal_dir)
    page_counts: dict[str, int] = {}
    for source in coverage_rows:
        pdf_path = formal_path / source["filename"]
        if not pdf_path.is_file():
            raise FileNotFoundError(f"正式 PDF 不存在: {pdf_path}")
        if _compute_sha256(pdf_path) != source["actual_sha256"]:
            raise ValueError(f"{source['source_id']} PDF SHA-256 漂移")
        with fitz.open(pdf_path) as document:
            page_counts[source["source_id"]] = len(document)
    return page_counts


def _rebuild_source_results(
    coverage_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    queue: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    candidate_counts = Counter(row["source_id"] for row in candidates)
    pending_counts = Counter(
        row["source_id"]
        for row in queue
        if row["review_status"] == "pending_author_review"
        and not row.get("author_reviewed_at")
    )
    reviewed_counts = Counter(
        row["source_id"]
        for row in queue
        if row["review_status"] == "pending_author_review"
        and row.get("author_reviewed_at")
    )
    overlap_counts = Counter(
        row["source_id"]
        for row in queue
        if row["review_status"] == "rejected_dev50_overlap"
    )
    results: list[dict[str, Any]] = []
    for source in sorted(coverage_rows, key=lambda row: row["source_id"]):
        source_id = source["source_id"]
        if pending_counts[source_id]:
            status = "pending_author_review"
            notes = ""
        elif reviewed_counts[source_id]:
            status = "author_review_complete"
            notes = "本来源候选已完成作者逐页核验。"
        elif _is_scope_limited_source(source, config):
            status = "scope_limited"
            notes = "资料仅支持药物身份、剂型或目录状态，不强制构造儿科临床锚点。"
        else:
            status = "no_candidate_after_shortlisting"
            notes = "候选筛选后无可进入人工核验的片段。"
        results.append(
            {
                "source_id": source_id,
                "candidate_count": candidate_counts[source_id],
                "pending_review_count": pending_counts[source_id],
                "dev50_overlap_rejected_count": overlap_counts[source_id],
                "processing_status": status,
                "scope_notes": notes,
            }
        )
    return results


def run_prepare(
    candidates_path: str | Path,
    coverage_path: str | Path,
    dev50_registry_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config = load_review_config(config_path)
    candidates = _load_jsonl(candidates_path)
    coverage_rows = _load_jsonl(coverage_path)
    if len(coverage_rows) != int(config["expected_source_count"]):
        raise ValueError("B1.0 coverage 来源数量与 review config 不一致")
    dev50_pairs = load_dev50_anchor_pairs(dev50_registry_path)
    queue, source_results = build_review_queue(
        candidates,
        coverage_rows,
        dev50_pairs,
        config,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    queue_path = output_path / "anchor_review_queue_v0_1.csv"
    if queue_path.exists():
        existing_rows = _read_review_queue(queue_path)
        if any(row.get("author_reviewed_at") for row in existing_rows):
            raise ValueError("现有 review queue 已含人工核验结果，拒绝覆盖")
    pool_path = output_path / "evidence_anchor_pool_v0_1.jsonl"
    coverage_v0_2_path = output_path / "source_coverage_matrix_v0_2.csv"
    report_path = output_path / "anchor_verification_report_v0_1.md"
    anchors: list[dict[str, Any]] = []
    _write_review_queue(queue, queue_path)
    _write_anchor_pool(anchors, pool_path)
    _write_coverage_csv(
        _build_coverage_v0_2(coverage_rows, source_results, anchors, config),
        coverage_v0_2_path,
    )
    _write_report(
        report_path,
        queue=queue,
        source_results=source_results,
        anchors=anchors,
        decisions=[],
        config=config,
        candidate_sha256=_compute_sha256(candidates_path),
        stage="pending_author_review",
    )
    counts = Counter(row["review_status"] for row in queue)
    return {
        "source_count": len(source_results),
        "candidate_count": len(candidates),
        "pending_author_review": counts.get("pending_author_review", 0),
        "rejected_dev50_overlap": counts.get("rejected_dev50_overlap", 0),
        "verified_anchor_count": 0,
        "human_checkpoint_required": True,
        "external_api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
        "output_files": {
            "review_queue": str(queue_path),
            "anchor_pool": str(pool_path),
            "coverage_v0_2": str(coverage_v0_2_path),
            "report": str(report_path),
        },
    }


def run_promote(
    queue_path: str | Path,
    candidates_path: str | Path,
    coverage_path: str | Path,
    dev50_registry_path: str | Path,
    formal_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config = load_review_config(config_path)
    queue = _read_review_queue(queue_path)
    candidates = _load_jsonl(candidates_path)
    coverage_rows = _load_jsonl(coverage_path)
    dev50_pairs = load_dev50_anchor_pairs(dev50_registry_path)
    source_results = _rebuild_source_results(
        coverage_rows,
        candidates,
        queue,
        config,
    )
    anchors, decisions = promote_verified_anchors(
        queue,
        coverage_rows,
        candidates,
        _build_page_counts(coverage_rows, formal_dir),
        dev50_pairs,
        config,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pool_path = output_path / "evidence_anchor_pool_v0_1.jsonl"
    coverage_v0_2_path = output_path / "source_coverage_matrix_v0_2.csv"
    report_path = output_path / "anchor_verification_report_v0_1.md"
    _write_anchor_pool(anchors, pool_path)
    _write_coverage_csv(
        _build_coverage_v0_2(coverage_rows, source_results, anchors, config),
        coverage_v0_2_path,
    )
    _write_report(
        report_path,
        queue=queue,
        source_results=source_results,
        anchors=anchors,
        decisions=decisions,
        config=config,
        candidate_sha256=_compute_sha256(candidates_path),
        stage="promotion_validation_complete",
    )
    decision_counts = Counter(row["promotion_status"] for row in decisions)
    return {
        "review_row_count": len(queue),
        "verified_anchor_count": len(anchors),
        "decision_counts": dict(sorted(decision_counts.items())),
        "external_api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
        "output_files": {
            "anchor_pool": str(pool_path),
            "coverage_v0_2": str(coverage_v0_2_path),
            "report": str(report_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成 Benchmark-v1 人工核验队列或升级已核验证据锚点"
    )
    parser.add_argument("--mode", choices=("prepare", "promote"), default="prepare")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("revision/benchmark/benchmark_v1/anchor_candidates_v0_1.jsonl"),
    )
    parser.add_argument(
        "--coverage",
        type=Path,
        default=Path("revision/benchmark/benchmark_v1/source_coverage_matrix_v0_1.jsonl"),
    )
    parser.add_argument(
        "--dev50-registry",
        type=Path,
        default=Path("revision/benchmark/dev50/evidence_anchor_registry.md"),
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
            "benchmark_anchor_review_v0_1.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("revision/benchmark/benchmark_v1"),
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("revision/benchmark/benchmark_v1/anchor_review_queue_v0_1.csv"),
    )
    args = parser.parse_args()
    if args.mode == "prepare":
        summary = run_prepare(
            args.candidates,
            args.coverage,
            args.dev50_registry,
            args.config,
            args.output_dir,
        )
    else:
        summary = run_promote(
            args.queue,
            args.candidates,
            args.coverage,
            args.dev50_registry,
            args.formal_dir,
            args.config,
            args.output_dir,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
