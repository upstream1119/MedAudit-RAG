from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
ANCHOR_REVIEW_PATH = MODULE_DIR / "benchmark_anchor_review.py"
QUEUE_FILENAME = "anchor_expansion_review_queue_v0_2.csv"
SUMMARY_FILENAME = "anchor_expansion_summary_v0_2.json"
GUIDE_FILENAME = "anchor_expansion_review_guide_v0_2.md"
WHITESPACE_RE = re.compile(r"\s+")
CITATION_MARKER_RE = re.compile(r"(?:\[|［)\s*\d+\s*(?:\]|］)")


def _load_anchor_review_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_anchor_review_for_expansion",
        ANCHOR_REVIEW_PATH,
    )
    if not spec or not spec.loader:
        raise RuntimeError("无法加载 benchmark_anchor_review.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


anchor_review = _load_anchor_review_module()


def _compute_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_text(value: Any) -> str:
    return WHITESPACE_RE.sub(" ", str(value or "")).strip()


def _source_page(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("source_id", "")), int(row.get("page_number", 0))


def collect_excluded_source_page_pairs(
    previous_review_rows: list[dict[str, Any]],
    existing_anchors: list[dict[str, Any]],
    dev50_pairs: dict[tuple[str, int], list[str]],
) -> set[tuple[str, int]]:
    pairs = {
        _source_page(row)
        for row in [*previous_review_rows, *existing_anchors]
        if row.get("source_id") and int(row.get("page_number", 0)) > 0
    }
    pairs.update(dev50_pairs)
    return pairs


def _contains_config_term(text: str, terms: list[str]) -> bool:
    casefolded = text.casefold()
    return any(str(term).casefold() in casefolded for term in terms)


def _candidate_is_relevant(
    candidate: dict[str, Any],
    source: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    text = _normalize_text(candidate.get("raw_text"))
    title = _normalize_text(source.get("title"))
    if anchor_review._is_shortlist_noise(candidate):
        return False
    reference_terms = list(config.get("reference_heading_terms") or [])
    if _contains_config_term(text, reference_terms):
        return False
    if len(CITATION_MARKER_RE.findall(text)) > int(config["max_citation_markers"]):
        return False
    if not config.get("require_pediatric_relevance", False):
        return True
    pediatric_terms = list(config.get("pediatric_terms") or [])
    return _contains_config_term(title, pediatric_terms) or _contains_config_term(
        text,
        pediatric_terms,
    )


def _topic_priority(candidate: dict[str, Any], config: dict[str, Any]) -> int:
    priority_topics = list(config.get("priority_topics") or [])
    matched_topics = set(candidate.get("matched_topics") or [])
    ranks = [
        index
        for index, topic in enumerate(priority_topics)
        if topic in matched_topics
    ]
    return min(ranks, default=len(priority_topics))


def _preselect_by_source(
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[str(candidate["source_id"])].append(candidate)

    limit = int(config["max_pending_per_source"])
    selected: list[dict[str, Any]] = []
    for source_id in sorted(grouped):
        ranked = sorted(
            grouped[source_id],
            key=lambda row: (
                _topic_priority(row, config),
                abs(int(row.get("granularity", 0)) - 512),
                abs(len(_normalize_text(row.get("raw_text"))) - 600),
                int(row["page_number"]),
                row["candidate_id"],
            ),
        )
        source_selected: list[dict[str, Any]] = []
        selected_pages: set[int] = set()
        selected_topics: set[str] = set()
        for candidate in ranked:
            page_number = int(candidate["page_number"])
            topics = set(candidate.get("matched_topics") or [])
            if page_number not in selected_pages and topics - selected_topics:
                source_selected.append(candidate)
                selected_pages.add(page_number)
                selected_topics.update(topics)
            if len(source_selected) >= limit:
                break
        for candidate in ranked:
            if len(source_selected) >= limit:
                break
            page_number = int(candidate["page_number"])
            if page_number not in selected_pages:
                source_selected.append(candidate)
                selected_pages.add(page_number)
        selected.extend(source_selected)
    return selected


def build_expansion_queue(
    *,
    candidates: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    previous_review_rows: list[dict[str, Any]],
    existing_anchors: list[dict[str, Any]],
    dev50_pairs: dict[tuple[str, int], list[str]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_map = {row["source_id"]: row for row in coverage_rows}
    if len(source_map) != len(coverage_rows):
        raise ValueError("coverage_rows 含重复 source_id")
    expected_source_count = int(config["expected_source_count"])
    if len(source_map) != expected_source_count:
        raise ValueError(
            f"来源数量不一致: expected={expected_source_count}, actual={len(source_map)}"
        )

    excluded_pairs = collect_excluded_source_page_pairs(
        previous_review_rows,
        existing_anchors,
        dev50_pairs,
    )
    fresh_candidates: list[dict[str, Any]] = []
    excluded_candidate_count = 0
    noise_filtered_count = 0
    for candidate in candidates:
        source_id = str(candidate.get("source_id", ""))
        if source_id not in source_map:
            raise ValueError(f"候选引用未知 source_id: {source_id}")
        if _source_page(candidate) in excluded_pairs:
            excluded_candidate_count += 1
            continue
        if not _candidate_is_relevant(candidate, source_map[source_id], config):
            noise_filtered_count += 1
            continue
        fresh_candidates.append(candidate)

    preselected_candidates = _preselect_by_source(fresh_candidates, config)
    review_config = dict(config)
    queue, source_results = anchor_review.build_review_queue(
        preselected_candidates,
        coverage_rows,
        {},
        review_config,
    )
    pending_rows = [
        row for row in queue if row["review_status"] == "pending_author_review"
    ]
    source_counts = Counter(row["source_id"] for row in pending_rows)
    target_review_queue_size = int(
        config.get("target_review_queue_size", len(pending_rows))
    )
    target_shortfall = max(0, target_review_queue_size - len(pending_rows))
    summary = {
        "config_version": config["config_version"],
        "dataset_version": config["dataset_version"],
        "kb_version": config["kb_version"],
        "candidate_count": len(candidates),
        "excluded_source_page_count": len(excluded_pairs),
        "excluded_candidate_count": excluded_candidate_count,
        "noise_filtered_count": noise_filtered_count,
        "eligible_candidate_count": len(fresh_candidates),
        "preselected_candidate_count": len(preselected_candidates),
        "pending_author_review": len(pending_rows),
        "target_review_queue_size": target_review_queue_size,
        "target_shortfall": target_shortfall,
        "target_met": target_shortfall == 0,
        "source_count_with_pending": len(source_counts),
        "pending_by_source": dict(sorted(source_counts.items())),
        "source_results": source_results,
        "human_checkpoint_required": True,
        "external_api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
    }
    return pending_rows, summary


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _queue_contains_author_review(path: Path) -> bool:
    if not path.exists():
        return False
    return any(row.get("author_reviewed_at") for row in _read_csv(path))


def _write_review_guide(
    path: Path,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Benchmark-v1 证据锚点扩展审核指南 v0.2",
        "",
        "## 当前状态",
        "",
        f"- 待作者逐页核验：{summary.get('pending_author_review', 0)} 条",
        f"- 覆盖来源：{summary.get('source_count_with_pending', 0)} 个",
        f"- 数据集版本：`{summary.get('dataset_version', '')}`",
        f"- 知识库版本：`{summary.get('kb_version', '')}`",
        "- 医学边界：本队列只是候选，不是 gold evidence。",
        "",
        "## 核验要求",
        "",
        "1. 打开对应 PDF，逐页确认页码与原文连续可读。",
        "2. 仅填写原文能够直接支持的 claim scope，不做临床外推。",
        "3. 参考文献、目录、标题残片或非儿科段落必须拒绝。",
        "4. 只有作者字段完整且 `scope_check=within_can_support` 的记录才能晋升。",
        "",
        "## 自动筛选审计",
        "",
        f"- 排除既有来源-页码单元：{summary.get('excluded_source_page_count', 0)}",
        f"- 排除对应候选块：{summary.get('excluded_candidate_count', 0)}",
        f"- 过滤噪声/非儿科候选：{summary.get('noise_filtered_count', 0)}",
        f"- 剩余可选候选：{summary.get('eligible_candidate_count', 0)}",
        "- 外部 API 调用：0",
        "- Token / 成本：0",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_expansion_outputs(
    *,
    queue: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    queue_path = output_path / QUEUE_FILENAME
    summary_path = output_path / SUMMARY_FILENAME
    guide_path = output_path / GUIDE_FILENAME
    if _queue_contains_author_review(queue_path):
        raise ValueError("现有扩展队列已含人工核验内容，拒绝覆盖")

    with queue_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=anchor_review.REVIEW_QUEUE_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in queue:
            writer.writerow(
                {
                    field: _csv_value(row.get(field, ""))
                    for field in anchor_review.REVIEW_QUEUE_FIELDS
                }
            )

    saved_summary = dict(summary)
    saved_summary["output_sha256"] = {
        "review_queue": _compute_sha256(queue_path),
    }
    summary_path.write_text(
        json.dumps(saved_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_review_guide(guide_path, saved_summary)
    return {
        "review_queue": str(queue_path),
        "summary": str(summary_path),
        "review_guide": str(guide_path),
    }


def run_prepare(
    *,
    candidates_path: str | Path,
    coverage_path: str | Path,
    previous_queue_path: str | Path,
    existing_anchor_pool_path: str | Path,
    dev50_registry_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config = _load_json(config_path)
    candidates = _load_jsonl(candidates_path)
    coverage_rows = _load_jsonl(coverage_path)
    previous_review_rows = _read_csv(previous_queue_path)
    existing_anchors = _load_jsonl(existing_anchor_pool_path)
    dev50_pairs = anchor_review.load_dev50_anchor_pairs(dev50_registry_path)
    queue, summary = build_expansion_queue(
        candidates=candidates,
        coverage_rows=coverage_rows,
        previous_review_rows=previous_review_rows,
        existing_anchors=existing_anchors,
        dev50_pairs=dev50_pairs,
        config=config,
    )
    summary["input_sha256"] = {
        "candidates": _compute_sha256(candidates_path),
        "coverage": _compute_sha256(coverage_path),
        "previous_review_queue": _compute_sha256(previous_queue_path),
        "existing_anchor_pool": _compute_sha256(existing_anchor_pool_path),
        "dev50_registry": _compute_sha256(dev50_registry_path),
        "config": _compute_sha256(config_path),
    }
    outputs = write_expansion_outputs(
        queue=queue,
        summary=summary,
        output_dir=output_dir,
    )
    return {**summary, "output_files": outputs}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成 Benchmark-v1 第二批独立证据锚点人工核验队列"
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("revision/benchmark/benchmark_v1/anchor_candidates_v0_1.jsonl"),
    )
    parser.add_argument(
        "--coverage",
        type=Path,
        default=Path(
            "revision/benchmark/benchmark_v1/source_coverage_matrix_v0_1.jsonl"
        ),
    )
    parser.add_argument(
        "--previous-queue",
        type=Path,
        default=Path("revision/benchmark/benchmark_v1/anchor_review_queue_v0_1.csv"),
    )
    parser.add_argument(
        "--existing-anchor-pool",
        type=Path,
        default=Path("revision/benchmark/benchmark_v1/evidence_anchor_pool_v0_1.jsonl"),
    )
    parser.add_argument(
        "--dev50-registry",
        type=Path,
        default=Path("revision/benchmark/dev50/evidence_anchor_registry.md"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "experiments/phase7_formal_experiments/configs/"
            "benchmark_anchor_expansion_v0_2.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("revision/benchmark/benchmark_v1"),
    )
    args = parser.parse_args()
    summary = run_prepare(
        candidates_path=args.candidates,
        coverage_path=args.coverage,
        previous_queue_path=args.previous_queue,
        existing_anchor_pool_path=args.existing_anchor_pool,
        dev50_registry_path=args.dev50_registry,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
