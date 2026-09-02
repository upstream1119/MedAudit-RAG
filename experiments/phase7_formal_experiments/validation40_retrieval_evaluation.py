"""Evaluate frozen Validation40 retrieval outputs against frozen Gold anchors."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_FILENAMES = {
    "sample_metrics": "validation40_retrieval_sample_metrics_v0_1.jsonl",
    "profile_summary": "validation40_retrieval_profile_summary_v0_1.json",
    "failure_cases": "validation40_retrieval_failure_cases_v0_1.jsonl",
    "audit": "validation40_retrieval_evaluation_audit_v0_1.json",
    "summary_markdown": "validation40_retrieval_evaluation_summary_v0_1.md",
}


def compute_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row must be an object at line {line_number}: {path}")
        rows.append(row)
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _canonical_jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def normalize_source_name(source: object) -> str:
    return Path(str(source)).stem.replace(" ", "").casefold()


def _normalize_text(text: object) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", str(text)).casefold()


def _character_ngrams(text: object, n: int) -> set[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return set()
    if len(normalized) <= n:
        return {normalized}
    return {normalized[index : index + n] for index in range(len(normalized) - n + 1)}


def _overlap_metrics(
    gold: dict[str, Any],
    evidence: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    ngram_size = int(config.get("text_ngram_size", 3))
    if ngram_size <= 0:
        raise ValueError("text_ngram_size must be positive")
    threshold = float(config.get("redundancy_jaccard_threshold", 0.8))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("redundancy_jaccard_threshold must be within [0, 1]")

    anchor_ngrams = _character_ngrams(gold.get("anchor_text_span", ""), ngram_size)
    evidence_ngrams = [
        _character_ngrams(item.get("content", ""), ngram_size) for item in evidence
    ]
    coverages = [
        len(anchor_ngrams & grams) / len(anchor_ngrams)
        if anchor_ngrams
        else 0.0
        for grams in evidence_ngrams
    ]

    pair_count = 0
    redundant_pair_count = 0
    for left, right in itertools.combinations(evidence_ngrams, 2):
        pair_count += 1
        union = left | right
        jaccard = len(left & right) / len(union) if union else 0.0
        redundant_pair_count += jaccard >= threshold

    return {
        "max_anchor_lexical_coverage": max(coverages, default=0.0),
        "evidence_pair_count": pair_count,
        "redundant_pair_count": redundant_pair_count,
        "redundant_pair_rate": (
            redundant_pair_count / pair_count if pair_count else 0.0
        ),
    }


def evaluate_physical_result(
    gold: dict[str, Any],
    retrieval: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return anchor-level retrieval proxies for one frozen physical result."""
    sample_id = str(retrieval.get("profile", {}).get("sample_id", ""))
    if sample_id != gold.get("candidate_id"):
        raise ValueError("Gold and retrieval sample_id mismatch")

    evidence = list(retrieval.get("evidence", []))
    gold_source = normalize_source_name(gold.get("source_filename", ""))
    gold_page = int(gold["page_number"])
    adjacent_tolerance = int(config.get("adjacent_page_tolerance", 1))
    if adjacent_tolerance < 0:
        raise ValueError("adjacent_page_tolerance must not be negative")
    source_ranks: list[int] = []
    source_page_ranks: list[int] = []
    adjacent_page_ranks: list[int] = []

    for rank, item in enumerate(evidence, start=1):
        if normalize_source_name(item.get("source_file", "")) != gold_source:
            continue
        source_ranks.append(rank)
        item_page = int(item.get("page_number", 0))
        if item_page == gold_page:
            source_page_ranks.append(rank)
        if abs(item_page - gold_page) <= adjacent_tolerance:
            adjacent_page_ranks.append(rank)

    evidence_count = len(evidence)
    overlap = _overlap_metrics(gold, evidence, config)
    source_page_rank = source_page_ranks[0] if source_page_ranks else None
    source_page_hit = bool(source_page_ranks)
    retrieval_status = retrieval.get("status")
    if retrieval_status == "failed":
        failure_type = "technical_failure"
    elif retrieval_status == "insufficient_evidence":
        failure_type = "insufficient_evidence"
    elif source_page_hit:
        failure_type = "gold_source_page_hit"
    elif adjacent_page_ranks:
        failure_type = "adjacent_gold_page_only"
    elif source_ranks:
        failure_type = "gold_page_miss"
    elif evidence:
        failure_type = "gold_source_miss"
    else:
        failure_type = "insufficient_evidence"

    return {
        "sample_id": sample_id,
        "physical_retrieval_key": retrieval.get("physical_retrieval_key"),
        "profile": retrieval.get("profile", {}).get("profile"),
        "top_k": retrieval.get("profile", {}).get("top_k"),
        "retrieval_status": retrieval_status,
        "retrieved_evidence_count": evidence_count,
        "gold_source_filename": gold.get("source_filename"),
        "gold_page_number": gold_page,
        "retrieved_source_pages": [
            {
                "rank": rank,
                "source_file": item.get("source_file"),
                "page_number": item.get("page_number"),
            }
            for rank, item in enumerate(evidence, start=1)
        ],
        "gold_source_hit": bool(source_ranks),
        "gold_source_page_hit": source_page_hit,
        "adjacent_gold_source_page_hit": bool(adjacent_page_ranks),
        "gold_source_page_rank": source_page_rank,
        "gold_source_page_reciprocal_rank": (
            1.0 / source_page_rank if source_page_rank else 0.0
        ),
        "gold_source_precision_at_k": (
            len(source_ranks) / evidence_count if evidence_count else 0.0
        ),
        "gold_source_page_precision_at_k": (
            len(source_page_ranks) / evidence_count if evidence_count else 0.0
        ),
        "failure_type": failure_type,
        **overlap,
    }


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows) if rows else 0.0


def _summarize_profile(profile: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    pair_count = sum(int(row["evidence_pair_count"]) for row in rows)
    redundant_pairs = sum(int(row["redundant_pair_count"]) for row in rows)
    statuses = Counter(str(row["retrieval_status"]) for row in rows)
    failures = Counter(str(row["failure_type"]) for row in rows)
    return {
        "profile": profile,
        "sample_count": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "failure_type_counts": dict(sorted(failures.items())),
        "gold_source_recall_at_k": _mean(rows, "gold_source_hit"),
        "gold_source_page_recall_at_k": _mean(rows, "gold_source_page_hit"),
        "adjacent_gold_source_page_recall_at_k": _mean(
            rows, "adjacent_gold_source_page_hit"
        ),
        "gold_source_page_mrr": _mean(
            rows, "gold_source_page_reciprocal_rank"
        ),
        "mean_gold_source_precision_at_k": _mean(
            rows, "gold_source_precision_at_k"
        ),
        "mean_gold_source_page_precision_at_k": _mean(
            rows, "gold_source_page_precision_at_k"
        ),
        "mean_max_anchor_lexical_coverage": _mean(
            rows, "max_anchor_lexical_coverage"
        ),
        "insufficient_evidence_rate": (
            statuses.get("insufficient_evidence", 0) / len(rows) if rows else 0.0
        ),
        "evidence_pair_count": pair_count,
        "redundant_pair_count": redundant_pairs,
        "redundant_pair_rate": (
            redundant_pairs / pair_count if pair_count else 0.0
        ),
    }


def _verify_hash(path: Path, expected: object, label: str) -> str:
    observed = compute_sha256(path)
    if observed != str(expected):
        raise ValueError(f"{label} SHA-256 mismatch")
    return observed


def _validate_config(config: dict[str, Any]) -> None:
    required = (
        "evaluation_version",
        "gold_path",
        "physical_results_path",
        "task_results_path",
        "retrieval_audit_path",
        "output_dir",
        "expected_gold_sha256",
        "expected_physical_results_sha256",
        "expected_task_results_sha256",
        "expected_retrieval_audit_sha256",
        "expected_sample_count",
        "expected_physical_result_count",
        "expected_logical_task_count",
        "expected_profiles",
        "expected_methods",
    )
    missing = [field for field in required if field not in config]
    if missing:
        raise ValueError(f"config fields missing: {', '.join(missing)}")
    if config.get("execute_retrieval") is not False:
        raise ValueError("execute_retrieval must remain false")
    if config.get("execute_model_calls") is not False:
        raise ValueError("execute_model_calls must remain false")
    if config.get("graph_reranking_executed") is not False:
        raise ValueError("graph_reranking_executed must remain false")


def _validate_task_results(
    task_rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, int]:
    if len(task_rows) != int(config["expected_logical_task_count"]):
        raise ValueError("logical retrieval task count mismatch")
    expected_methods = set(config["expected_methods"])
    by_sample: dict[str, set[str]] = {}
    counts: Counter[str] = Counter()
    for row in task_rows:
        if row.get("graph_reranking_executed") is not False:
            raise ValueError("task result unexpectedly executed graph reranking")
        sample_id = str(row.get("sample_id", ""))
        method_id = str(row.get("method_id", ""))
        if method_id not in expected_methods:
            raise ValueError(f"unexpected method_id: {method_id}")
        by_sample.setdefault(sample_id, set()).add(method_id)
        counts[method_id] += 1
    for sample_id, methods in by_sample.items():
        if methods != expected_methods:
            raise ValueError(f"logical method matrix mismatch: {sample_id}")
    return dict(sorted(counts.items()))


def _summary_markdown(
    profile_summary: list[dict[str, Any]], audit: dict[str, Any]
) -> str:
    lines = [
        "# Validation40 冻结检索质量评测摘要",
        "",
        f"- 评测版本：`{audit['evaluation_version']}`",
        f"- 样本数：{audit['sample_count']}",
        f"- 物理检索结果：{audit['physical_result_count']}",
        "- 外部模型调用：0",
        "- Pilot Test80 读取：false",
        "- Graph reranking executed: false",
        "",
        "## 按物理检索配置汇总",
        "",
    ]
    for row in profile_summary:
        lines.extend(
            [
                f"### {row['profile']}",
                "",
                f"- Gold 来源命中率@K：{row['gold_source_recall_at_k']:.4f}",
                f"- Gold 来源页精确命中率@K：{row['gold_source_page_recall_at_k']:.4f}",
                f"- Gold 来源页 ±1 页诊断命中率@K：{row['adjacent_gold_source_page_recall_at_k']:.4f}",
                f"- Gold 来源页 MRR：{row['gold_source_page_mrr']:.4f}",
                f"- 平均锚点词法覆盖：{row['mean_max_anchor_lexical_coverage']:.4f}",
                f"- 证据不足率：{row['insufficient_evidence_rate']:.4f}",
                f"- 词法冗余证据对比例：{row['redundant_pair_rate']:.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "精确来源页为主指标；相邻页和字符 n-gram 指标仅用于诊断。",
            "每题只有一个冻结 Gold 主锚点，因此这些结果不是完整临床证据精确率，",
            "也不构成四种完整方法、独立专家验证或临床有效性结论。",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_replace(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", delete=False, dir=path.parent, prefix=f".{path.name}."
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def _write_immutable_outputs(
    result: dict[str, Any], output_dir: Path
) -> dict[str, str]:
    sample_metrics = sorted(
        result["sample_metrics"],
        key=lambda row: (str(row["sample_id"]), str(row["profile"])),
    )
    failures = [
        row
        for row in sample_metrics
        if row["failure_type"] != "gold_source_page_hit"
    ]
    payloads = {
        OUTPUT_FILENAMES["sample_metrics"]: _canonical_jsonl_bytes(sample_metrics),
        OUTPUT_FILENAMES["profile_summary"]: _canonical_json_bytes(
            result["profile_summary"]
        ),
        OUTPUT_FILENAMES["failure_cases"]: _canonical_jsonl_bytes(failures),
        OUTPUT_FILENAMES["audit"]: _canonical_json_bytes(result["audit"]),
        OUTPUT_FILENAMES["summary_markdown"]: _summary_markdown(
            result["profile_summary"], result["audit"]
        ).encode("utf-8"),
    }
    for filename, content in payloads.items():
        target = output_dir / filename
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"immutable output conflict: refusing to overwrite {target}")
    for filename, content in payloads.items():
        target = output_dir / filename
        if not target.exists():
            _atomic_replace(target, content)
    return {filename: _sha256_bytes(content) for filename, content in payloads.items()}


def run(
    config_path: str | Path,
    *,
    repo_root: str | Path = ROOT,
) -> dict[str, Any]:
    root = Path(repo_root)
    config = _load_json(Path(config_path))
    _validate_config(config)

    gold_path = root / str(config["gold_path"])
    physical_path = root / str(config["physical_results_path"])
    task_path = root / str(config["task_results_path"])
    retrieval_audit_path = root / str(config["retrieval_audit_path"])
    input_hashes = {
        "gold_sha256": _verify_hash(
            gold_path, config["expected_gold_sha256"], "Validation40 Gold"
        ),
        "physical_results_sha256": _verify_hash(
            physical_path,
            config["expected_physical_results_sha256"],
            "physical retrieval results",
        ),
        "task_results_sha256": _verify_hash(
            task_path, config["expected_task_results_sha256"], "task retrieval results"
        ),
        "retrieval_audit_sha256": _verify_hash(
            retrieval_audit_path,
            config["expected_retrieval_audit_sha256"],
            "retrieval audit",
        ),
    }

    retrieval_audit = _load_json(retrieval_audit_path)
    if retrieval_audit.get("external_model_calls") != 0:
        raise ValueError("retrieval audit contains external model calls")
    if retrieval_audit.get("graph_reranking_executed") is not False:
        raise ValueError("retrieval audit unexpectedly executed graph reranking")
    if retrieval_audit.get("pilot_test_accessed") is not False:
        raise ValueError("retrieval audit accessed Pilot Test80")

    gold_rows = _read_jsonl(gold_path)
    physical_rows = _read_jsonl(physical_path)
    task_rows = _read_jsonl(task_path)
    method_counts = _validate_task_results(task_rows, config)
    result = build_retrieval_evaluation(gold_rows, physical_rows, config)
    result["audit"].update(
        {
            **input_hashes,
            "evaluation_version": config["evaluation_version"],
            "dataset_version": config.get("expected_dataset_version"),
            "kb_version": config.get("expected_kb_version"),
            "gold_join_stage": "post_retrieval_evaluation_only",
            "pilot_test_read": False,
            "logical_method_task_counts": method_counts,
            "logical_task_count": len(task_rows),
            "main_metric_scope": "single_frozen_gold_anchor_proxy",
            "adjacent_page_metric_is_diagnostic_only": True,
            "lexical_overlap_is_semantic_metric": False,
        }
    )

    output_dir = root / str(config["output_dir"])
    output_hashes = _write_immutable_outputs(result, output_dir)
    return {
        "audit": result["audit"],
        "profile_summary": result["profile_summary"],
        "output_dir": str(output_dir),
        "output_sha256": output_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen Validation40 retrieval outputs without model calls."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root", default=str(ROOT))
    args = parser.parse_args()
    result = run(args.config, repo_root=args.repo_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def build_retrieval_evaluation(
    gold_rows: list[dict[str, Any]],
    physical_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Join frozen Gold after retrieval and aggregate physical-profile metrics."""
    expected_sample_count = int(config["expected_sample_count"])
    expected_physical_count = int(config["expected_physical_result_count"])
    expected_profiles = list(config["expected_profiles"])
    if len(gold_rows) != expected_sample_count:
        raise ValueError("Validation40 Gold sample count mismatch")
    if len(physical_rows) != expected_physical_count:
        raise ValueError("physical retrieval result count mismatch")

    gold_by_id = {str(row["candidate_id"]): row for row in gold_rows}
    if len(gold_by_id) != len(gold_rows):
        raise ValueError("duplicate Gold candidate_id")

    sample_metrics: list[dict[str, Any]] = []
    observed_matrix: dict[str, set[str]] = {}
    for physical in physical_rows:
        profile = physical.get("profile", {})
        sample_id = str(profile.get("sample_id", ""))
        profile_name = str(profile.get("profile", ""))
        if sample_id not in gold_by_id:
            raise ValueError(f"retrieval sample missing from Gold: {sample_id}")
        if profile_name not in expected_profiles:
            raise ValueError(f"unexpected retrieval profile: {profile_name}")
        observed_matrix.setdefault(sample_id, set()).add(profile_name)
        sample_metrics.append(
            evaluate_physical_result(gold_by_id[sample_id], physical, config)
        )

    expected_profile_set = set(expected_profiles)
    for sample_id, profiles in observed_matrix.items():
        if profiles != expected_profile_set:
            raise ValueError(f"retrieval profile matrix mismatch: {sample_id}")
    if set(observed_matrix) != set(gold_by_id):
        raise ValueError("retrieval samples do not cover all Gold samples")

    profile_summary = [
        _summarize_profile(
            profile,
            [row for row in sample_metrics if row["profile"] == profile],
        )
        for profile in expected_profiles
    ]
    return {
        "sample_metrics": sample_metrics,
        "profile_summary": profile_summary,
        "audit": {
            "sample_count": len(gold_rows),
            "physical_result_count": len(physical_rows),
            "profile_count": len(expected_profiles),
            "method_comparison_generated": False,
            "graph_reranking_executed": False,
            "retrieval_executed": False,
            "external_model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost": 0,
            "clinically_validated": False,
        },
    }


if __name__ == "__main__":
    main()
