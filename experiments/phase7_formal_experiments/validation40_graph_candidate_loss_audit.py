from __future__ import annotations

import argparse
import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from statistics import median
from typing import Any


def _is_strict_gold(candidate: dict[str, Any], gold: dict[str, Any]) -> bool:
    return (
        str(candidate.get("source_file", "")).casefold()
        == str(gold.get("source_filename", "")).casefold()
        and int(candidate.get("page_number") or 0) == int(gold.get("page_number") or 0)
    )


def classify_loss_mechanism(
    *,
    gold: dict[str, Any],
    f_candidates: list[dict[str, Any]],
    g1_candidates: list[dict[str, Any]],
    replaced_candidate_keys: list[str],
    candidate_budget: int,
) -> dict[str, Any]:
    """Classify a strict Gold loss using only frozen cached candidates."""
    if candidate_budget <= 0:
        raise ValueError("candidate_budget must be positive")
    replaced_keys = {str(key) for key in replaced_candidate_keys}
    g1_keys = {str(candidate.get("candidate_key", "")) for candidate in g1_candidates}
    removed_gold = [
        candidate
        for candidate in f_candidates
        if _is_strict_gold(candidate, gold)
        and str(candidate.get("candidate_key", "")) in replaced_keys
        and str(candidate.get("candidate_key", "")) not in g1_keys
    ]
    cached_ranks = [
        int(candidate.get("post_rerank_rank") or 0)
        for candidate in removed_gold
        if int(candidate.get("post_rerank_rank") or 0) > 0
    ]
    would_survive_rerank = any(rank <= candidate_budget for rank in cached_ranks)
    mechanism = (
        "pre_rerank_budget_replacement"
        if removed_gold and would_survive_rerank
        else "unresolved"
    )
    return {
        "primary_mechanism": mechanism,
        "removed_gold_candidate_keys": sorted(
            str(candidate.get("candidate_key", "")) for candidate in removed_gold
        ),
        "removed_gold_best_cached_rerank_rank": min(cached_ranks) if cached_ranks else None,
    }


def audit_shared_candidate_score_parity(
    f_candidates: list[dict[str, Any]],
    g1_candidates: list[dict[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Measure cached reranker parity for candidates shared by F and G1."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    f_by_key = {
        str(candidate.get("candidate_key", "")): candidate for candidate in f_candidates
    }
    differences: list[float] = []
    content_mismatch_count = 0
    for candidate in g1_candidates:
        key = str(candidate.get("candidate_key", ""))
        baseline = f_by_key.get(key)
        if baseline is None:
            continue
        if str(baseline.get("content", "")) != str(candidate.get("content", "")):
            content_mismatch_count += 1
        differences.append(
            abs(
                float(baseline.get("reranker_score") or 0.0)
                - float(candidate.get("reranker_score") or 0.0)
            )
        )
    return {
        "shared_candidate_count": len(differences),
        "nonzero_score_difference_count": sum(value > 0 for value in differences),
        "score_difference_above_tolerance_count": sum(
            value > tolerance for value in differences
        ),
        "content_mismatch_count": content_mismatch_count,
        "median_absolute_score_difference": median(differences) if differences else 0.0,
        "maximum_absolute_score_difference": max(differences, default=0.0),
        "tolerance": tolerance,
    }


def _normalize_text(text: object) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", str(text)).casefold()


def _character_ngrams(text: object, n: int) -> set[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return set()
    if len(normalized) <= n:
        return {normalized}
    return {normalized[index : index + n] for index in range(len(normalized) - n + 1)}


def _high_overlap(left: object, right: object, *, n: int, threshold: float) -> bool:
    left_ngrams = _character_ngrams(left, n)
    right_ngrams = _character_ngrams(right, n)
    if not left_ngrams or not right_ngrams:
        return False
    intersection = len(left_ngrams & right_ngrams)
    union = len(left_ngrams | right_ngrams)
    jaccard = intersection / union if union else 0.0
    overlap_coefficient = intersection / min(len(left_ngrams), len(right_ngrams))
    return max(jaccard, overlap_coefficient) >= threshold


def _select_deduplicated_evidence(
    candidates: list[dict[str, Any]],
    *,
    max_evidence: int,
    ngram_size: int,
    overlap_threshold: float,
) -> list[dict[str, Any]]:
    admitted: list[dict[str, Any]] = []
    normalized_seen: set[str] = set()
    for candidate in candidates:
        content = str(candidate.get("content", "")).strip()
        source = str(candidate.get("source_file", "")).strip()
        page = int(candidate.get("page_number") or 0)
        if not content or not source or page <= 0:
            continue
        normalized = _normalize_text(content)
        if normalized in normalized_seen:
            continue
        if any(
            source.casefold() == str(item.get("source_file", "")).casefold()
            and page == int(item.get("page_number") or 0)
            and _high_overlap(
                content,
                item.get("content", ""),
                n=ngram_size,
                threshold=overlap_threshold,
            )
            for item in admitted
        ):
            continue
        normalized_seen.add(normalized)
        admitted.append(candidate)
        if len(admitted) >= max_evidence:
            break
    return admitted


def simulate_late_union_from_cached_scores(
    *,
    f_candidates: list[dict[str, Any]],
    g1_candidates: list[dict[str, Any]],
    candidate_budget: int,
    final_evidence_k: int,
    dedup_ngram_size: int,
    dedup_overlap_threshold: float,
) -> dict[str, Any]:
    """Diagnose late pruning with cached scores; this is not a fair method result."""
    if candidate_budget <= 0 or final_evidence_k <= 0 or dedup_ngram_size <= 0:
        raise ValueError("candidate and evidence budgets must be positive")
    if not 0.0 <= dedup_overlap_threshold <= 1.0:
        raise ValueError("dedup_overlap_threshold must be within [0, 1]")
    f_keys = {str(candidate.get("candidate_key", "")) for candidate in f_candidates}
    union = [deepcopy(candidate) for candidate in f_candidates]
    union.extend(
        deepcopy(candidate)
        for candidate in g1_candidates
        if str(candidate.get("candidate_key", "")) not in f_keys
    )
    union.sort(
        key=lambda candidate: (
            -float(candidate.get("reranker_score") or 0.0),
            -float(candidate.get("rrf_score") or 0.0),
            str(candidate.get("candidate_key", "")),
        )
    )
    for rank, candidate in enumerate(union, start=1):
        candidate["post_rerank_rank"] = rank
    candidates = union[:candidate_budget]
    evidence = _select_deduplicated_evidence(
        candidates,
        max_evidence=final_evidence_k,
        ngram_size=dedup_ngram_size,
        overlap_threshold=dedup_overlap_threshold,
    )
    return {
        "reranker_input_count": len(union),
        "candidates": candidates,
        "evidence": evidence,
        "diagnostic_only": True,
        "matched_non_graph_compute_control_required": True,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_hash(path: Path, expected: object, label: str) -> str:
    observed = _sha256_file(path)
    if expected and observed != str(expected):
        raise ValueError(f"{label} SHA-256 mismatch")
    return observed


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(payload)
    return rows


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _require_empty_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {path}")


def _strict_hit(candidates: list[dict[str, Any]], gold: dict[str, Any]) -> bool:
    return any(_is_strict_gold(candidate, gold) for candidate in candidates)


def _pair_counts(pairs: list[tuple[bool, bool]]) -> dict[str, int]:
    counts = {"added": 0, "lost": 0, "both": 0, "neither": 0}
    for baseline_hit, comparison_hit in pairs:
        if baseline_hit and comparison_hit:
            counts["both"] += 1
        elif baseline_hit:
            counts["lost"] += 1
        elif comparison_hit:
            counts["added"] += 1
        else:
            counts["neither"] += 1
    return counts


def _candidate_snapshot(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_key": str(candidate.get("candidate_key", "")),
        "candidate_origin": str(candidate.get("candidate_origin", "")),
        "source_file": str(candidate.get("source_file", "")),
        "page_number": int(candidate.get("page_number") or 0),
        "pre_rerank_rank": candidate.get("pre_rerank_rank"),
        "post_rerank_rank": candidate.get("post_rerank_rank"),
        "reranker_score": float(candidate.get("reranker_score") or 0.0),
        "rrf_score": float(candidate.get("rrf_score") or 0.0),
    }


def _final_gain_attribution(
    *,
    gold: dict[str, Any],
    f_candidates: list[dict[str, Any]],
    g1_candidates: list[dict[str, Any]],
    g1_evidence: list[dict[str, Any]],
    added_candidate_keys: list[str],
) -> str | None:
    if not _strict_hit(g1_evidence, gold):
        return None
    added_keys = {str(key) for key in added_candidate_keys}
    if any(
        _is_strict_gold(candidate, gold)
        and str(candidate.get("candidate_key", "")) in added_keys
        for candidate in g1_evidence
    ):
        return "graph_added_gold_selected"
    if _strict_hit(f_candidates, gold) and _strict_hit(g1_candidates, gold):
        return "baseline_gold_promoted_after_replacement"
    return "other"


def _render_report(summary: dict[str, Any], samples: list[dict[str, Any]]) -> str:
    loss_rows = [row for row in samples if row["candidate_lost"] or row["final_lost"]]
    gain_rows = [row for row in samples if row["final_added"]]
    lines = [
        "# Validation40 G1-v0.2 新增丢失样本审计",
        "",
        "## 审计边界",
        "",
        "- 本报告只读取冻结 F/G1 缓存结果和 Gold-only 逐样本指标。",
        "- 未重新检索、未重新运行 Cross-Encoder、未读取 Pilot Test80。",
        "- late-union 使用缓存分数，仅用于诊断，不构成正式 G2 方法结果。",
        "",
        "## 汇总",
        "",
        f"- 候选阶段新增丢失：`{summary['candidate_loss_count']}` 条。",
        f"- 最终阶段新增丢失：`{summary['final_loss_count']}` 条。",
        f"- late-union 候选严格命中：`{summary['late_union_diagnostic']['candidate_strict_hits']}/{summary['sample_count']}`。",
        f"- late-union 最终严格命中：`{summary['late_union_diagnostic']['final_strict_hits']}/{summary['sample_count']}`。",
        f"- 共享候选最大缓存分数差：`{summary['shared_candidate_score_parity']['maximum_absolute_score_difference']:.10g}`。",
        "",
        "## 丢失样本",
        "",
    ]
    for row in loss_rows:
        lines.extend(
            [
                f"### {row['sample_id']}",
                "",
                f"- 问题：{row['question']}",
                f"- Gold：`{row['gold']['source_filename']}` 第 `{row['gold']['page_number']}` 页。",
                f"- 主机制：`{row['loss_mechanism']['primary_mechanism']}`。",
                f"- 被替换 Gold 候选：`{', '.join(row['loss_mechanism']['removed_gold_candidate_keys']) or '无'}`。",
                "",
            ]
        )
    lines.extend(["## 最终新增收益归因", ""])
    for row in gain_rows:
        lines.append(
            f"- `{row['sample_id']}`：`{row['final_gain_attribution'] or 'unresolved'}`。"
        )
    lines.extend(
        [
            "",
            "## 方法学结论",
            "",
            "1. 当前两个新增丢失均由重排前固定配额替换造成，而不是去冗余阶段新增损失。",
            "2. late-union 可以作为信息损失诊断，但其 Cross-Encoder 输入最多为 24 条，必须增加匹配的非图计算预算控制。",
            "3. 下一阶段应比较 F20、F24 控制、G1-v0.2 与版本化 G2；不得覆盖冻结 G1 产物。",
            "4. 本报告不支持统计显著性、临床安全、幻觉率下降或完整 Graph-enhanced 方法有效性结论。",
            "",
        ]
    )
    return "\n".join(lines)


def run_loss_audit(
    *,
    results_path: Path,
    sample_metrics_path: Path,
    paired_summary_path: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run a deterministic Gold-only audit over frozen cached F/G1 outputs."""
    _require_empty_output_dir(output_dir)
    observed_hashes = {
        "results_sha256": _verify_hash(
            results_path, config.get("expected_results_sha256"), "results"
        ),
        "sample_metrics_sha256": _verify_hash(
            sample_metrics_path,
            config.get("expected_sample_metrics_sha256"),
            "sample metrics",
        ),
        "paired_summary_sha256": _verify_hash(
            paired_summary_path,
            config.get("expected_paired_summary_sha256"),
            "paired summary",
        ),
    }
    result_rows = _read_jsonl(results_path)
    metric_rows = _read_jsonl(sample_metrics_path)
    paired_summary = _read_json(paired_summary_path)
    expected_count = int(config.get("expected_count", 40))
    if len(result_rows) != expected_count or len(metric_rows) != expected_count:
        raise ValueError("Validation40 row count mismatch")
    if paired_summary.get("sample_count") not in (None, expected_count):
        raise ValueError("paired summary sample count mismatch")
    metrics_by_id = {str(row.get("sample_id")): row for row in metric_rows}
    if len(metrics_by_id) != len(metric_rows):
        raise ValueError("Duplicate sample_id in sample metrics")

    f_method = str(config["f_method"])
    g1_method = str(config["g1_method"])
    candidate_budget = int(config.get("candidate_budget", 20))
    final_evidence_k = int(config.get("final_evidence_k", 4))
    score_tolerance = float(config.get("score_tolerance", 1e-5))
    sample_audits: list[dict[str, Any]] = []
    score_parity_totals = {
        "shared_candidate_count": 0,
        "nonzero_score_difference_count": 0,
        "score_difference_above_tolerance_count": 0,
        "content_mismatch_count": 0,
        "maximum_absolute_score_difference": 0.0,
        "tolerance": score_tolerance,
    }
    late_candidate_pairs: list[tuple[bool, bool]] = []
    late_final_pairs: list[tuple[bool, bool]] = []
    g1_late_candidate_changes: list[str] = []
    g1_late_final_changes: list[str] = []

    for result_row in result_rows:
        sample_id = str(result_row.get("sample_id", ""))
        metric = metrics_by_id.get(sample_id)
        if metric is None or str(metric.get("question")) != str(result_row.get("question")):
            raise ValueError("sample_id/question mismatch between results and metrics")
        methods = result_row.get("methods") or {}
        if f_method not in methods or g1_method not in methods:
            raise ValueError("Configured method output is missing")
        f_payload = methods[f_method]
        g1_payload = methods[g1_method]
        f_candidates = list(f_payload.get("candidates_top20") or [])
        g1_candidates = list(g1_payload.get("candidates_top20") or [])
        f_evidence = list(f_payload.get("evidence_top4") or [])
        g1_evidence = list(g1_payload.get("evidence_top4") or [])
        gold = {
            "source_filename": str(metric.get("gold_source_filename", "")),
            "page_number": int(metric.get("gold_page_number") or 0),
        }
        observed_flags = {
            "f_candidate_strict_hit": _strict_hit(f_candidates, gold),
            "g1_candidate_strict_hit": _strict_hit(g1_candidates, gold),
            "f_final_strict_hit": _strict_hit(f_evidence, gold),
            "g1_final_strict_hit": _strict_hit(g1_evidence, gold),
        }
        for key, observed in observed_flags.items():
            if bool(metric.get(key)) != observed:
                raise ValueError(f"Frozen sample metric mismatch for {sample_id}: {key}")
        expansion = result_row.get("graph_expansion_audit") or {}
        added_keys = [str(key) for key in expansion.get("added_candidate_keys") or []]
        replaced_keys = [
            str(key) for key in expansion.get("replaced_candidate_keys") or []
        ]
        mechanism = classify_loss_mechanism(
            gold=gold,
            f_candidates=f_candidates,
            g1_candidates=g1_candidates,
            replaced_candidate_keys=replaced_keys,
            candidate_budget=candidate_budget,
        )
        parity = audit_shared_candidate_score_parity(
            f_candidates, g1_candidates, tolerance=score_tolerance
        )
        for key in (
            "shared_candidate_count",
            "nonzero_score_difference_count",
            "score_difference_above_tolerance_count",
            "content_mismatch_count",
        ):
            score_parity_totals[key] += int(parity[key])
        score_parity_totals["maximum_absolute_score_difference"] = max(
            float(score_parity_totals["maximum_absolute_score_difference"]),
            float(parity["maximum_absolute_score_difference"]),
        )
        late = simulate_late_union_from_cached_scores(
            f_candidates=f_candidates,
            g1_candidates=g1_candidates,
            candidate_budget=candidate_budget,
            final_evidence_k=final_evidence_k,
            dedup_ngram_size=int(config.get("dedup_ngram_size", 3)),
            dedup_overlap_threshold=float(
                config.get("dedup_overlap_threshold", 0.75)
            ),
        )
        late_candidate_hit = _strict_hit(late["candidates"], gold)
        late_final_hit = _strict_hit(late["evidence"], gold)
        late_candidate_pairs.append(
            (observed_flags["f_candidate_strict_hit"], late_candidate_hit)
        )
        late_final_pairs.append((observed_flags["f_final_strict_hit"], late_final_hit))
        if late_candidate_hit != observed_flags["g1_candidate_strict_hit"]:
            g1_late_candidate_changes.append(sample_id)
        if late_final_hit != observed_flags["g1_final_strict_hit"]:
            g1_late_final_changes.append(sample_id)
        candidate_lost = (
            observed_flags["f_candidate_strict_hit"]
            and not observed_flags["g1_candidate_strict_hit"]
        )
        final_lost = (
            observed_flags["f_final_strict_hit"]
            and not observed_flags["g1_final_strict_hit"]
        )
        final_added = (
            not observed_flags["f_final_strict_hit"]
            and observed_flags["g1_final_strict_hit"]
        )
        f_by_key = {
            str(candidate.get("candidate_key", "")): candidate
            for candidate in f_candidates
        }
        g1_by_key = {
            str(candidate.get("candidate_key", "")): candidate
            for candidate in g1_candidates
        }
        sample_audits.append(
            {
                "sample_id": sample_id,
                "question": str(result_row.get("question", "")),
                "gold": gold,
                **observed_flags,
                "candidate_lost": candidate_lost,
                "final_lost": final_lost,
                "final_added": final_added,
                "loss_mechanism": mechanism,
                "removed_candidates": [
                    _candidate_snapshot(f_by_key[key])
                    for key in replaced_keys
                    if key in f_by_key
                ],
                "added_graph_candidates": [
                    _candidate_snapshot(g1_by_key[key])
                    for key in added_keys
                    if key in g1_by_key
                ],
                "shared_candidate_score_parity": parity,
                "late_union_diagnostic": {
                    "reranker_input_count": late["reranker_input_count"],
                    "candidate_strict_hit": late_candidate_hit,
                    "final_strict_hit": late_final_hit,
                },
                "final_gain_attribution": (
                    _final_gain_attribution(
                        gold=gold,
                        f_candidates=f_candidates,
                        g1_candidates=g1_candidates,
                        g1_evidence=g1_evidence,
                        added_candidate_keys=added_keys,
                    )
                    if final_added
                    else None
                ),
            }
        )

    sample_audits.sort(key=lambda row: row["sample_id"])
    candidate_loss_ids = sorted(
        row["sample_id"] for row in sample_audits if row["candidate_lost"]
    )
    final_loss_ids = sorted(row["sample_id"] for row in sample_audits if row["final_lost"])
    summary = {
        "summary_version": config.get(
            "summary_version", "phase7-c1c4e2e2c-loss-audit-v0.1"
        ),
        "phase": config.get("phase", "Phase 7-C1c-4e-2e-2c"),
        "sample_count": len(sample_audits),
        "candidate_loss_count": len(candidate_loss_ids),
        "candidate_loss_sample_ids": candidate_loss_ids,
        "final_loss_count": len(final_loss_ids),
        "final_loss_sample_ids": final_loss_ids,
        "candidate_and_final_loss_sets_identical": candidate_loss_ids == final_loss_ids,
        "shared_candidate_score_parity": score_parity_totals,
        "late_union_diagnostic": {
            "candidate_strict_hits": sum(pair[1] for pair in late_candidate_pairs),
            "final_strict_hits": sum(pair[1] for pair in late_final_pairs),
            "candidate_pair_counts_vs_f": _pair_counts(late_candidate_pairs),
            "final_pair_counts_vs_f": _pair_counts(late_final_pairs),
            "changed_candidate_sample_ids_vs_g1": sorted(g1_late_candidate_changes),
            "changed_final_sample_ids_vs_g1": sorted(g1_late_final_changes),
            "minimum_reranker_input_count": min(
                row["late_union_diagnostic"]["reranker_input_count"]
                for row in sample_audits
            ),
            "maximum_reranker_input_count": max(
                row["late_union_diagnostic"]["reranker_input_count"]
                for row in sample_audits
            ),
            "diagnostic_only": True,
            "matched_non_graph_compute_control_required": True,
        },
        "interpretation": {
            "g1_frozen_artifacts_modified": False,
            "formal_g2_result": False,
            "statistical_significance_claimed": False,
            "clinical_safety_claimed": False,
        },
    }
    audit = {
        "audit_version": config.get(
            "audit_version", "phase7-c1c4e2e2c-loss-audit-audit-v0.1"
        ),
        "config_version": config.get("config_version"),
        "sample_count": len(sample_audits),
        "input_hashes": observed_hashes,
        "pilot_test_accessed": False,
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0.0,
        "counterfactual_diagnostic_only": True,
        "matched_non_graph_compute_control_required": True,
        "clinical_validation_claimed": False,
    }
    report = _render_report(summary, sample_audits)
    _require_empty_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "samples": str(config.get("samples_filename", "samples.jsonl")),
        "summary": str(config.get("summary_filename", "summary.json")),
        "report": str(config.get("report_filename", "report.md")),
        "audit": str(config.get("audit_filename", "audit.json")),
    }
    contents = {
        "samples": _jsonl_bytes(sample_audits),
        "summary": _json_bytes(summary),
        "report": report.encode("utf-8"),
        "audit": _json_bytes(audit),
    }
    for key, filename in filenames.items():
        _atomic_write(output_dir / filename, contents[key])
    manifest = {
        "manifest_version": config.get(
            "manifest_version", "phase7-c1c4e2e2c-loss-audit-manifest-v0.1"
        ),
        "ready": True,
        "files": {
            key: {
                "path": filename,
                "sha256": _sha256_file(output_dir / filename),
            }
            for key, filename in filenames.items()
        },
        "pilot_test_accessed": False,
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0.0,
    }
    _atomic_write(
        output_dir / str(config.get("manifest_filename", "manifest.json")),
        _json_bytes(manifest),
    )
    return {
        "samples": sample_audits,
        "summary": summary,
        "audit": audit,
        "manifest": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = _read_json(args.config)
    root = args.repo_root.resolve()
    output_dir = args.output_dir or root / str(config["output_dir"])
    result = run_loss_audit(
        results_path=root / str(config["results_path"]),
        sample_metrics_path=root / str(config["sample_metrics_path"]),
        paired_summary_path=root / str(config["paired_summary_path"]),
        output_dir=output_dir,
        config=config,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
