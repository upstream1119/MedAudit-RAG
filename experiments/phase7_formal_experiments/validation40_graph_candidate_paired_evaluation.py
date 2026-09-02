"""Gold-only offline evaluation for the paired F_exact versus G1_exact run."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any


F_METHOD = "f_exact_hybrid_reranker_dedup"
G1_METHOD = "g1_exact_graph_expand_reranker_dedup"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


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


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", delete=False, dir=path.parent, prefix=f".{path.name}."
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def _require_empty_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")


def rank_metrics(
    candidates: list[dict[str, Any]], gold: dict[str, Any], *, cutoff: int
) -> dict[str, Any]:
    source = str(gold["source_filename"]).casefold()
    page = int(gold["page_number"])
    strict_rank = None
    source_rank = None
    adjacent_rank = None
    for rank, candidate in enumerate(candidates[:cutoff], start=1):
        candidate_source = str(candidate.get("source_file", "")).casefold()
        candidate_page = int(candidate.get("page_number") or 0)
        if source_rank is None and candidate_source == source:
            source_rank = rank
        if (
            adjacent_rank is None
            and candidate_source == source
            and abs(candidate_page - page) <= 1
        ):
            adjacent_rank = rank
        if strict_rank is None and candidate_source == source and candidate_page == page:
            strict_rank = rank
    return {
        "strict_hit": strict_rank is not None,
        "strict_rank": strict_rank,
        "strict_mrr": 1.0 / strict_rank if strict_rank else 0.0,
        "source_hit": source_rank is not None,
        "source_rank": source_rank,
        "adjacent_hit": adjacent_rank is not None,
        "adjacent_rank": adjacent_rank,
    }


def _pair_counts(rows: list[dict[str, Any]], f_field: str, g1_field: str) -> dict[str, int]:
    counts = {"added": 0, "lost": 0, "both": 0, "neither": 0}
    for row in rows:
        f_hit = bool(row[f_field])
        g1_hit = bool(row[g1_field])
        if not f_hit and g1_hit:
            counts["added"] += 1
        elif f_hit and not g1_hit:
            counts["lost"] += 1
        elif f_hit and g1_hit:
            counts["both"] += 1
        else:
            counts["neither"] += 1
    return counts


def summarize_paired_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "candidate_pair_counts": _pair_counts(
            rows, "f_candidate_strict_hit", "g1_candidate_strict_hit"
        ),
        "final_pair_counts": _pair_counts(
            rows, "f_final_strict_hit", "g1_final_strict_hit"
        ),
    }


def freeze_recommendation(
    *,
    f_candidate_strict_recall: float,
    g1_candidate_strict_recall: float,
    f_final_strict_recall: float,
    g1_final_strict_recall: float,
) -> dict[str, Any]:
    candidate_gain = g1_candidate_strict_recall > f_candidate_strict_recall
    final_non_degradation = g1_final_strict_recall >= f_final_strict_recall
    return {
        "decision": (
            "freeze_g1_candidate_expansion"
            if candidate_gain and final_non_degradation
            else "do_not_freeze_g1"
        ),
        "candidate_strict_recall_improved": candidate_gain,
        "final_strict_recall_non_degraded": final_non_degradation,
        "scope": "Validation40 development decision only",
        "clinical_significance_claimed": False,
    }


def _method_summary(
    rows: list[dict[str, Any]], *, prefix: str, candidate_cutoff: int = 20
) -> dict[str, float]:
    count = len(rows)
    if count == 0:
        raise ValueError("Cannot summarize zero rows")
    return {
        f"candidate_strict_recall_at_{candidate_cutoff}": sum(
            bool(row[f"{prefix}_candidate_strict_hit"]) for row in rows
        ) / count,
        f"candidate_source_recall_at_{candidate_cutoff}": sum(
            bool(row[f"{prefix}_candidate_source_hit"]) for row in rows
        ) / count,
        f"candidate_adjacent_recall_at_{candidate_cutoff}": sum(
            bool(row[f"{prefix}_candidate_adjacent_hit"]) for row in rows
        ) / count,
        "candidate_strict_mrr": sum(
            float(row[f"{prefix}_candidate_strict_mrr"]) for row in rows
        ) / count,
        "final_strict_recall_at_4": sum(
            bool(row[f"{prefix}_final_strict_hit"]) for row in rows
        ) / count,
        "final_source_recall_at_4": sum(
            bool(row[f"{prefix}_final_source_hit"]) for row in rows
        ) / count,
        "final_adjacent_recall_at_4": sum(
            bool(row[f"{prefix}_final_adjacent_hit"]) for row in rows
        ) / count,
        "final_strict_mrr": sum(
            float(row[f"{prefix}_final_strict_mrr"]) for row in rows
        ) / count,
    }


def _evaluate_sample(
    result: dict[str, Any],
    gold: dict[str, Any],
    *,
    f_method: str,
    g1_method: str,
    candidate_cutoff: int = 20,
    candidate_output_field: str = "candidates_top20",
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "sample_id": result["sample_id"],
        "question": result["question"],
        "gold_source_filename": gold["source_filename"],
        "gold_page_number": int(gold["page_number"]),
        "graph_expansion_audit": result.get("graph_expansion_audit", {}),
    }
    for method, prefix in ((f_method, "f"), (g1_method, "g1")):
        payload = result["methods"][method]
        candidates = payload[candidate_output_field]
        evidence = payload["evidence_top4"]
        candidate_metrics = rank_metrics(candidates, gold, cutoff=candidate_cutoff)
        final_metrics = rank_metrics(evidence, gold, cutoff=4)
        for key, value in candidate_metrics.items():
            output[f"{prefix}_candidate_{key}"] = value
        for key, value in final_metrics.items():
            output[f"{prefix}_final_{key}"] = value
    return output


def _markdown_report(
    summary: dict[str, Any],
    *,
    f_method: str,
    g1_method: str,
    candidate_cutoff: int = 20,
) -> str:
    f = summary["methods"][f_method]
    g1 = summary["methods"][g1_method]
    recommendation = summary["freeze_recommendation"]
    candidate_metric_key = f"candidate_strict_recall_at_{candidate_cutoff}"
    return "\n".join(
        [
            "# Validation40 F_exact vs G1_exact paired evaluation",
            "",
            "本报告仅用于 Validation40 开发集方法选择，不构成临床有效性结论。",
            "",
            f"| Method | Candidate strict recall@{candidate_cutoff} | Final strict recall@4 | Final source recall@4 |",
            "|---|---:|---:|---:|",
            f"| {f_method} | {f[candidate_metric_key]:.4f} | {f['final_strict_recall_at_4']:.4f} | {f['final_source_recall_at_4']:.4f} |",
            f"| {g1_method} | {g1[candidate_metric_key]:.4f} | {g1['final_strict_recall_at_4']:.4f} | {g1['final_source_recall_at_4']:.4f} |",
            "",
            f"冻结建议：`{recommendation['decision']}`。",
            "",
            "主要配对差异仅作开发诊断；本阶段不声称统计显著性或临床安全增益。",
            "",
        ]
    )


def evaluate_paired_retrieval(
    *,
    results_path: Path,
    retrieval_manifest_path: Path,
    gold_path: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate paired results after retrieval has completed without Gold access."""
    _require_empty_output_dir(output_dir)
    f_method = str(config.get("f_method", F_METHOD)).strip()
    g1_method = str(config.get("g1_method", G1_METHOD)).strip()
    if not f_method or not g1_method or f_method == g1_method:
        raise ValueError("paired method IDs must be distinct and non-empty")
    candidate_budget = int(config.get("candidate_budget", 20))
    if candidate_budget <= 0:
        raise ValueError("candidate_budget must be positive")
    candidate_output_field = str(
        config.get("candidate_output_field", f"candidates_top{candidate_budget}")
    ).strip()
    if not candidate_output_field:
        raise ValueError("candidate_output_field must be non-empty")
    candidate_metric_key = f"candidate_strict_recall_at_{candidate_budget}"
    expected_manifest_sha = str(
        config.get("expected_retrieval_manifest_sha256", "")
    ).lower()
    if (
        expected_manifest_sha
        and sha256_file(retrieval_manifest_path) != expected_manifest_sha
    ):
        raise ValueError("retrieval manifest SHA-256 mismatch")
    manifest = _read_json(retrieval_manifest_path)
    expected_results_sha = manifest.get("files", {}).get("results", {}).get("sha256")
    configured_results_sha = str(
        config.get("expected_retrieval_results_sha256", "")
    ).lower()
    if configured_results_sha and configured_results_sha != expected_results_sha:
        raise ValueError("configured retrieval results SHA-256 mismatch")
    if not expected_results_sha or sha256_file(results_path) != expected_results_sha:
        raise ValueError("retrieval results SHA-256 mismatch")
    expected_gold_sha = str(config.get("expected_gold_sha256", "")).lower()
    if not expected_gold_sha or sha256_file(gold_path) != expected_gold_sha:
        raise ValueError("Validation40 Gold SHA-256 mismatch")
    result_rows = _read_jsonl(results_path)
    gold_rows = _read_jsonl(gold_path)
    expected_count = int(config.get("expected_count", 40))
    if len(result_rows) != expected_count or len(gold_rows) != expected_count:
        raise ValueError("Validation40 row count mismatch")
    gold_by_id = {str(row.get("candidate_id")): row for row in gold_rows}
    if len(gold_by_id) != len(gold_rows):
        raise ValueError("Duplicate candidate_id in Validation40 Gold")

    sample_metrics: list[dict[str, Any]] = []
    for result in result_rows:
        sample_id = str(result.get("sample_id"))
        gold = gold_by_id.get(sample_id)
        if gold is None or str(gold.get("question")) != str(result.get("question")):
            raise ValueError("sample_id/question mismatch between results and Gold")
        for method in (f_method, g1_method):
            payload = result.get("methods", {}).get(method)
            if not isinstance(payload, dict):
                raise ValueError(f"Missing paired method: {method}")
            candidates = payload.get(candidate_output_field)
            if not isinstance(candidates, list):
                raise ValueError(f"Missing candidate field: {candidate_output_field}")
            if len(candidates) > candidate_budget:
                raise ValueError(f"Candidate budget exceeds {candidate_budget}")
            if len(payload.get("evidence_top4", [])) > 4:
                raise ValueError("Final evidence budget exceeds 4")
        sample_metrics.append(
            _evaluate_sample(
                result,
                gold,
                f_method=f_method,
                g1_method=g1_method,
                candidate_cutoff=candidate_budget,
                candidate_output_field=candidate_output_field,
            )
        )

    paired = summarize_paired_results(sample_metrics)
    method_summaries = {
        f_method: _method_summary(
            sample_metrics, prefix="f", candidate_cutoff=candidate_budget
        ),
        g1_method: _method_summary(
            sample_metrics, prefix="g1", candidate_cutoff=candidate_budget
        ),
    }
    recommendation = freeze_recommendation(
        f_candidate_strict_recall=method_summaries[f_method][candidate_metric_key],
        g1_candidate_strict_recall=method_summaries[g1_method][candidate_metric_key],
        f_final_strict_recall=method_summaries[f_method]["final_strict_recall_at_4"],
        g1_final_strict_recall=method_summaries[g1_method]["final_strict_recall_at_4"],
    )
    summary = {
        "summary_version": config.get(
            "summary_version", "phase7-c1c4e2b2-paired-evaluation-v0.1"
        ),
        "sample_count": len(sample_metrics),
        "candidate_budget": candidate_budget,
        "candidate_output_field": candidate_output_field,
        "methods": method_summaries,
        **paired,
        "zero_expansion_count": sum(
            bool(row.get("graph_expansion_audit", {}).get("zero_expansion"))
            for row in sample_metrics
        ),
        "graph_added_candidate_count": sum(
            len(row.get("graph_expansion_audit", {}).get("added_candidate_keys", []))
            for row in sample_metrics
        ),
        "freeze_recommendation": recommendation,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / str(config.get("sample_metrics_filename", "sample_metrics.jsonl"))
    summary_path = output_dir / str(config.get("summary_filename", "summary.json"))
    report_path = output_dir / str(config.get("report_filename", "report.md"))
    audit_path = output_dir / str(config.get("audit_filename", "audit.json"))
    manifest_path = output_dir / str(config.get("manifest_filename", "manifest.json"))
    _atomic_write(sample_path, _jsonl_bytes(sample_metrics))
    _atomic_write(summary_path, _json_bytes(summary))
    _atomic_write(
        report_path,
        _markdown_report(
            summary,
            f_method=f_method,
            g1_method=g1_method,
            candidate_cutoff=candidate_budget,
        ).encode("utf-8"),
    )
    audit = {
        "audit_version": config.get(
            "audit_version", "phase7-c1c4e2b2-paired-evaluation-audit-v0.1"
        ),
        "gold_accessed": True,
        "pilot_test_accessed": False,
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0.0,
        "clinical_validation_claimed": False,
        "candidate_budget": candidate_budget,
        "candidate_output_field": candidate_output_field,
        "retrieval_results_sha256": sha256_file(results_path),
        "gold_sha256": sha256_file(gold_path),
    }
    _atomic_write(audit_path, _json_bytes(audit))
    output_manifest = {
        "manifest_version": config.get(
            "manifest_version",
            "phase7-c1c4e2b2-paired-evaluation-manifest-v0.1",
        ),
        "ready": True,
        "files": {
            "sample_metrics": {"path": sample_path.name, "sha256": sha256_file(sample_path)},
            "summary": {"path": summary_path.name, "sha256": sha256_file(summary_path)},
            "report": {"path": report_path.name, "sha256": sha256_file(report_path)},
            "audit": {"path": audit_path.name, "sha256": sha256_file(audit_path)},
        },
    }
    _atomic_write(manifest_path, _json_bytes(output_manifest))
    return {
        "sample_metrics": sample_metrics,
        "summary": summary,
        "audit": audit,
        "manifest": output_manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--results-path", type=Path)
    parser.add_argument("--retrieval-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = _read_json(args.config)
    root = args.repo_root.resolve()
    result = evaluate_paired_retrieval(
        results_path=args.results_path or root / config["results_path"],
        retrieval_manifest_path=args.retrieval_manifest or root / config["retrieval_manifest_path"],
        gold_path=root / config["gold_path"],
        output_dir=args.output_dir or root / config["output_dir"],
        config=config,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
