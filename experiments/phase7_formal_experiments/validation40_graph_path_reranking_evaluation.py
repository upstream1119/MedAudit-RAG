"""Gold-only offline evaluation for frozen Validation40 G1 versus G2."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any


_CANDIDATE_IDENTITY_FIELDS = (
    "candidate_key",
    "collection",
    "document_id",
    "content",
    "source_file",
    "page_number",
    "chapter_title",
)

_PREDECLARED_FREEZE_RULE = {
    "final_strict_recall_at_4_must_improve": True,
    "final_source_recall_at_4_must_not_degrade": True,
    "equal_strict_mrr_gain_is_diagnostic_only": True,
}


def _candidate_identity(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(candidate.get(field) for field in _CANDIDATE_IDENTITY_FIELDS)


def assert_same_candidate_pool(
    g1_candidates: list[dict[str, Any]], g2_candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    """Fail closed unless G2 is a pure reordering of the frozen G1 pool."""
    if not isinstance(g1_candidates, list) or not isinstance(g2_candidates, list):
        raise TypeError("candidate pools must be lists")
    g1_identities = [_candidate_identity(candidate) for candidate in g1_candidates]
    g2_identities = [_candidate_identity(candidate) for candidate in g2_candidates]
    if len(set(g1_identities)) != len(g1_identities):
        raise ValueError("G1 candidate identities must be unique")
    if len(set(g2_identities)) != len(g2_identities):
        raise ValueError("G2 candidate identities must be unique")
    if set(g1_identities) != set(g2_identities):
        raise ValueError("G1/G2 candidate identity/content drift")
    return {
        "candidate_count": len(g1_identities),
        "candidate_identity_set_equal": True,
    }


def g2_freeze_recommendation(
    *,
    g1_final_strict_recall: float,
    g2_final_strict_recall: float,
    g1_final_source_recall: float,
    g2_final_source_recall: float,
    g1_final_strict_mrr: float,
    g2_final_strict_mrr: float,
) -> dict[str, Any]:
    """Apply the predeclared Validation40 development-only G2 decision rule."""
    strict_improved = g2_final_strict_recall > g1_final_strict_recall
    strict_equal = g2_final_strict_recall == g1_final_strict_recall
    source_non_degraded = g2_final_source_recall >= g1_final_source_recall
    mrr_improved = g2_final_strict_mrr > g1_final_strict_mrr
    if strict_improved and source_non_degraded:
        decision = "freeze_g2"
    elif strict_equal and source_non_degraded and mrr_improved:
        decision = "diagnostic_only_g2"
    else:
        decision = "do_not_freeze_g2"
    return {
        "decision": decision,
        "final_strict_recall_improved": strict_improved,
        "final_strict_recall_equal": strict_equal,
        "final_source_recall_non_degraded": source_non_degraded,
        "final_strict_mrr_improved": mrr_improved,
        "scope": "Validation40 development decision only",
        "statistical_significance_claimed": False,
        "clinical_significance_claimed": False,
    }


def validate_freeze_rule(rule: Any) -> None:
    if rule != _PREDECLARED_FREEZE_RULE:
        raise ValueError("Predeclared G2 freeze rule mismatch")


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


def _verify_hash(path: Path, expected: str, *, label: str) -> str:
    actual = sha256_file(path)
    if not expected or actual != expected.lower():
        raise ValueError(f"{label} SHA-256 mismatch")
    return actual


def _validate_execution_guards(config: dict[str, Any]) -> None:
    expected = {
        "validation40_gold_only": True,
        "pilot_test_content_access": False,
        "external_model_calls": False,
        "clinical_validation_claimed": False,
    }
    if config.get("execution_guards") != expected:
        raise ValueError("Gold-only execution guards mismatch")


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


def _pair_counts(
    rows: list[dict[str, Any]], g1_field: str, g2_field: str
) -> dict[str, int]:
    counts = {"added": 0, "lost": 0, "both": 0, "neither": 0}
    for row in rows:
        g1_hit = bool(row[g1_field])
        g2_hit = bool(row[g2_field])
        if not g1_hit and g2_hit:
            counts["added"] += 1
        elif g1_hit and not g2_hit:
            counts["lost"] += 1
        elif g1_hit and g2_hit:
            counts["both"] += 1
        else:
            counts["neither"] += 1
    return counts


def _method_summary(
    rows: list[dict[str, Any]], *, prefix: str, candidate_budget: int
) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot summarize zero rows")
    count = len(rows)
    return {
        f"candidate_strict_recall_at_{candidate_budget}": sum(
            bool(row[f"{prefix}_candidate_strict_hit"]) for row in rows
        )
        / count,
        f"candidate_source_recall_at_{candidate_budget}": sum(
            bool(row[f"{prefix}_candidate_source_hit"]) for row in rows
        )
        / count,
        f"candidate_adjacent_recall_at_{candidate_budget}": sum(
            bool(row[f"{prefix}_candidate_adjacent_hit"]) for row in rows
        )
        / count,
        "candidate_strict_mrr": sum(
            float(row[f"{prefix}_candidate_strict_mrr"]) for row in rows
        )
        / count,
        "final_strict_recall_at_4": sum(
            bool(row[f"{prefix}_final_strict_hit"]) for row in rows
        )
        / count,
        "final_source_recall_at_4": sum(
            bool(row[f"{prefix}_final_source_hit"]) for row in rows
        )
        / count,
        "final_adjacent_recall_at_4": sum(
            bool(row[f"{prefix}_final_adjacent_hit"]) for row in rows
        )
        / count,
        "final_strict_mrr": sum(
            float(row[f"{prefix}_final_strict_mrr"]) for row in rows
        )
        / count,
    }


def _evaluate_sample(
    result: dict[str, Any],
    gold: dict[str, Any],
    *,
    g1_method: str,
    g2_method: str,
    candidate_field: str,
    candidate_budget: int,
) -> dict[str, Any]:
    g1_candidates = result["methods"][g1_method][candidate_field]
    g2_candidates = result["methods"][g2_method][candidate_field]
    pool_audit = assert_same_candidate_pool(g1_candidates, g2_candidates)
    output: dict[str, Any] = {
        "sample_id": result["sample_id"],
        "question": result["question"],
        "gold_source_filename": gold["source_filename"],
        "gold_page_number": int(gold["page_number"]),
        "candidate_pool_audit": pool_audit,
        "graph_rerank_audit": result.get("graph_rerank_audit", {}),
    }
    for method, prefix in ((g1_method, "g1"), (g2_method, "g2")):
        payload = result["methods"][method]
        candidate_metrics = rank_metrics(
            payload[candidate_field], gold, cutoff=candidate_budget
        )
        final_metrics = rank_metrics(payload["evidence_top4"], gold, cutoff=4)
        for key, value in candidate_metrics.items():
            output[f"{prefix}_candidate_{key}"] = value
        for key, value in final_metrics.items():
            output[f"{prefix}_final_{key}"] = value
    for metric in ("strict_hit", "source_hit", "adjacent_hit"):
        if output[f"g1_candidate_{metric}"] != output[f"g2_candidate_{metric}"]:
            raise ValueError(f"Candidate recall identity violated: {metric}")
    return output


def _markdown_report(
    summary: dict[str, Any], *, g1_method: str, g2_method: str
) -> str:
    g1 = summary["methods"][g1_method]
    g2 = summary["methods"][g2_method]
    budget = summary["candidate_budget"]
    candidate_key = f"candidate_strict_recall_at_{budget}"
    return "\n".join(
        [
            "# Validation40 G1 vs G2 Gold-only paired evaluation",
            "",
            "本报告仅用于 Validation40 开发集方法选择，不构成统计显著性或临床有效性结论。",
            "Pilot Test80 内容未读取；本评测没有调用外部模型。",
            "",
            f"| Method | Candidate strict recall@{budget} | Final strict recall@4 | Final source recall@4 | Final strict MRR |",
            "|---|---:|---:|---:|---:|",
            f"| {g1_method} | {g1[candidate_key]:.4f} | {g1['final_strict_recall_at_4']:.4f} | {g1['final_source_recall_at_4']:.4f} | {g1['final_strict_mrr']:.4f} |",
            f"| {g2_method} | {g2[candidate_key]:.4f} | {g2['final_strict_recall_at_4']:.4f} | {g2['final_source_recall_at_4']:.4f} | {g2['final_strict_mrr']:.4f} |",
            "",
            f"逐题候选身份集合相同：{summary['candidate_identity_equal_count']}/{summary['sample_count']}。",
            f"最终严格页命中配对：`{json.dumps(summary['final_strict_pair_counts'], ensure_ascii=False, sort_keys=True)}`。",
            f"预声明冻结裁决：`{summary['freeze_recommendation']['decision']}`。",
            "",
        ]
    )


def evaluate_g1_g2(
    *,
    results_path: Path,
    results_manifest_path: Path,
    gold_path: Path,
    pilot_test_path: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate frozen G1/G2 outputs while keeping Pilot Test80 content sealed."""
    _require_empty_output_dir(output_dir)
    _validate_execution_guards(config)
    validate_freeze_rule(config.get("freeze_rule"))
    g1_method = str(config.get("g1_method", "")).strip()
    g2_method = str(config.get("g2_method", "")).strip()
    if not g1_method or not g2_method or g1_method == g2_method:
        raise ValueError("G1/G2 method IDs must be distinct and non-empty")
    candidate_budget = int(config.get("candidate_budget", 0))
    if candidate_budget <= 0:
        raise ValueError("candidate_budget must be positive")
    if int(config.get("final_evidence_k", 4)) != 4:
        raise ValueError("final_evidence_k must remain frozen at 4")
    candidate_field = str(config.get("candidate_output_field", "")).strip()
    if not candidate_field:
        raise ValueError("candidate_output_field must be non-empty")

    results_sha = _verify_hash(
        results_path,
        str(config.get("expected_results_sha256", "")),
        label="G1/G2 results",
    )
    results_manifest_sha = _verify_hash(
        results_manifest_path,
        str(config.get("expected_results_manifest_sha256", "")),
        label="G1/G2 results manifest",
    )
    gold_sha = _verify_hash(
        gold_path,
        str(config.get("expected_gold_sha256", "")),
        label="Validation40 Gold",
    )
    # Hash-only integrity check: Pilot Test80 is never decoded or parsed here.
    pilot_sha = _verify_hash(
        pilot_test_path,
        str(config.get("expected_pilot_test_sha256", "")),
        label="Pilot Test80",
    )
    retrieval_manifest = _read_json(results_manifest_path)
    bound_results_sha = (
        retrieval_manifest.get("files", {}).get("results", {}).get("sha256")
    )
    if retrieval_manifest.get("ready") is not True or bound_results_sha != results_sha:
        raise ValueError("G1/G2 results manifest does not bind the evaluated results")

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
        methods = result.get("methods")
        if not isinstance(methods, dict):
            raise ValueError("Result methods must be an object")
        for method in (g1_method, g2_method):
            payload = methods.get(method)
            if not isinstance(payload, dict):
                raise ValueError(f"Missing paired method: {method}")
            candidates = payload.get(candidate_field)
            evidence = payload.get("evidence_top4")
            if not isinstance(candidates, list):
                raise ValueError(f"Missing candidate field: {candidate_field}")
            if len(candidates) > candidate_budget:
                raise ValueError(f"Candidate budget exceeds {candidate_budget}")
            if not isinstance(evidence, list) or len(evidence) > 4:
                raise ValueError("Final evidence budget exceeds 4")
        sample_metrics.append(
            _evaluate_sample(
                result,
                gold,
                g1_method=g1_method,
                g2_method=g2_method,
                candidate_field=candidate_field,
                candidate_budget=candidate_budget,
            )
        )

    method_summaries = {
        g1_method: _method_summary(
            sample_metrics, prefix="g1", candidate_budget=candidate_budget
        ),
        g2_method: _method_summary(
            sample_metrics, prefix="g2", candidate_budget=candidate_budget
        ),
    }
    recommendation = g2_freeze_recommendation(
        g1_final_strict_recall=method_summaries[g1_method][
            "final_strict_recall_at_4"
        ],
        g2_final_strict_recall=method_summaries[g2_method][
            "final_strict_recall_at_4"
        ],
        g1_final_source_recall=method_summaries[g1_method][
            "final_source_recall_at_4"
        ],
        g2_final_source_recall=method_summaries[g2_method][
            "final_source_recall_at_4"
        ],
        g1_final_strict_mrr=method_summaries[g1_method]["final_strict_mrr"],
        g2_final_strict_mrr=method_summaries[g2_method]["final_strict_mrr"],
    )
    summary = {
        "summary_version": config.get("summary_version"),
        "phase": config.get("phase"),
        "sample_count": len(sample_metrics),
        "candidate_budget": candidate_budget,
        "candidate_output_field": candidate_field,
        "candidate_recall_identity_asserted": True,
        "candidate_identity_equal_count": sum(
            bool(row["candidate_pool_audit"]["candidate_identity_set_equal"])
            for row in sample_metrics
        ),
        "methods": method_summaries,
        "final_strict_pair_counts": _pair_counts(
            sample_metrics, "g1_final_strict_hit", "g2_final_strict_hit"
        ),
        "final_source_pair_counts": _pair_counts(
            sample_metrics, "g1_final_source_hit", "g2_final_source_hit"
        ),
        "final_adjacent_pair_counts": _pair_counts(
            sample_metrics, "g1_final_adjacent_hit", "g2_final_adjacent_hit"
        ),
        "freeze_recommendation": recommendation,
        "statistical_significance_claimed": False,
        "clinical_significance_claimed": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / str(config["sample_metrics_filename"])
    summary_path = output_dir / str(config["summary_filename"])
    report_path = output_dir / str(config["report_filename"])
    audit_path = output_dir / str(config["audit_filename"])
    manifest_path = output_dir / str(config["manifest_filename"])
    _atomic_write(sample_path, _jsonl_bytes(sample_metrics))
    _atomic_write(summary_path, _json_bytes(summary))
    _atomic_write(
        report_path,
        _markdown_report(summary, g1_method=g1_method, g2_method=g2_method).encode(
            "utf-8"
        ),
    )
    audit = {
        "audit_version": config.get("audit_version"),
        "phase": config.get("phase"),
        "config_version": config.get("config_version"),
        "dataset_version": config.get("dataset_version"),
        "kb_version": config.get("kb_version"),
        "gold_accessed": True,
        "pilot_test_accessed": False,
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0.0,
        "results_sha256": results_sha,
        "results_manifest_sha256": results_manifest_sha,
        "gold_sha256": gold_sha,
        "pilot_test_sha256_verified_without_content_access": pilot_sha,
        "candidate_identity_equal_count": summary[
            "candidate_identity_equal_count"
        ],
        "statistical_significance_claimed": False,
        "clinical_validation_claimed": False,
    }
    _atomic_write(audit_path, _json_bytes(audit))
    output_manifest = {
        "manifest_version": config.get("manifest_version"),
        "ready": True,
        "files": {
            "sample_metrics": {
                "path": sample_path.name,
                "sha256": sha256_file(sample_path),
            },
            "summary": {
                "path": summary_path.name,
                "sha256": sha256_file(summary_path),
            },
            "report": {
                "path": report_path.name,
                "sha256": sha256_file(report_path),
            },
            "audit": {
                "path": audit_path.name,
                "sha256": sha256_file(audit_path),
            },
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
    parser.add_argument("--results-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = _read_json(args.config)
    root = args.repo_root.resolve()
    result = evaluate_g1_g2(
        results_path=args.results_path or root / config["results_path"],
        results_manifest_path=(
            args.results_manifest or root / config["results_manifest_path"]
        ),
        gold_path=root / config["gold_path"],
        pilot_test_path=root / config["pilot_test_path"],
        output_dir=args.output_dir or root / config["output_dir"],
        config=config,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
