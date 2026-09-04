"""Audit sample-level changes introduced by frozen Validation40 G2 reranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


_EPSILON = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_hash(path: Path, expected: str, *, label: str) -> str:
    actual = sha256_file(path)
    if not expected or actual != expected.lower():
        raise ValueError(f"{label} SHA-256 mismatch")
    return actual


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


def validate_execution_guards(guards: Any) -> None:
    if not isinstance(guards, dict):
        raise ValueError("execution_guards must be an object")
    expected = {
        "validation40_gold_only": True,
        "pilot_test_content_access": False,
        "external_model_calls": False,
        "causal_claims": False,
    }
    for key, value in expected.items():
        if guards.get(key) is not value:
            raise ValueError(f"Unsafe execution guard: {key}")


def _candidate_key(candidate: dict[str, Any]) -> str:
    value = str(candidate.get("candidate_key", "")).strip()
    if not value:
        raise ValueError("candidate_key must be non-empty")
    return value


def _unique_keys(candidates: list[dict[str, Any]]) -> list[str]:
    keys = [_candidate_key(candidate) for candidate in candidates]
    if len(keys) != len(set(keys)):
        raise ValueError("candidate keys must be unique")
    return keys


def classify_final_change(
    g1_evidence: list[dict[str, Any]], g2_evidence: list[dict[str, Any]]
) -> dict[str, Any]:
    """Separate evidence membership changes from pure order changes."""
    g1_keys = _unique_keys(g1_evidence)
    g2_keys = _unique_keys(g2_evidence)
    g1_set = set(g1_keys)
    g2_set = set(g2_keys)
    added = [key for key in g2_keys if key not in g1_set]
    removed = [key for key in g1_keys if key not in g2_set]
    if g1_set != g2_set:
        change_type = "membership_changed"
    elif g1_keys != g2_keys:
        change_type = "order_only"
    else:
        change_type = "unchanged"
    return {
        "change_type": change_type,
        "g1_candidate_keys": g1_keys,
        "g2_candidate_keys": g2_keys,
        "added_candidate_keys": added,
        "removed_candidate_keys": removed,
    }


def classify_gold_effect(metric_row: dict[str, Any], *, level: str) -> dict[str, Any]:
    prefix = f"{level}_strict"
    g1_hit = bool(metric_row.get(f"g1_{prefix}_hit"))
    g2_hit = bool(metric_row.get(f"g2_{prefix}_hit"))
    g1_rank = metric_row.get(f"g1_{prefix}_rank")
    g2_rank = metric_row.get(f"g2_{prefix}_rank")
    if not g1_hit and g2_hit:
        effect = "strict_hit_added"
    elif g1_hit and not g2_hit:
        effect = "strict_hit_lost"
    elif g1_hit and g2_hit and int(g2_rank) < int(g1_rank):
        effect = "strict_rank_improved"
    elif g1_hit and g2_hit and int(g2_rank) > int(g1_rank):
        effect = "strict_rank_worsened"
    else:
        effect = "strict_metric_unchanged"
    g1_mrr = float(metric_row.get(f"g1_{prefix}_mrr", 0.0))
    g2_mrr = float(metric_row.get(f"g2_{prefix}_mrr", 0.0))
    return {
        "effect": effect,
        "g1_hit": g1_hit,
        "g2_hit": g2_hit,
        "g1_rank": g1_rank,
        "g2_rank": g2_rank,
        "g1_mrr": g1_mrr,
        "g2_mrr": g2_mrr,
        "mrr_delta": g2_mrr - g1_mrr,
    }


def _rank_map(candidates: list[dict[str, Any]]) -> dict[str, int]:
    return {_candidate_key(candidate): rank for rank, candidate in enumerate(candidates, 1)}


def _candidate_observation(
    candidate: dict[str, Any],
    *,
    g1_candidate_rank: int | None,
    g2_candidate_rank: int | None,
    g1_final_rank: int | None,
    g2_final_rank: int | None,
) -> dict[str, Any]:
    return {
        "candidate_key": _candidate_key(candidate),
        "source_file": candidate.get("source_file"),
        "page_number": candidate.get("page_number"),
        "collection": candidate.get("collection"),
        "candidate_origin": candidate.get("candidate_origin"),
        "g1_candidate_rank": g1_candidate_rank,
        "g2_candidate_rank": g2_candidate_rank,
        "g1_final_rank": g1_final_rank,
        "g2_final_rank": g2_final_rank,
        "g2_path_eligible": bool(candidate.get("g2_path_eligible", False)),
        "graph_path_score": float(candidate.get("graph_path_score", 0.0)),
        "graph_path_score_components": candidate.get(
            "graph_path_score_components", {}
        ),
        "g2_graph_path_trace": candidate.get("g2_graph_path_trace"),
    }


def audit_changed_sample(
    result_row: dict[str, Any],
    metric_row: dict[str, Any],
    *,
    g1_method: str,
    g2_method: str,
) -> dict[str, Any]:
    if str(result_row.get("sample_id")) != str(metric_row.get("sample_id")):
        raise ValueError("sample_id mismatch between result and metrics")
    if str(result_row.get("question")) != str(metric_row.get("question")):
        raise ValueError("question mismatch between result and metrics")
    methods = result_row.get("methods", {})
    g1_payload = methods.get(g1_method)
    g2_payload = methods.get(g2_method)
    if not isinstance(g1_payload, dict) or not isinstance(g2_payload, dict):
        raise ValueError("paired G1/G2 methods are missing")
    g1_candidates = g1_payload.get("candidates_top24")
    g2_candidates = g2_payload.get("candidates_top24")
    g1_final = g1_payload.get("evidence_top4")
    g2_final = g2_payload.get("evidence_top4")
    if not all(isinstance(value, list) for value in (g1_candidates, g2_candidates, g1_final, g2_final)):
        raise ValueError("paired candidate/evidence lists are missing")

    final_change = classify_final_change(g1_final, g2_final)
    g1_candidate_ranks = _rank_map(g1_candidates)
    g2_candidate_ranks = _rank_map(g2_candidates)
    g1_final_ranks = _rank_map(g1_final)
    g2_final_ranks = _rank_map(g2_final)
    g1_candidate_by_key = {_candidate_key(item): item for item in g1_candidates}
    g2_candidate_by_key = {_candidate_key(item): item for item in g2_candidates}

    def observation(key: str) -> dict[str, Any]:
        candidate = g2_candidate_by_key.get(key) or g1_candidate_by_key[key]
        return _candidate_observation(
            candidate,
            g1_candidate_rank=g1_candidate_ranks.get(key),
            g2_candidate_rank=g2_candidate_ranks.get(key),
            g1_final_rank=g1_final_ranks.get(key),
            g2_final_rank=g2_final_ranks.get(key),
        )

    candidate_effect = classify_gold_effect(metric_row, level="candidate")
    final_effect = classify_gold_effect(metric_row, level="final")
    graph_audit = result_row.get("graph_rerank_audit", {})
    return {
        "sample_id": result_row["sample_id"],
        "question": result_row["question"],
        "gold_source_filename": metric_row.get("gold_source_filename"),
        "gold_page_number": metric_row.get("gold_page_number"),
        "audit_inclusion_reasons": {
            "final_evidence_changed": final_change["change_type"] != "unchanged",
            "candidate_strict_mrr_changed": abs(candidate_effect["mrr_delta"])
            > _EPSILON,
        },
        "final_change": final_change,
        "candidate_gold_effect": candidate_effect,
        "final_gold_effect": final_effect,
        "added_evidence": [
            observation(key) for key in final_change["added_candidate_keys"]
        ],
        "removed_evidence": [
            observation(key) for key in final_change["removed_candidate_keys"]
        ],
        "graph_rerank_observation": {
            "candidate_order_changed": bool(
                graph_audit.get("candidate_order_changed", False)
            ),
            "evidence_order_changed": bool(
                graph_audit.get("evidence_order_changed", False)
            ),
            "path_eligible_count": int(graph_audit.get("path_eligible_count", 0)),
            "selected_path_count": int(graph_audit.get("selected_path_count", 0)),
            "max_rank_shift": float(graph_audit.get("max_rank_shift", 0.0)),
        },
        "interpretation_scope": "observational_structural_audit_only",
    }


def _direction(value: float) -> str:
    if value > _EPSILON:
        return "positive"
    if value < -_EPSILON:
        return "negative"
    return "zero"


def _effect_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter = Counter(row[key]["effect"] for row in rows)
    return {
        name: counter.get(name, 0)
        for name in (
            "strict_hit_added",
            "strict_hit_lost",
            "strict_rank_improved",
            "strict_rank_worsened",
            "strict_metric_unchanged",
        )
    }


def _markdown_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Validation40 G2 变化样本审计",
            "",
            "本报告是 Validation40 开发集上的观察性结构审计，不用于因果归因、统计显著性或临床有效性声明。",
            "Pilot Test80 仅校验 SHA-256，未读取内容；本步骤未调用外部模型，token 与费用均为 0。",
            "",
            "## 已观察证据",
            "",
            f"- Validation40 样本数：{summary['sample_count']}。",
            f"- 最终证据发生变化：{summary['final_evidence_changed_count']} 条，其中成员变化 {summary['membership_changed_count']} 条、仅顺序变化 {summary['order_only_count']} 条。",
            f"- 审计并集：{summary['audited_sample_count']} 条，定义为最终证据变化或候选严格页 MRR 变化。",
            f"- 候选严格页 MRR：改善 {summary['candidate_mrr_direction_counts']['positive']} 条、下降 {summary['candidate_mrr_direction_counts']['negative']} 条、不变 {summary['candidate_mrr_direction_counts']['zero']} 条。",
            f"- 最终严格页 MRR：改善 {summary['final_mrr_direction_counts']['positive']} 条、下降 {summary['final_mrr_direction_counts']['negative']} 条、不变 {summary['final_mrr_direction_counts']['zero']} 条。",
            f"- 最终严格页命中净变化：新增 {summary['final_gold_effect_counts']['strict_hit_added']} 条、丢失 {summary['final_gold_effect_counts']['strict_hit_lost']} 条。",
            "",
            "## 证据缺口",
            "",
            "- 本审计只能说明重排前后同时出现了哪些路径特征与排名变化，不能证明路径分数导致了 Gold 命中变化。",
            "- Validation40 是开发集；这些结果不能外推为独立测试集、临床安全性或统计显著性结论。",
            "",
            "## 后续检查",
            "",
            "- 用本报告定位正向、负向和无效变化样本，形成 G3 图一致性审计的最小规则候选。",
            "- G3 仍须与冻结 G2 做同数据、同候选预算、同最终证据预算的配对比较。",
            "- 在冻结前保留反例，不得只根据新增命中样本调整规则。",
            "",
        ]
    )


def run_change_audit(
    *,
    results_path: Path,
    results_manifest_path: Path,
    sample_metrics_path: Path,
    metrics_manifest_path: Path,
    pilot_test_path: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run a deterministic, Gold-only audit without parsing Pilot Test80."""
    validate_execution_guards(config.get("execution_guards"))
    results_sha = _verify_hash(
        results_path, str(config.get("expected_results_sha256", "")), label="results"
    )
    results_manifest_sha = _verify_hash(
        results_manifest_path,
        str(config.get("expected_results_manifest_sha256", "")),
        label="results manifest",
    )
    metrics_sha = _verify_hash(
        sample_metrics_path,
        str(config.get("expected_sample_metrics_sha256", "")),
        label="sample metrics",
    )
    metrics_manifest_sha = _verify_hash(
        metrics_manifest_path,
        str(config.get("expected_metrics_manifest_sha256", "")),
        label="metrics manifest",
    )
    pilot_sha = _verify_hash(
        pilot_test_path,
        str(config.get("expected_pilot_test_sha256", "")),
        label="Pilot Test80",
    )
    _require_empty_output_dir(output_dir)

    result_rows = _read_jsonl(results_path)
    metric_rows = _read_jsonl(sample_metrics_path)
    expected_count = int(config.get("expected_count", 0))
    if len(result_rows) != expected_count or len(metric_rows) != expected_count:
        raise ValueError("Validation40 row count mismatch")
    result_by_id = {str(row.get("sample_id")): row for row in result_rows}
    metric_by_id = {str(row.get("sample_id")): row for row in metric_rows}
    if len(result_by_id) != expected_count or len(metric_by_id) != expected_count:
        raise ValueError("Duplicate sample_id in paired inputs")
    if set(result_by_id) != set(metric_by_id):
        raise ValueError("sample_id set mismatch between results and metrics")

    g1_method = str(config.get("g1_method", "")).strip()
    g2_method = str(config.get("g2_method", "")).strip()
    if not g1_method or not g2_method or g1_method == g2_method:
        raise ValueError("G1/G2 method IDs must be distinct and non-empty")
    all_audits: list[dict[str, Any]] = []
    changed_audits: list[dict[str, Any]] = []
    for sample_id in sorted(result_by_id):
        audit = audit_changed_sample(
            result_by_id[sample_id],
            metric_by_id[sample_id],
            g1_method=g1_method,
            g2_method=g2_method,
        )
        all_audits.append(audit)
        reasons = audit["audit_inclusion_reasons"]
        if reasons["final_evidence_changed"] or reasons["candidate_strict_mrr_changed"]:
            changed_audits.append(audit)

    final_changed = [
        row for row in all_audits if row["final_change"]["change_type"] != "unchanged"
    ]
    expected_changed = int(config.get("expected_changed_evidence_count", -1))
    expected_audited = int(config.get("expected_audited_sample_count", -1))
    if expected_changed >= 0 and len(final_changed) != expected_changed:
        raise ValueError("Unexpected final evidence change count")
    if expected_audited >= 0 and len(changed_audits) != expected_audited:
        raise ValueError("Unexpected audited sample count")

    change_counts = Counter(row["final_change"]["change_type"] for row in final_changed)
    candidate_direction_counts = Counter(
        _direction(row["candidate_gold_effect"]["mrr_delta"]) for row in all_audits
    )
    final_direction_counts = Counter(
        _direction(row["final_gold_effect"]["mrr_delta"]) for row in all_audits
    )
    summary = {
        "summary_version": config.get("summary_version"),
        "phase": config.get("phase"),
        "sample_count": len(all_audits),
        "final_evidence_changed_count": len(final_changed),
        "audited_sample_count": len(changed_audits),
        "membership_changed_count": change_counts.get("membership_changed", 0),
        "order_only_count": change_counts.get("order_only", 0),
        "candidate_mrr_changed_count": len(all_audits)
        - candidate_direction_counts.get("zero", 0),
        "candidate_mrr_direction_counts": {
            key: candidate_direction_counts.get(key, 0)
            for key in ("positive", "negative", "zero")
        },
        "final_mrr_direction_counts": {
            key: final_direction_counts.get(key, 0)
            for key in ("positive", "negative", "zero")
        },
        "candidate_gold_effect_counts": _effect_counts(
            all_audits, "candidate_gold_effect"
        ),
        "final_gold_effect_counts": _effect_counts(all_audits, "final_gold_effect"),
        "strict_hit_added_count": sum(
            row["final_gold_effect"]["effect"] == "strict_hit_added"
            for row in all_audits
        ),
        "strict_hit_lost_count": sum(
            row["final_gold_effect"]["effect"] == "strict_hit_lost"
            for row in all_audits
        ),
        "candidate_strict_mrr_delta_mean": sum(
            row["candidate_gold_effect"]["mrr_delta"] for row in all_audits
        )
        / len(all_audits),
        "final_strict_mrr_delta_mean": sum(
            row["final_gold_effect"]["mrr_delta"] for row in all_audits
        )
        / len(all_audits),
        "audited_sample_ids": [row["sample_id"] for row in changed_audits],
        "strict_hit_added_sample_ids": [
            row["sample_id"]
            for row in all_audits
            if row["final_gold_effect"]["effect"] == "strict_hit_added"
        ],
        "strict_hit_lost_sample_ids": [
            row["sample_id"]
            for row in all_audits
            if row["final_gold_effect"]["effect"] == "strict_hit_lost"
        ],
        "interpretation_scope": "observational_structural_audit_only",
        "causal_claims_made": False,
        "statistical_significance_claimed": False,
        "clinical_significance_claimed": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / str(config["samples_filename"])
    summary_path = output_dir / str(config["summary_filename"])
    report_path = output_dir / str(config["report_filename"])
    audit_path = output_dir / str(config["audit_filename"])
    manifest_path = output_dir / str(config["manifest_filename"])
    _atomic_write(samples_path, _jsonl_bytes(changed_audits))
    _atomic_write(summary_path, _json_bytes(summary))
    _atomic_write(report_path, _markdown_report(summary).encode("utf-8"))
    execution_audit = {
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
        "causal_claims_made": False,
        "statistical_significance_claimed": False,
        "clinical_validation_claimed": False,
        "results_sha256": results_sha,
        "results_manifest_sha256": results_manifest_sha,
        "sample_metrics_sha256": metrics_sha,
        "metrics_manifest_sha256": metrics_manifest_sha,
        "pilot_test_sha256_verified_without_content_access": pilot_sha,
    }
    _atomic_write(audit_path, _json_bytes(execution_audit))
    manifest = {
        "manifest_version": config.get("manifest_version"),
        "ready": True,
        "files": {
            "samples": {"path": samples_path.name, "sha256": sha256_file(samples_path)},
            "summary": {"path": summary_path.name, "sha256": sha256_file(summary_path)},
            "report": {"path": report_path.name, "sha256": sha256_file(report_path)},
            "audit": {"path": audit_path.name, "sha256": sha256_file(audit_path)},
        },
    }
    _atomic_write(manifest_path, _json_bytes(manifest))
    return {
        "samples": changed_audits,
        "summary": summary,
        "audit": execution_audit,
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
    result = run_change_audit(
        results_path=root / config["results_path"],
        results_manifest_path=root / config["results_manifest_path"],
        sample_metrics_path=root / config["sample_metrics_path"],
        metrics_manifest_path=root / config["metrics_manifest_path"],
        pilot_test_path=root / config["pilot_test_path"],
        output_dir=args.output_dir or root / config["output_dir"],
        config=config,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
