"""Evaluate four local Validation40 retrieval configurations against frozen Gold."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import tempfile
from pathlib import Path
from typing import Any


METHODS = [
    "bge_m3_dense",
    "bge_m3_sparse",
    "dense_sparse_rrf",
    "hybrid_reranker_dedup",
]
SELECTION_RULE = [
    "final_strict_source_page_recall_at_4",
    "candidate_strict_source_page_mrr",
    "final_source_recall_at_4",
    "mean_redundant_pair_rate",
    "mean_latency_seconds",
]


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


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", delete=False, dir=path.parent, prefix=f".{path.name}."
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def normalize_source_name(source: object) -> str:
    return Path(str(source)).stem.replace(" ", "").casefold()


def _normalize_text(text: object) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", str(text)).casefold()


def _ngrams(text: object, n: int) -> set[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return set()
    if len(normalized) <= n:
        return {normalized}
    return {normalized[index : index + n] for index in range(len(normalized) - n + 1)}


def measure_redundancy(
    evidence: list[dict[str, Any]], *, ngram_size: int, threshold: float
) -> dict[str, Any]:
    if ngram_size <= 0 or not 0.0 <= threshold <= 1.0:
        raise ValueError("Invalid redundancy configuration")
    pair_count = 0
    redundant_pair_count = 0
    for left, right in itertools.combinations(evidence, 2):
        pair_count += 1
        left_ngrams = _ngrams(left.get("content", ""), ngram_size)
        right_ngrams = _ngrams(right.get("content", ""), ngram_size)
        if not left_ngrams or not right_ngrams:
            continue
        intersection = len(left_ngrams & right_ngrams)
        union = len(left_ngrams | right_ngrams)
        jaccard = intersection / union if union else 0.0
        overlap = intersection / min(len(left_ngrams), len(right_ngrams))
        redundant_pair_count += max(jaccard, overlap) >= threshold
    return {
        "pair_count": pair_count,
        "redundant_pair_count": redundant_pair_count,
        "redundant_pair_rate": (
            redundant_pair_count / pair_count if pair_count else 0.0
        ),
    }


def _ranks(
    evidence: list[dict[str, Any]],
    *,
    gold_source: str,
    gold_page: int,
    adjacent_page_tolerance: int,
) -> dict[str, list[int]]:
    source: list[int] = []
    strict: list[int] = []
    adjacent: list[int] = []
    for rank, item in enumerate(evidence, start=1):
        if normalize_source_name(item.get("source_file", "")) != gold_source:
            continue
        source.append(rank)
        page = int(item.get("page_number") or 0)
        if page == gold_page:
            strict.append(rank)
        if abs(page - gold_page) <= adjacent_page_tolerance:
            adjacent.append(rank)
    return {"source": source, "strict": strict, "adjacent": adjacent}


def evaluate_method_result(
    *,
    sample_id: str,
    method_id: str,
    gold: dict[str, Any],
    method_result: dict[str, Any],
    adjacent_page_tolerance: int,
    redundancy_ngram_size: int,
    redundancy_threshold: float,
) -> dict[str, Any]:
    candidates = list(method_result.get("candidates_top20") or [])
    evidence = list(method_result.get("evidence_top4") or [])
    gold_source = normalize_source_name(gold.get("source_filename", ""))
    gold_page = int(gold.get("page_number") or 0)
    candidate_ranks = _ranks(
        candidates,
        gold_source=gold_source,
        gold_page=gold_page,
        adjacent_page_tolerance=adjacent_page_tolerance,
    )
    final_ranks = _ranks(
        evidence,
        gold_source=gold_source,
        gold_page=gold_page,
        adjacent_page_tolerance=adjacent_page_tolerance,
    )
    strict_rank = candidate_ranks["strict"][0] if candidate_ranks["strict"] else None
    redundancy = measure_redundancy(
        evidence,
        ngram_size=redundancy_ngram_size,
        threshold=redundancy_threshold,
    )
    return {
        "sample_id": sample_id,
        "method_id": method_id,
        "gold_source_filename": gold.get("source_filename"),
        "gold_page_number": gold_page,
        "candidate_count": len(candidates),
        "final_evidence_count": len(evidence),
        "candidate_source_recall_at_20": int(bool(candidate_ranks["source"])),
        "candidate_strict_source_page_recall_at_20": int(bool(candidate_ranks["strict"])),
        "candidate_adjacent_source_page_recall_at_20": int(bool(candidate_ranks["adjacent"])),
        "candidate_strict_source_page_rank": strict_rank,
        "candidate_strict_source_page_mrr": 1.0 / strict_rank if strict_rank else 0.0,
        "final_source_recall_at_4": int(bool(final_ranks["source"])),
        "final_strict_source_page_recall_at_4": int(bool(final_ranks["strict"])),
        "final_adjacent_source_page_recall_at_4": int(bool(final_ranks["adjacent"])),
        "latency_seconds": float(method_result.get("latency_seconds") or 0.0),
        **redundancy,
    }


def join_gold_and_retrieval(
    gold_rows: list[dict[str, Any]], retrieval_rows: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    gold_by_id = {str(row.get("candidate_id")): row for row in gold_rows}
    retrieval_by_id = {str(row.get("sample_id")): row for row in retrieval_rows}
    if set(gold_by_id) != set(retrieval_by_id):
        raise ValueError("Gold and retrieval sample IDs do not match")
    return [(gold_by_id[sample_id], retrieval_by_id[sample_id]) for sample_id in sorted(gold_by_id)]


def effective_method_result(
    method_id: str, methods: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    result = dict(methods[method_id])
    if method_id != "hybrid_reranker_dedup":
        return result
    retrieval_latency = float(
        methods["dense_sparse_rrf"].get("latency_seconds") or 0.0
    )
    reranker_latency = float(result.get("latency_seconds") or 0.0)
    result["retrieval_latency_seconds"] = retrieval_latency
    result["reranker_latency_seconds"] = reranker_latency
    result["latency_seconds"] = retrieval_latency + reranker_latency
    return result


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows) if rows else 0.0


def summarize_method(method_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "method_id": method_id,
        "sample_count": len(rows),
        "candidate_source_recall_at_20": _mean(rows, "candidate_source_recall_at_20"),
        "candidate_strict_source_page_recall_at_20": _mean(
            rows, "candidate_strict_source_page_recall_at_20"
        ),
        "candidate_adjacent_source_page_recall_at_20": _mean(
            rows, "candidate_adjacent_source_page_recall_at_20"
        ),
        "candidate_strict_source_page_mrr": _mean(
            rows, "candidate_strict_source_page_mrr"
        ),
        "final_source_recall_at_4": _mean(rows, "final_source_recall_at_4"),
        "final_strict_source_page_recall_at_4": _mean(
            rows, "final_strict_source_page_recall_at_4"
        ),
        "final_adjacent_source_page_recall_at_4": _mean(
            rows, "final_adjacent_source_page_recall_at_4"
        ),
        "mean_candidate_count": _mean(rows, "candidate_count"),
        "mean_final_evidence_count": _mean(rows, "final_evidence_count"),
        "mean_redundant_pair_rate": _mean(rows, "redundant_pair_rate"),
        "mean_latency_seconds": _mean(rows, "latency_seconds"),
    }


def select_strong_non_graph_config(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    if not summaries:
        raise ValueError("No method summaries available")
    winner = max(
        summaries,
        key=lambda row: (
            float(row["final_strict_source_page_recall_at_4"]),
            float(row["candidate_strict_source_page_mrr"]),
            float(row["final_source_recall_at_4"]),
            -float(row["mean_redundant_pair_rate"]),
            -float(row["mean_latency_seconds"]),
        ),
    )
    return {**winner, "selection_rule": SELECTION_RULE}


def build_frozen_config(
    *,
    winner: dict[str, Any],
    retrieval_audit: dict[str, Any],
    gold_sha256: str,
    retrieval_sha256: str,
) -> dict[str, Any]:
    contract_fields = [
        "embedding_model_id",
        "reranker_model_id",
        "dense_index_status_sha256",
        "sparse_index_manifest_sha256",
        "candidate_k",
        "route_top_k",
        "rrf_k",
        "final_evidence_k",
        "dedup_ngram_size",
        "dedup_overlap_threshold",
    ]
    return {
        "freeze_version": "phase7-c1c4d-strong-non-graph-v0.1",
        "selection_split": "Validation40",
        "selected_method": winner["method_id"],
        "selection_rule": SELECTION_RULE,
        "selected_metrics": {
            key: value for key, value in winner.items() if key != "selection_rule"
        },
        "retrieval_contract": {
            field: retrieval_audit.get(field) for field in contract_fields
        },
        "input_hashes": {
            "validation40_gold_sha256": gold_sha256,
            "retrieval_results_sha256": retrieval_sha256,
            "pilot_test_sha256": retrieval_audit.get("pilot_test_sha256_before"),
        },
        "status": "validation_selected_pilot_test_unopened",
        "pilot_test_accessed": False,
        "clinically_validated": False,
        "graph_enhanced_effect_claimed": False,
    }


def _summary_markdown(summaries: list[dict[str, Any]], winner: dict[str, Any]) -> str:
    lines = [
        "# Validation40 强非图检索评测",
        "",
        "| 方法 | 来源@20 | 严格来源页@20 | MRR@20 | 来源@4 | 严格来源页@4 | ±1页@4 | 冗余率 | 平均延迟(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {method_id} | {candidate_source_recall_at_20:.4f} | "
            "{candidate_strict_source_page_recall_at_20:.4f} | "
            "{candidate_strict_source_page_mrr:.4f} | {final_source_recall_at_4:.4f} | "
            "{final_strict_source_page_recall_at_4:.4f} | "
            "{final_adjacent_source_page_recall_at_4:.4f} | "
            "{mean_redundant_pair_rate:.4f} | {mean_latency_seconds:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            f"按预声明规则选中的强非图配置：`{winner['method_id']}`。",
            "",
            "该选择仅基于 Validation40 的 guideline-grounded 检索代理指标，",
            "不构成临床有效性、Graph-enhanced 改善或独立专家验证结论。",
            "Pilot Test80 内容未用于本次选择。",
            "",
        ]
    )
    return "\n".join(lines)


def run_evaluation(
    *,
    gold_path: Path,
    retrieval_path: Path,
    retrieval_audit_path: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    expected_gold_sha = str(config["expected_gold_sha256"]).lower()
    expected_retrieval_sha = str(config["expected_retrieval_sha256"]).lower()
    if sha256_file(gold_path).lower() != expected_gold_sha:
        raise ValueError("Validation40 Gold SHA-256 mismatch")
    if sha256_file(retrieval_path).lower() != expected_retrieval_sha:
        raise ValueError("Validation40 retrieval SHA-256 mismatch")
    retrieval_audit = _read_json(retrieval_audit_path)
    if retrieval_audit.get("pilot_test_accessed") is not False:
        raise ValueError("Retrieval audit does not prove Pilot Test isolation")
    if retrieval_audit.get("pilot_test_sha256_before") != retrieval_audit.get(
        "pilot_test_sha256_after"
    ):
        raise ValueError("Pilot Test hash changed during retrieval")

    joined = join_gold_and_retrieval(_read_jsonl(gold_path), _read_jsonl(retrieval_path))
    sample_metrics: list[dict[str, Any]] = []
    for gold, retrieval in joined:
        methods = retrieval.get("methods") or {}
        if set(methods) != set(METHODS):
            raise ValueError(f"Method set mismatch for {retrieval.get('sample_id')}")
        for method_id in METHODS:
            sample_metrics.append(
                evaluate_method_result(
                    sample_id=str(retrieval["sample_id"]),
                    method_id=method_id,
                    gold=gold,
                    method_result=effective_method_result(method_id, methods),
                    adjacent_page_tolerance=int(config.get("adjacent_page_tolerance", 1)),
                    redundancy_ngram_size=int(config.get("redundancy_ngram_size", 3)),
                    redundancy_threshold=float(config.get("redundancy_threshold", 0.8)),
                )
            )
    summaries = [
        summarize_method(
            method_id,
            [row for row in sample_metrics if row["method_id"] == method_id],
        )
        for method_id in METHODS
    ]
    winner = select_strong_non_graph_config(summaries)
    frozen_config = build_frozen_config(
        winner=winner,
        retrieval_audit=retrieval_audit,
        gold_sha256=sha256_file(gold_path),
        retrieval_sha256=sha256_file(retrieval_path),
    )
    audit = {
        "evaluation_version": "validation40-hybrid-retrieval-evaluation-v0.1",
        "gold_sha256": sha256_file(gold_path),
        "retrieval_sha256": sha256_file(retrieval_path),
        "retrieval_audit_sha256": sha256_file(retrieval_audit_path),
        "sample_count": len(joined),
        "method_count": len(METHODS),
        "metric_row_count": len(sample_metrics),
        "selected_method": winner["method_id"],
        "pilot_test_sha256_before": retrieval_audit.get(
            "pilot_test_sha256_before"
        ),
        "pilot_test_sha256_after": retrieval_audit.get("pilot_test_sha256_after"),
        "retrieval_resource_profile": {
            "dense_index_bytes": retrieval_audit.get("dense_index_bytes"),
            "sparse_index_bytes": retrieval_audit.get("sparse_index_bytes"),
            "dense_query_encoding_seconds": retrieval_audit.get(
                "dense_query_encoding_seconds"
            ),
            "sparse_query_encoding_seconds": retrieval_audit.get(
                "sparse_query_encoding_seconds"
            ),
            "peak_vram_bytes": retrieval_audit.get("peak_vram_bytes"),
        },
        "pilot_test_accessed": False,
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0.0,
        "clinically_validated": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "validation40_hybrid_sample_metrics_v0_1.jsonl": _jsonl_bytes(sample_metrics),
        "validation40_hybrid_method_summary_v0_1.json": _json_bytes(summaries),
        "validation40_strong_non_graph_config_v0_1.json": _json_bytes(frozen_config),
        "validation40_hybrid_evaluation_audit_v0_1.json": _json_bytes(audit),
        "validation40_hybrid_evaluation_summary_v0_1.md": _summary_markdown(
            summaries, winner
        ).encode("utf-8"),
    }
    for name, content in outputs.items():
        _atomic_write(output_dir / name, content)
    return {
        "sample_metrics": sample_metrics,
        "summaries": summaries,
        "winner": winner,
        "audit": audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    config = _read_json(args.config)
    root = args.repo_root.resolve()
    result = run_evaluation(
        gold_path=root / config["gold_path"],
        retrieval_path=root / config["retrieval_path"],
        retrieval_audit_path=root / config["retrieval_audit_path"],
        output_dir=root / config["output_dir"],
        config=config,
    )
    print(json.dumps(result["summaries"], ensure_ascii=False, indent=2))
    print(json.dumps(result["winner"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
