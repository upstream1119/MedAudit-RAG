"""Run the exact-control F versus G1 graph candidate expansion experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from experiments.phase6_evidence_graph.graph_contract import (
    assert_no_gold_only_content,
)
from experiments.phase7_formal_experiments import graph_candidate_expansion as graph
from experiments.phase7_formal_experiments import runtime_graph_path_router as path_router


F_METHOD = "f_exact_hybrid_reranker_dedup"
G1_METHOD = "g1_exact_graph_expand_reranker_dedup"
RUNTIME_FIELDS = {"sample_id", "question", "dataset_version", "kb_version"}


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


def _verify_hash(path: Path, expected: object, label: str) -> str:
    actual = sha256_file(path)
    if not expected or actual != str(expected).strip().lower():
        raise ValueError(f"{label} SHA-256 mismatch")
    return actual


def validate_frozen_inputs(
    *,
    config: dict[str, Any],
    runtime_projection_path: Path,
    exact_results_path: Path,
    exact_audit_path: Path,
    graph_manifest_path: Path,
    pilot_test_path: Path,
    runtime_lexicon_path: Path | None = None,
    routing_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Hash frozen inputs; Pilot Test80 is never decoded or parsed."""
    observed = {
        "runtime_projection_sha256": _verify_hash(
            runtime_projection_path,
            config.get("expected_runtime_projection_sha256"),
            "runtime projection",
        ),
        "exact_results_sha256": _verify_hash(
            exact_results_path,
            config.get("expected_exact_results_sha256"),
            "exact results",
        ),
        "exact_audit_sha256": _verify_hash(
            exact_audit_path,
            config.get("expected_exact_audit_sha256"),
            "exact audit",
        ),
        "graph_manifest_sha256": _verify_hash(
            graph_manifest_path,
            config.get("expected_graph_manifest_sha256"),
            "graph manifest",
        ),
        "pilot_test_sha256": _verify_hash(
            pilot_test_path,
            config.get("expected_pilot_test_sha256"),
            "pilot test",
        ),
    }
    routing_inputs = (runtime_lexicon_path, routing_manifest_path)
    if any(path is not None for path in routing_inputs) and not all(
        path is not None for path in routing_inputs
    ):
        raise ValueError("runtime routing input paths must be provided together")
    if runtime_lexicon_path is not None and routing_manifest_path is not None:
        observed["runtime_lexicon_sha256"] = _verify_hash(
            runtime_lexicon_path,
            config.get("expected_runtime_lexicon_sha256"),
            "runtime lexicon",
        )
        observed["routing_manifest_sha256"] = _verify_hash(
            routing_manifest_path,
            config.get("expected_routing_manifest_sha256"),
            "routing manifest",
        )
    return {**observed, "pilot_test_accessed": False, "gold_accessed": False}


def _method_output(
    *,
    question: str,
    candidates: list[dict[str, Any]],
    scorer: Any,
    final_evidence_k: int,
    reranker_batch_size: int,
    dedup_ngram_size: int,
    dedup_overlap_threshold: float,
    candidate_output_field: str,
) -> dict[str, Any]:
    reranked = _rerank_candidates(
        question=question,
        candidates=candidates,
        scorer=scorer,
        batch_size=reranker_batch_size,
    )
    evidence, dedup_audit = _deduplicate_evidence(
        reranked,
        max_evidence=final_evidence_k,
        ngram_size=dedup_ngram_size,
        overlap_threshold=dedup_overlap_threshold,
    )
    return {
        candidate_output_field: reranked,
        "evidence_top4": evidence,
        "dedup_audit": dedup_audit,
    }


def _rerank_candidates(
    *, question: str, candidates: list[dict[str, Any]], scorer: Any, batch_size: int
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    pairs = [[question, str(item.get("content", ""))] for item in candidates]
    scores = list(
        scorer.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    )
    if len(scores) != len(candidates):
        raise RuntimeError("Reranker output count does not match candidate count")
    enriched = [
        {
            **item,
            "pre_rerank_rank": rank,
            "reranker_score": float(score),
        }
        for rank, (item, score) in enumerate(zip(candidates, scores), start=1)
    ]
    enriched.sort(
        key=lambda item: (
            -float(item["reranker_score"]),
            -float(item.get("rrf_score", 0.0)),
            str(item["candidate_key"]),
        )
    )
    for rank, item in enumerate(enriched, start=1):
        item["post_rerank_rank"] = rank
    return enriched


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


def _deduplicate_evidence(
    candidates: list[dict[str, Any]],
    *,
    max_evidence: int,
    ngram_size: int,
    overlap_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if max_evidence <= 0 or ngram_size <= 0:
        raise ValueError("max_evidence and ngram_size must be positive")
    if not 0.0 <= overlap_threshold <= 1.0:
        raise ValueError("overlap_threshold must be within [0, 1]")
    admitted: list[dict[str, Any]] = []
    normalized_seen: set[str] = set()
    exact_duplicate_count = 0
    same_page_overlap_count = 0
    invalid_provenance_count = 0
    for candidate in candidates:
        content = str(candidate.get("content", "")).strip()
        source = str(candidate.get("source_file", "")).strip()
        page = int(candidate.get("page_number") or 0)
        if not content or not source or page <= 0:
            invalid_provenance_count += 1
            continue
        normalized = _normalize_text(content)
        if normalized in normalized_seen:
            exact_duplicate_count += 1
            continue
        overlaps = any(
            source.casefold() == str(item.get("source_file", "")).casefold()
            and page == int(item.get("page_number") or 0)
            and _high_overlap(
                content,
                item.get("content", ""),
                n=ngram_size,
                threshold=overlap_threshold,
            )
            for item in admitted
        )
        if overlaps:
            same_page_overlap_count += 1
            continue
        normalized_seen.add(normalized)
        admitted.append(candidate)
        if len(admitted) >= max_evidence:
            break
    return admitted, {
        "input_count": len(candidates),
        "exact_duplicate_count": exact_duplicate_count,
        "same_page_overlap_count": same_page_overlap_count,
        "invalid_provenance_count": invalid_provenance_count,
        "output_count": len(admitted),
    }


def _load_reranker(model_path: Path, device: str) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(str(model_path), device=device, local_files_only=True)


def _release_model(model: Any) -> None:
    del model
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def build_paired_sample(
    *,
    sample_id: str,
    question: str,
    baseline_candidates: list[dict[str, Any]],
    graph_index: dict[str, Any],
    scorer: Any,
    total_budget: int,
    graph_quota: int,
    final_evidence_k: int,
    reranker_batch_size: int,
    dedup_ngram_size: int,
    dedup_overlap_threshold: float,
    runtime_path_catalog: dict[str, Any] | None = None,
    runtime_lexicon: dict[str, Any] | None = None,
    routing_policy: dict[str, Any] | None = None,
    f_method: str = F_METHOD,
    g1_method: str = G1_METHOD,
    candidate_output_field: str = "candidates_top20",
    protected_prefix_budget: int = 0,
) -> dict[str, Any]:
    """Build paired methods from the same exact RRF Top-K candidate source."""
    assert_no_gold_only_content(baseline_candidates)
    assert_no_gold_only_content(graph_index)
    f_pool = graph.expand_candidates(
        question,
        baseline_candidates,
        graph_index,
        total_budget=total_budget,
        graph_quota=0,
    )
    g1_pool = graph.expand_candidates(
        question,
        baseline_candidates,
        graph_index,
        total_budget=total_budget,
        graph_quota=graph_quota,
        runtime_path_catalog=runtime_path_catalog,
        runtime_lexicon=runtime_lexicon,
        routing_policy=routing_policy,
    )
    if len(f_pool) > total_budget or len(g1_pool) > total_budget:
        raise RuntimeError("Candidate budget exceeded")

    if protected_prefix_budget < 0 or protected_prefix_budget > len(
        baseline_candidates
    ):
        raise ValueError("protected_prefix_budget is invalid")
    protected_prefix_keys = {
        str(item["candidate_key"])
        for item in baseline_candidates[:protected_prefix_budget]
    }

    f_keys = [str(item["candidate_key"]) for item in f_pool]
    g1_keys = [str(item["candidate_key"]) for item in g1_pool]
    f_key_set = set(f_keys)
    g1_key_set = set(g1_keys)
    baseline_prefix_preserved = protected_prefix_keys.issubset(
        f_key_set
    ) and protected_prefix_keys.issubset(g1_key_set)
    if not baseline_prefix_preserved:
        raise RuntimeError("Protected baseline prefix was replaced")
    row = {
        "sample_id": sample_id,
        "question": question,
        "methods": {
            f_method: _method_output(
                question=question,
                candidates=f_pool,
                scorer=scorer,
                final_evidence_k=final_evidence_k,
                reranker_batch_size=reranker_batch_size,
                dedup_ngram_size=dedup_ngram_size,
                dedup_overlap_threshold=dedup_overlap_threshold,
                candidate_output_field=candidate_output_field,
            ),
            g1_method: _method_output(
                question=question,
                candidates=g1_pool,
                scorer=scorer,
                final_evidence_k=final_evidence_k,
                reranker_batch_size=reranker_batch_size,
                dedup_ngram_size=dedup_ngram_size,
                dedup_overlap_threshold=dedup_overlap_threshold,
                candidate_output_field=candidate_output_field,
            ),
        },
        "graph_expansion_audit": {
            "candidate_budget": total_budget,
            "graph_quota": graph_quota,
            "baseline_prefix_budget": protected_prefix_budget,
            "baseline_prefix_preserved": baseline_prefix_preserved,
            "baseline_reserve_count": total_budget - protected_prefix_budget,
            "baseline_count": len(f_pool),
            "expanded_count": len(g1_pool),
            "graph_candidate_count": sum(
                item.get("candidate_origin") == "graph_expansion" for item in g1_pool
            ),
            "added_candidate_keys": sorted(g1_key_set - f_key_set),
            "replaced_candidate_keys": sorted(f_key_set - g1_key_set),
            "zero_expansion": not bool(g1_key_set - f_key_set),
        },
    }
    assert_no_gold_only_content(row)
    return row


def _build_prefix_stable_baseline(
    *,
    prefix_candidates: list[dict[str, Any]],
    source_candidates: list[dict[str, Any]],
    prefix_budget: int,
    candidate_budget: int,
) -> list[dict[str, Any]]:
    if prefix_budget <= 0 or prefix_budget > candidate_budget:
        raise ValueError("baseline_prefix_budget is invalid")
    if len(prefix_candidates) < prefix_budget:
        raise ValueError("Exact-control baseline prefix is too short")

    prefix = prefix_candidates[:prefix_budget]
    prefix_keys = [str(item["candidate_key"]) for item in prefix]
    if len(set(prefix_keys)) != prefix_budget:
        raise ValueError("Exact-control baseline prefix contains duplicate candidates")

    baseline = list(prefix)
    seen = set(prefix_keys)
    for candidate in source_candidates:
        key = str(candidate["candidate_key"])
        if key in seen:
            continue
        baseline.append(candidate)
        seen.add(key)
        if len(baseline) == candidate_budget:
            break
    if len(baseline) != candidate_budget:
        raise ValueError("Exact-control source cannot fill the candidate budget")
    return baseline


def _cuda_peak_memory_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    except ImportError:
        pass
    return 0.0


def _reset_cuda_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def run_paired_retrieval(
    *,
    runtime_projection_path: Path,
    exact_results_path: Path,
    exact_audit_path: Path,
    graph_manifest_path: Path,
    pilot_test_path: Path,
    runtime_lexicon_path: Path | None = None,
    routing_manifest_path: Path | None = None,
    output_dir: Path,
    config: dict[str, Any],
    scorer_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Run paired local reranking without reading Validation40 Gold labels."""
    _require_empty_output_dir(output_dir)
    frozen = validate_frozen_inputs(
        config=config,
        runtime_projection_path=runtime_projection_path,
        exact_results_path=exact_results_path,
        exact_audit_path=exact_audit_path,
        graph_manifest_path=graph_manifest_path,
        pilot_test_path=pilot_test_path,
        runtime_lexicon_path=runtime_lexicon_path,
        routing_manifest_path=routing_manifest_path,
    )
    graph_manifest = _read_json(graph_manifest_path)
    if not graph_manifest.get("ready"):
        raise ValueError("Graph manifest is not ready")
    graph_file = graph_manifest["files"]["graph_index"]
    graph_index_path = graph_manifest_path.parent / graph_file["path"]
    graph_index_sha = _verify_hash(
        graph_index_path, graph_file.get("sha256"), "graph index"
    )
    graph_index = _read_json(graph_index_path)
    runtime_lexicon: dict[str, Any] | None = None
    runtime_path_catalog: dict[str, Any] | None = None
    routing_policy: dict[str, Any] | None = None
    if runtime_lexicon_path is not None and routing_manifest_path is not None:
        runtime_lexicon = _read_json(runtime_lexicon_path)
        routing_manifest = _read_json(routing_manifest_path)
        assert_no_gold_only_content(runtime_lexicon)
        assert_no_gold_only_content(routing_manifest)
        if routing_manifest.get("router_version") != path_router.ROUTER_VERSION:
            raise ValueError("routing manifest router_version mismatch")
        routing_policy = config.get("routing_policy")
        if not isinstance(routing_policy, dict):
            raise ValueError("routing_policy is required for source-routed G1")
        assert_no_gold_only_content(routing_policy)
        runtime_path_catalog = path_router.build_runtime_path_catalog(
            graph_index,
            runtime_lexicon,
        )
    runtime_rows = _read_jsonl(runtime_projection_path)
    exact_rows = _read_jsonl(exact_results_path)
    assert_no_gold_only_content(exact_rows)
    assert_no_gold_only_content(graph_index)

    expected_count = int(config.get("expected_count", 40))
    if len(runtime_rows) != expected_count or len(exact_rows) != expected_count:
        raise ValueError("Validation row count mismatch")
    forbidden_runtime = sorted(
        set().union(*(set(row) for row in runtime_rows)) - RUNTIME_FIELDS
    )
    if forbidden_runtime:
        raise ValueError(
            f"Runtime projection contains non-whitelisted fields: {forbidden_runtime}"
        )
    exact_by_id = {str(row.get("sample_id")): row for row in exact_rows}
    if len(exact_by_id) != len(exact_rows):
        raise ValueError("Duplicate sample_id in exact results")

    source_method = str(config.get("source_method", "dense_exact_sparse_rrf"))
    source_budget = int(config.get("source_budget", 20))
    candidate_budget = int(config.get("candidate_budget", 20))
    baseline_prefix_value = config.get("baseline_prefix_budget")
    baseline_prefix_budget = (
        int(baseline_prefix_value) if baseline_prefix_value is not None else None
    )
    candidate_output_field = f"candidates_top{candidate_budget}"
    f_method = str(config.get("f_method", F_METHOD))
    g1_method = str(config.get("g1_method", G1_METHOD))
    if not f_method or not g1_method or f_method == g1_method:
        raise ValueError("F and G1 method identifiers must be distinct")
    if source_budget < candidate_budget:
        raise ValueError("Source budget cannot be smaller than candidate budget")
    if baseline_prefix_budget is not None and not (
        0 < baseline_prefix_budget <= candidate_budget
    ):
        raise ValueError("baseline_prefix_budget is invalid")

    paired_inputs: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for runtime_row in runtime_rows:
        sample_id = str(runtime_row["sample_id"])
        exact_row = exact_by_id.get(sample_id)
        if exact_row is None or str(exact_row.get("question")) != str(runtime_row["question"]):
            raise ValueError("sample_id/question mismatch between runtime and exact results")
        if runtime_row.get("dataset_version") != config.get("dataset_version"):
            raise ValueError("dataset_version mismatch")
        if runtime_row.get("kb_version") != config.get("kb_version"):
            raise ValueError("kb_version mismatch")
        try:
            source_candidates = exact_row["methods"][source_method][str(source_budget)]
        except (KeyError, TypeError) as exc:
            raise ValueError("Exact-control candidate source is missing") from exc
        if not isinstance(source_candidates, list) or len(source_candidates) < candidate_budget:
            raise ValueError("Exact-control candidate budget mismatch")
        if baseline_prefix_budget is None:
            baseline = source_candidates[:candidate_budget]
        else:
            try:
                prefix_candidates = exact_row["methods"][source_method][
                    str(baseline_prefix_budget)
                ]
            except (KeyError, TypeError) as exc:
                raise ValueError("Exact-control baseline prefix is missing") from exc
            if not isinstance(prefix_candidates, list):
                raise ValueError("Exact-control baseline prefix is invalid")
            baseline = _build_prefix_stable_baseline(
                prefix_candidates=prefix_candidates,
                source_candidates=source_candidates,
                prefix_budget=baseline_prefix_budget,
                candidate_budget=candidate_budget,
            )
        paired_inputs.append((runtime_row, baseline))

    if scorer_factory is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        model_path = Path(os.environ[config["reranker_model_path_env"]])
        device = os.environ.get(config.get("reranker_device_env", "RERANKER_DEVICE"), "cuda")
        scorer_factory = lambda: _load_reranker(model_path, device)

    _reset_cuda_peak_memory()
    scorer = scorer_factory()
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    try:
        for runtime_row, baseline in paired_inputs:
            paired = build_paired_sample(
                sample_id=str(runtime_row["sample_id"]),
                question=str(runtime_row["question"]),
                baseline_candidates=baseline,
                graph_index=graph_index,
                scorer=scorer,
                total_budget=candidate_budget,
                graph_quota=int(config.get("graph_quota", 4)),
                final_evidence_k=int(config.get("final_evidence_k", 4)),
                reranker_batch_size=int(config.get("reranker_batch_size", 8)),
                dedup_ngram_size=int(config.get("dedup_ngram_size", 3)),
                dedup_overlap_threshold=float(
                    config.get("dedup_overlap_threshold", 0.75)
                ),
                runtime_path_catalog=runtime_path_catalog,
                runtime_lexicon=runtime_lexicon,
                routing_policy=routing_policy,
                f_method=f_method,
                g1_method=g1_method,
                candidate_output_field=candidate_output_field,
                protected_prefix_budget=baseline_prefix_budget or 0,
            )
            paired["dataset_version"] = runtime_row["dataset_version"]
            paired["kb_version"] = runtime_row["kb_version"]
            results.append(paired)
    finally:
        elapsed = time.perf_counter() - started
        peak_memory_mb = _cuda_peak_memory_mb()
        _release_model(scorer)

    results_filename = str(config.get("results_filename", "validation40_f_g1_exact_results_v0_1.jsonl"))
    audit_filename = str(config.get("audit_filename", "validation40_f_g1_exact_retrieval_audit_v0_1.json"))
    manifest_filename = str(config.get("manifest_filename", "manifest.json"))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / results_filename
    audit_path = output_dir / audit_filename
    _atomic_write(results_path, _jsonl_bytes(results))
    audit = {
        "audit_version": config.get(
            "audit_version",
            "phase7-c1c4e2b2-paired-retrieval-audit-v0.1",
        ),
        "config_version": config.get("config_version"),
        "dataset_version": config.get("dataset_version"),
        "kb_version": config.get("kb_version"),
        "sample_count": len(results),
        "methods": [f_method, g1_method],
        "source_budget": source_budget,
        "candidate_budget": candidate_budget,
        "baseline_prefix_budget": baseline_prefix_budget,
        "candidate_output_field": candidate_output_field,
        "graph_quota": int(config.get("graph_quota", 4)),
        "final_evidence_k": int(config.get("final_evidence_k", 4)),
        "total_latency_seconds": elapsed,
        "mean_latency_seconds_per_sample": elapsed / len(results) if results else 0.0,
        "peak_cuda_memory_mb": peak_memory_mb,
        "graph_index_sha256": graph_index_sha,
        "router_version": (
            path_router.ROUTER_VERSION if runtime_path_catalog is not None else None
        ),
        "routing_policy": routing_policy,
        **frozen,
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0.0,
        "clinical_validation_claimed": False,
    }
    _atomic_write(audit_path, _json_bytes(audit))
    manifest = {
        "manifest_version": config.get(
            "manifest_version",
            "phase7-c1c4e2b2-paired-retrieval-manifest-v0.1",
        ),
        "ready": True,
        "files": {
            "results": {"path": results_path.name, "sha256": sha256_file(results_path)},
            "audit": {"path": audit_path.name, "sha256": sha256_file(audit_path)},
        },
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0.0,
    }
    _atomic_write(output_dir / manifest_filename, _json_bytes(manifest))
    return {"results": results, "audit": audit, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = _read_json(args.config)
    root = args.repo_root.resolve()
    output_dir = args.output_dir or root / config["output_dir"]
    result = run_paired_retrieval(
        runtime_projection_path=root / config["runtime_projection_path"],
        exact_results_path=root / config["exact_results_path"],
        exact_audit_path=root / config["exact_audit_path"],
        graph_manifest_path=root / config["graph_manifest_path"],
        pilot_test_path=root / config["pilot_test_path"],
        runtime_lexicon_path=(
            root / config["runtime_lexicon_path"]
            if config.get("runtime_lexicon_path")
            else None
        ),
        routing_manifest_path=(
            root / config["routing_manifest_path"]
            if config.get("routing_manifest_path")
            else None
        ),
        output_dir=output_dir,
        config=config,
    )
    print(json.dumps(result["audit"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
