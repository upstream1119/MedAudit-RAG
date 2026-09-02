"""Validation40 offline retrieval execution and cache materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


PLAN_FIELDS = {
    "dataset_version",
    "evidence_context_max",
    "evidence_context_min",
    "execution_status",
    "kb_version",
    "method_id",
    "method_version",
    "preflight_version",
    "question",
    "retrieval_cache_key",
    "retrieval_mode",
    "retrieval_task_id",
    "retrieval_top_k",
    "sample_id",
    "seed",
    "use_graph",
}

OUTPUT_FILENAMES = {
    "physical_results": "validation40_physical_retrieval_cache_v0_1.jsonl",
    "task_results": "validation40_retrieval_results_v0_1.jsonl",
    "failures": "validation40_retrieval_failures_v0_1.jsonl",
    "audit": "validation40_retrieval_audit_v0_1.json",
    "summary_markdown": "validation40_retrieval_summary_v0_1.md",
}


def compute_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _canonical_jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


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


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _physical_profile(row: dict[str, Any], naive_granularity: int) -> dict[str, Any]:
    mode = row["retrieval_mode"]
    if mode == "single_granularity":
        granularity: int | None = naive_granularity
        profile = "single_granularity"
    elif mode in {"multi_granularity", "multi_granularity_graph"}:
        granularity = None
        profile = "multi_granularity"
    else:
        raise ValueError(f"unsupported retrieval_mode: {mode}")
    return {
        "sample_id": row["sample_id"],
        "question": row["question"],
        "kb_version": row["kb_version"],
        "profile": profile,
        "granularity": granularity,
        "top_k": row["retrieval_top_k"],
    }


def _project_chunk(chunk: object) -> dict[str, Any]:
    if is_dataclass(chunk):
        raw = asdict(chunk)
    elif isinstance(chunk, dict):
        raw = dict(chunk)
    else:
        raw = dict(vars(chunk))
    return {
        "content": str(raw.get("content", "")),
        "granularity": int(raw.get("granularity", 0)),
        "distance": float(raw.get("distance", 0.0)),
        "relevance_score": float(raw.get("relevance_score", 0.0)),
        "authority_weight": float(raw.get("authority_weight", 0.0)),
        "final_score": float(raw.get("final_score", 0.0)),
        "source_file": str(raw.get("source_file", "")),
        "page_number": int(raw.get("page_number", 0)),
        "chapter_title": str(raw.get("chapter_title", "")),
        "block_type": str(raw.get("block_type", "text")),
    }


def _normalize_evidence_text(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _is_title_only_evidence(evidence: dict[str, Any]) -> bool:
    content = _normalize_evidence_text(evidence["content"])
    source_title = _normalize_evidence_text(Path(evidence["source_file"]).stem)
    return bool(content and source_title and content == source_title)


def _admit_evidence(
    chunks: list[object],
    *,
    evidence_max: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    evidence: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    audit = {
        "raw_chunk_count": len(chunks),
        "admitted_evidence_count": 0,
        "duplicate_count": 0,
        "title_only_count": 0,
        "invalid_provenance_count": 0,
    }

    for chunk in chunks:
        projected = _project_chunk(chunk)
        content = projected["content"].strip()
        source_file = projected["source_file"].strip()
        page_number = projected["page_number"]
        if not content or not source_file or page_number <= 0:
            audit["invalid_provenance_count"] += 1
            continue
        if _is_title_only_evidence(projected):
            audit["title_only_count"] += 1
            continue

        dedup_key = (
            source_file.casefold(),
            page_number,
            _normalize_evidence_text(content),
        )
        if dedup_key in seen:
            audit["duplicate_count"] += 1
            continue
        seen.add(dedup_key)
        if len(evidence) < evidence_max:
            evidence.append(projected)

    audit["admitted_evidence_count"] = len(evidence)
    return evidence, audit


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _verify_file_hash(path: Path, expected: str, label: str) -> str:
    observed = compute_sha256(path)
    if observed.lower() != str(expected).strip().lower():
        raise ValueError(f"{label} SHA-256 mismatch")
    return observed


def validate_frozen_inputs(
    config: dict[str, Any],
    *,
    repo_root: str | Path,
    runtime_embedding_provider: str,
    runtime_embedding_model: str,
    runtime_chroma_persist_dir: str | Path,
) -> dict[str, Any]:
    """Validate frozen retrieval inputs without reading Pilot Test80 content."""
    root = Path(repo_root).resolve()
    expected_provider = str(config["expected_embedding_provider"]).strip()
    expected_model = str(config["expected_embedding_model"]).strip()
    if runtime_embedding_provider.strip() != expected_provider:
        raise ValueError("runtime embedding provider mismatch")
    if runtime_embedding_model.strip() != expected_model:
        raise ValueError("runtime embedding model mismatch")

    expected_chroma_dir = (root / str(config["chroma_persist_dir"])).resolve()
    runtime_chroma_dir = Path(runtime_chroma_persist_dir)
    if not runtime_chroma_dir.is_absolute():
        runtime_chroma_dir = root / runtime_chroma_dir
    runtime_chroma_dir = runtime_chroma_dir.resolve()
    if runtime_chroma_dir != expected_chroma_dir:
        raise ValueError("runtime Chroma persist directory mismatch")

    plan_path = root / str(config["retrieval_plan_path"])
    status_path = (root / str(config["index_status_path"])).resolve()
    if status_path.parent != expected_chroma_dir:
        raise ValueError("index status path does not belong to configured Chroma directory")
    summary_path = root / str(config["index_rebuild_summary_path"])
    pilot_path = root / str(config["pilot_test_path"])
    hashes = {
        "retrieval_plan_sha256": _verify_file_hash(
            plan_path,
            config["expected_retrieval_plan_sha256"],
            "retrieval plan",
        ),
        "index_status_sha256": _verify_file_hash(
            status_path,
            config["expected_index_status_sha256"],
            "index status",
        ),
        "index_rebuild_summary_sha256": _verify_file_hash(
            summary_path,
            config["expected_index_rebuild_summary_sha256"],
            "index rebuild summary",
        ),
        "pilot_test_sha256_before": _verify_file_hash(
            pilot_path,
            config["expected_pilot_test_sha256"],
            "Pilot Test80",
        ),
    }

    status = _load_json(status_path)
    expected_source_count = int(config["expected_index_source_count"])
    expected_sources = status.get("expected_sources", [])
    indexed_sources = status.get("indexed_sources", [])
    if status.get("ready") is not True:
        raise ValueError("index status is not ready")
    if len(expected_sources) != expected_source_count:
        raise ValueError("expected index source count mismatch")
    if len(indexed_sources) != expected_source_count:
        raise ValueError("indexed source count mismatch")
    if set(expected_sources) != set(indexed_sources) or status.get("missing_sources"):
        raise ValueError("index source completeness mismatch")

    rebuild_summary = _load_json(summary_path)
    rebuild_status = rebuild_summary.get("index_status", {})
    if rebuild_summary.get("pdf_count") != expected_source_count:
        raise ValueError("index rebuild PDF count mismatch")
    if rebuild_status.get("embedding_provider") != expected_provider:
        raise ValueError("index embedding provider mismatch")
    if rebuild_status.get("embedding_model") != expected_model:
        raise ValueError("index embedding model mismatch")

    return {
        **hashes,
        "index_ready": True,
        "index_source_count": expected_source_count,
        "embedding_provider": expected_provider,
        "embedding_model": expected_model,
        "chroma_persist_dir": str(expected_chroma_dir),
        "pilot_test_accessed": False,
    }


def _validate_plan_rows(
    plan_rows: list[dict[str, Any]], config: dict[str, Any]
) -> None:
    expected_dataset = config["expected_dataset_version"]
    expected_kb = config["expected_kb_version"]
    for row in plan_rows:
        if set(row) != PLAN_FIELDS:
            raise ValueError("retrieval plan field allowlist mismatch")
        if row.get("dataset_version") != expected_dataset:
            raise ValueError("dataset version mismatch")
        if row.get("kb_version") != expected_kb:
            raise ValueError("KB version mismatch")


def _validate_execution_config(config: dict[str, Any]) -> None:
    required = (
        "config_version",
        "execution_version",
        "retrieval_plan_path",
        "output_dir",
        "cache_dir",
        "chroma_persist_dir",
        "expected_retrieval_task_count",
        "expected_retrieval_plan_sha256",
        "expected_dataset_version",
        "expected_kb_version",
        "expected_embedding_provider",
        "expected_embedding_model",
    )
    missing = [field for field in required if not str(config.get(field, "")).strip()]
    if missing:
        raise ValueError(f"config fields missing: {', '.join(missing)}")
    if config.get("execute_retrieval") is not True:
        raise ValueError("execute_retrieval must be true")
    if config.get("execute_model_calls") is not False:
        raise ValueError("execute_model_calls must remain false")
    if config.get("graph_reranking_enabled") is not False:
        raise ValueError("graph reranking must remain disabled in C1c-1")
    evidence_min = int(config.get("evidence_context_min", 0))
    evidence_max = int(config.get("evidence_context_max", 0))
    if evidence_min <= 0 or evidence_max < evidence_min:
        raise ValueError("invalid evidence context bounds")


def _load_physical_cache(
    cache_dir: Path, config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    cached: dict[str, dict[str, Any]] = {}
    if not cache_dir.exists():
        return cached
    for path in sorted(cache_dir.glob("*.json")):
        payload = _load_json(path)
        if payload.get("execution_version") != config["execution_version"]:
            continue
        if payload.get("kb_version") != config["expected_kb_version"]:
            continue
        if payload.get("embedding_provider") != config["expected_embedding_provider"]:
            continue
        if payload.get("embedding_model") != config["expected_embedding_model"]:
            continue
        if payload.get("chroma_persist_dir") != config["chroma_persist_dir"]:
            continue
        if payload.get("index_status_sha256") != config["expected_index_status_sha256"]:
            continue
        key = str(payload.get("physical_retrieval_key", ""))
        if key:
            cached[key] = payload
    return cached


def _atomic_replace(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", delete=False, dir=path.parent, prefix=f".{path.name}."
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def _write_physical_cache(
    physical_results: list[dict[str, Any]], cache_dir: Path
) -> None:
    for row in physical_results:
        key = row["physical_retrieval_key"]
        _atomic_replace(cache_dir / f"{key}.json", _canonical_json_bytes(row))


def _summary_markdown(audit: dict[str, Any]) -> str:
    status = audit["physical_status_counts"]
    return "\n".join(
        [
            "# Validation40 离线检索与证据审计摘要",
            "",
            f"- 执行版本：`{audit['execution_version']}`",
            f"- Dataset：`{audit['dataset_version']}`",
            f"- KB：`{audit['kb_version']}`",
            f"- 逻辑检索任务：{audit['logical_retrieval_task_count']}",
            f"- 物理检索任务：{audit['physical_retrieval_count']}",
            f"- 完成：{status.get('completed', 0)}",
            f"- 证据不足：{status.get('insufficient_evidence', 0)}",
            f"- 技术失败：{status.get('failed', 0)}",
            f"- 标题占位过滤：{audit['title_only_evidence_count']}",
            f"- 重复证据过滤：{audit['duplicate_evidence_count']}",
            f"- 无效来源过滤：{audit['invalid_provenance_count']}",
            "- Graph reranking executed: `false`",
            "- External model calls: `0`",
            "- Pilot Test80 accessed: `false`",
            "",
            "本阶段只冻结 Validation40 的检索证据上下文，不生成医学回答，",
            "也不构成 Graph-enhanced 方法效果、独立专家验证或临床验证证据。",
            "",
        ]
    )


def _write_immutable_outputs(
    result: dict[str, Any], output_dir: Path
) -> dict[str, str]:
    physical_rows = sorted(
        result["physical_results"], key=lambda row: row["physical_retrieval_key"]
    )
    payloads = {
        OUTPUT_FILENAMES["physical_results"]: _canonical_jsonl_bytes(physical_rows),
        OUTPUT_FILENAMES["task_results"]: _canonical_jsonl_bytes(result["task_results"]),
        OUTPUT_FILENAMES["failures"]: _canonical_jsonl_bytes(result["failures"]),
        OUTPUT_FILENAMES["audit"]: _canonical_json_bytes(result["audit"]),
        OUTPUT_FILENAMES["summary_markdown"]: _summary_markdown(result["audit"]).encode(
            "utf-8"
        ),
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


def build_retrieval_execution(
    plan_rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    observed_plan_sha256: str,
    retriever: object,
    cached_physical_results: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute unique retrieval profiles while preserving all logical tasks."""
    if observed_plan_sha256 != config["expected_retrieval_plan_sha256"]:
        raise ValueError("retrieval plan SHA-256 mismatch")
    if len(plan_rows) != config["expected_retrieval_task_count"]:
        raise ValueError("retrieval task count mismatch")
    if config.get("execute_model_calls") is not False:
        raise ValueError("execute_model_calls must remain false")

    _validate_plan_rows(plan_rows, config)

    naive_granularity = int(config["naive_granularity"])
    evidence_min = int(config.get("evidence_context_min", 2))
    evidence_max = int(config["evidence_context_max"])
    physical_cache: dict[str, dict[str, Any]] = {}
    task_results = []
    failures = []
    reusable_cache = cached_physical_results or {}

    for row in plan_rows:
        profile = _physical_profile(row, naive_granularity)
        physical_key = _canonical_sha256(profile)
        if physical_key not in physical_cache:
            cached = reusable_cache.get(physical_key)
            if cached and cached.get("status") in {
                "completed",
                "insufficient_evidence",
            }:
                if cached.get("profile") != profile:
                    raise ValueError("cached physical retrieval profile mismatch")
                physical_cache[physical_key] = dict(cached)
            else:
                try:
                    chunks = retriever.retrieve(
                        profile["question"],
                        top_k=profile["top_k"],
                        granularity=profile["granularity"],
                    )
                    evidence, evidence_audit = _admit_evidence(
                        list(chunks), evidence_max=evidence_max
                    )
                    status = (
                        "completed"
                        if len(evidence) >= evidence_min
                        else "insufficient_evidence"
                    )
                except Exception as exc:
                    evidence = []
                    evidence_audit = {
                        "raw_chunk_count": 0,
                        "admitted_evidence_count": 0,
                        "duplicate_count": 0,
                        "title_only_count": 0,
                        "invalid_provenance_count": 0,
                    }
                    status = "failed"
                    error = {
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                    failures.append(
                        {
                            "physical_retrieval_key": physical_key,
                            "sample_id": profile["sample_id"],
                            "profile": profile["profile"],
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    )
                physical_cache[physical_key] = {
                    "physical_retrieval_key": physical_key,
                    "profile": profile,
                    "evidence": evidence,
                    "evidence_audit": evidence_audit,
                    "status": status,
                }
                if status == "failed":
                    physical_cache[physical_key]["error"] = error

        physical_result = physical_cache[physical_key]
        task_results.append(
            {
                **row,
                "execution_status": physical_result["status"],
                "physical_retrieval_key": physical_key,
                "evidence": physical_result["evidence"],
                "graph_reranking_executed": False,
            }
        )

    physical_results = list(physical_cache.values())
    return {
        "physical_results": physical_results,
        "task_results": task_results,
        "failures": failures,
        "audit": {
            "logical_retrieval_task_count": len(task_results),
            "physical_retrieval_count": len(physical_cache),
            "failed_physical_retrieval_count": len(failures),
            "duplicate_evidence_count": sum(
                item["evidence_audit"]["duplicate_count"]
                for item in physical_results
            ),
            "title_only_evidence_count": sum(
                item["evidence_audit"]["title_only_count"]
                for item in physical_results
            ),
            "invalid_provenance_count": sum(
                item["evidence_audit"]["invalid_provenance_count"]
                for item in physical_results
            ),
            "external_model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost": 0,
            "graph_reranking_executed": False,
        },
    }


def _select_sample_limit(
    plan_rows: list[dict[str, Any]], limit: int | None
) -> list[dict[str, Any]]:
    if limit is None:
        return plan_rows
    if limit <= 0:
        raise ValueError("limit must be positive")
    selected_ids: list[str] = []
    for row in plan_rows:
        sample_id = row["sample_id"]
        if sample_id not in selected_ids:
            selected_ids.append(sample_id)
        if len(selected_ids) == limit:
            break
    selected = set(selected_ids)
    return [row for row in plan_rows if row["sample_id"] in selected]


def _validate_method_matrix(plan_rows: list[dict[str, Any]]) -> int:
    expected_methods = {
        "naive_rag",
        "multi_granularity_rag",
        "trust_gated_rag",
        "graph_enhanced_full",
    }
    by_sample: dict[str, set[str]] = {}
    for row in plan_rows:
        by_sample.setdefault(row["sample_id"], set()).add(row["method_id"])
    for sample_id, methods in by_sample.items():
        if methods != expected_methods:
            raise ValueError(f"retrieval method matrix mismatch: {sample_id}")
    if len(plan_rows) != len(by_sample) * len(expected_methods):
        raise ValueError("duplicate retrieval method task detected")
    return len(by_sample)


def _default_embedding_identity(repo_root: Path) -> tuple[str, str, str]:
    backend_path = str(repo_root / "backend")
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    from app.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    return (
        settings.EMBEDDING_PROVIDER,
        settings.EMBEDDING_MODEL,
        settings.CHROMA_PERSIST_DIR,
    )


def _create_default_retriever(repo_root: Path) -> object:
    backend_path = str(repo_root / "backend")
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    from app.knowledge.retriever import MultiGranularityRetriever

    return MultiGranularityRetriever()


def run(
    config_path: str | Path,
    *,
    repo_root: str | Path = ROOT,
    retriever: object | None = None,
    runtime_embedding_provider: str | None = None,
    runtime_embedding_model: str | None = None,
    runtime_chroma_persist_dir: str | Path | None = None,
    limit: int | None = None,
    output_dir_override: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root)
    config = _load_json(Path(config_path))
    _validate_execution_config(config)

    create_retriever_after_validation = retriever is None
    if create_retriever_after_validation:
        (
            runtime_embedding_provider,
            runtime_embedding_model,
            runtime_chroma_persist_dir,
        ) = _default_embedding_identity(root)
    provider = runtime_embedding_provider or ""
    model = runtime_embedding_model or ""
    chroma_persist_dir = runtime_chroma_persist_dir or ""
    provenance = validate_frozen_inputs(
        config,
        repo_root=root,
        runtime_embedding_provider=provider,
        runtime_embedding_model=model,
        runtime_chroma_persist_dir=chroma_persist_dir,
    )
    if create_retriever_after_validation:
        retriever = _create_default_retriever(root)
    assert retriever is not None

    plan_path = root / str(config["retrieval_plan_path"])
    all_plan_rows = _read_jsonl(plan_path)
    _validate_plan_rows(all_plan_rows, config)
    if len(all_plan_rows) != int(config["expected_retrieval_task_count"]):
        raise ValueError("retrieval task count mismatch")
    plan_rows = _select_sample_limit(all_plan_rows, limit)
    sample_count = _validate_method_matrix(plan_rows)
    effective_config = dict(config)
    effective_config["expected_retrieval_task_count"] = len(plan_rows)

    cache_dir = root / str(config["cache_dir"])
    cached = _load_physical_cache(cache_dir, config)
    result = build_retrieval_execution(
        plan_rows,
        effective_config,
        observed_plan_sha256=provenance["retrieval_plan_sha256"],
        retriever=retriever,
        cached_physical_results=cached,
    )

    cache_identity = {
        "execution_version": config["execution_version"],
        "dataset_version": config["expected_dataset_version"],
        "kb_version": config["expected_kb_version"],
        "embedding_provider": config["expected_embedding_provider"],
        "embedding_model": config["expected_embedding_model"],
        "chroma_persist_dir": config["chroma_persist_dir"],
        "index_status_sha256": provenance["index_status_sha256"],
    }
    for row in result["physical_results"]:
        row.update(cache_identity)
    _write_physical_cache(result["physical_results"], cache_dir)

    expected_physical_count = sample_count * 2
    if result["audit"]["physical_retrieval_count"] != expected_physical_count:
        raise ValueError("physical retrieval count mismatch")
    pilot_path = root / str(config["pilot_test_path"])
    pilot_hash_after = _verify_file_hash(
        pilot_path,
        config["expected_pilot_test_sha256"],
        "Pilot Test80 after retrieval",
    )

    status_counts = Counter(row["status"] for row in result["physical_results"])
    method_counts = Counter(row["method_id"] for row in result["task_results"])
    result["audit"].update(
        {
            **provenance,
            "pilot_test_sha256_after": pilot_hash_after,
            "execution_version": config["execution_version"],
            "dataset_version": config["expected_dataset_version"],
            "kb_version": config["expected_kb_version"],
            "sample_count": sample_count,
            "expected_physical_retrieval_count": expected_physical_count,
            "physical_status_counts": dict(sorted(status_counts.items())),
            "method_task_counts": dict(sorted(method_counts.items())),
            "gold_only_field_leakage_count": 0,
            "clinically_validated": False,
        }
    )
    if result["failures"]:
        raise RuntimeError(
            "technical retrieval failures were cached; rerun retries failed cases only"
        )

    output_dir = (
        Path(output_dir_override)
        if output_dir_override is not None
        else root / str(config["output_dir"])
    )
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_hashes = _write_immutable_outputs(result, output_dir)
    return {
        **result["audit"],
        "retrieval_plan_path": str(plan_path),
        "cache_dir": str(cache_dir),
        "output_dir": str(output_dir),
        "output_sha256": output_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute Validation40 offline retrieval without model calls."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    summary = run(
        args.config,
        repo_root=args.repo_root,
        limit=args.limit,
        output_dir_override=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
