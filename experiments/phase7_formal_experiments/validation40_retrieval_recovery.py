"""Validation40 retrieval recovery without external model calls or Gold leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_FIELDS = {"sample_id", "question", "dataset_version", "kb_version"}
OUTPUT_FILENAMES = {
    "retrieval_results": "validation40_retrieval_recovery_results_v0_1.jsonl",
    "retrieval_audit": "validation40_retrieval_recovery_audit_v0_1.json",
    "sample_metrics": "validation40_retrieval_recovery_sample_metrics_v0_1.jsonl",
    "profile_summary": "validation40_retrieval_recovery_profile_summary_v0_1.json",
    "summary_markdown": "validation40_retrieval_recovery_summary_v0_1.md",
}


def compute_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row must be an object at line {line_number}")
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


def _validate_runtime_sample(sample: dict[str, Any]) -> None:
    leaked = sorted(set(sample) - RUNTIME_FIELDS)
    if leaked:
        raise ValueError(f"Gold-only field leakage: {', '.join(leaked)}")
    missing = sorted({"sample_id", "question"} - set(sample))
    if missing:
        raise ValueError(f"runtime fields missing: {', '.join(missing)}")


def _project_chunk(chunk: object) -> dict[str, Any]:
    if is_dataclass(chunk):
        raw = asdict(chunk)
    elif isinstance(chunk, dict):
        raw = dict(chunk)
    else:
        raw = dict(vars(chunk))
    return {
        "content": str(raw.get("content", "")).strip(),
        "granularity": int(raw.get("granularity", 0)),
        "source_file": str(raw.get("source_file", "")).strip(),
        "page_number": int(raw.get("page_number", 0)),
        "distance": float(raw.get("distance", 0.0)),
        "relevance_score": float(raw.get("relevance_score", 0.0)),
        "authority_weight": float(raw.get("authority_weight", 0.0)),
        "final_score": float(raw.get("final_score", 0.0)),
        "chapter_title": str(raw.get("chapter_title", "")).strip(),
        "block_type": str(raw.get("block_type", "text")).strip(),
    }


def _source_page_key(candidate: dict[str, Any]) -> tuple[str, int]:
    source = Path(candidate["source_file"]).name.replace(" ", "").casefold()
    return source, int(candidate["page_number"])


def reciprocal_rank_fuse(
    ranked_lists: dict[int, list[dict[str, Any]]], *, rrf_constant: int
) -> list[dict[str, Any]]:
    if rrf_constant <= 0:
        raise ValueError("rrf_constant must be positive")
    fused: dict[tuple[str, int], dict[str, Any]] = {}
    for granularity in sorted(ranked_lists):
        for rank, candidate in enumerate(ranked_lists[granularity], start=1):
            key = _source_page_key(candidate)
            current = fused.get(key)
            if current is None:
                current = dict(candidate)
                current["rrf_score"] = 0.0
                current["granularity_ranks"] = {}
                current["matched_granularities"] = []
                fused[key] = current
            elif candidate["final_score"] > current["final_score"]:
                preserved = {
                    "rrf_score": current["rrf_score"],
                    "granularity_ranks": current["granularity_ranks"],
                    "matched_granularities": current["matched_granularities"],
                }
                current.update(candidate)
                current.update(preserved)
            current["rrf_score"] += 1.0 / (rrf_constant + rank)
            current["granularity_ranks"][str(granularity)] = rank
            current["matched_granularities"].append(granularity)

    return sorted(
        fused.values(),
        key=lambda item: (
            -float(item["rrf_score"]),
            -float(item["final_score"]),
            item["source_file"],
            int(item["page_number"]),
        ),
    )


def _normalize_text(value: object) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", str(value)).casefold()


def _question_overlap(question: str, content: str) -> float:
    question_chars = set(_normalize_text(question))
    if not question_chars:
        return 0.0
    return len(question_chars & set(_normalize_text(content))) / len(question_chars)


def expand_same_source_neighbors(
    candidates: list[dict[str, Any]],
    *,
    page_provider: object | None,
    radius: int,
) -> tuple[list[dict[str, Any]], int]:
    if radius < 0:
        raise ValueError("neighbor_radius must not be negative")
    if radius == 0 or page_provider is None:
        return candidates, 0

    expanded = list(candidates)
    seen = {_source_page_key(candidate) for candidate in candidates}
    admitted = 0
    for seed_rank, seed in enumerate(candidates, start=1):
        for offset in range(-radius, radius + 1):
            if offset == 0:
                continue
            page_number = int(seed["page_number"]) + offset
            if page_number <= 0:
                continue
            chunk = page_provider.get_page(seed["source_file"], page_number)
            if chunk is None:
                continue
            neighbor = _project_chunk(chunk)
            if _source_page_key(neighbor)[0] != _source_page_key(seed)[0]:
                continue
            if int(neighbor["page_number"]) != page_number:
                continue
            key = _source_page_key(neighbor)
            if key in seen:
                continue
            neighbor.update(
                {
                    "rrf_score": 0.0,
                    "granularity_ranks": {},
                    "matched_granularities": [neighbor["granularity"]],
                    "is_neighbor": True,
                    "neighbor_of": {
                        "source_file": seed["source_file"],
                        "page_number": seed["page_number"],
                        "seed_rank": seed_rank,
                        "offset": offset,
                    },
                }
            )
            expanded.append(neighbor)
            seen.add(key)
            admitted += 1
    return expanded, admitted


def select_evidence_window(
    fused: list[dict[str, Any]],
    expanded: list[dict[str, Any]],
    *,
    question: str,
    final_k: int,
    max_neighbors_per_seed: int,
) -> list[dict[str, Any]]:
    if max_neighbors_per_seed < 0:
        raise ValueError("max_neighbors_per_seed must not be negative")
    neighbors_by_parent: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for candidate in expanded:
        parent = candidate.get("neighbor_of")
        if not isinstance(parent, dict):
            continue
        parent_key = (
            Path(str(parent["source_file"])).name.replace(" ", "").casefold(),
            int(parent["page_number"]),
        )
        neighbors_by_parent.setdefault(parent_key, []).append(candidate)

    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for seed in fused:
        if len(selected) >= final_k:
            break
        seed_key = _source_page_key(seed)
        if seed_key not in seen:
            selected.append(seed)
            seen.add(seed_key)
        neighbors = sorted(
            neighbors_by_parent.get(seed_key, []),
            key=lambda item: (
                -_question_overlap(question, item["content"]),
                abs(int(item["neighbor_of"]["offset"])),
                int(item["neighbor_of"]["offset"]),
            ),
        )
        admitted_for_seed = 0
        for neighbor in neighbors:
            if len(selected) >= final_k or admitted_for_seed >= max_neighbors_per_seed:
                break
            key = _source_page_key(neighbor)
            if key in seen:
                continue
            selected.append(neighbor)
            seen.add(key)
            admitted_for_seed += 1
    return selected


def build_recovery_result(
    *,
    sample: dict[str, Any],
    retriever: object,
    config: dict[str, Any],
    page_provider: object | None = None,
) -> dict[str, Any]:
    _validate_runtime_sample(sample)
    if config.get("execute_model_calls") is not False:
        raise ValueError("execute_model_calls must remain false")
    for flag in ("query_rewrite_enabled", "cross_encoder_enabled"):
        if config.get(flag) is not False:
            raise ValueError(f"{flag} must remain disabled in C1c-3 v0.1")
    candidate_pool_k = int(config["candidate_pool_k"])
    final_evidence_k = int(config["final_evidence_k"])
    if candidate_pool_k <= final_evidence_k or final_evidence_k <= 0:
        raise ValueError("candidate_pool_k must be greater than final_evidence_k")

    question = str(sample["question"]).strip()
    ranked_lists: dict[int, list[dict[str, Any]]] = {}
    for granularity in config["granularities"]:
        chunks = retriever.retrieve(
            question, top_k=candidate_pool_k, granularity=int(granularity)
        )
        ranked_lists[int(granularity)] = [_project_chunk(chunk) for chunk in chunks]

    fused = reciprocal_rank_fuse(
        ranked_lists, rrf_constant=int(config["rrf_constant"])
    )
    expanded, neighbor_count = expand_same_source_neighbors(
        fused,
        page_provider=page_provider,
        radius=int(config.get("neighbor_radius", 0)),
    )
    evidence = select_evidence_window(
        fused,
        expanded,
        question=question,
        final_k=final_evidence_k,
        max_neighbors_per_seed=int(config.get("max_neighbors_per_seed", 1)),
    )
    evidence_min = int(config.get("evidence_context_min", 1))
    return {
        "sample_id": str(sample["sample_id"]),
        "question": question,
        "dataset_version": sample.get("dataset_version"),
        "kb_version": sample.get("kb_version"),
        "candidate_pool_size": len(expanded),
        "status": "completed" if len(evidence) >= evidence_min else "insufficient_evidence",
        "evidence": evidence,
        "audit": {
            "external_model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost": 0,
            "query_rewrite_executed": False,
            "cross_encoder_executed": False,
            "gold_fields_accessed": 0,
            "pilot_test_accessed": False,
            "same_source_neighbor_count": neighbor_count,
            "candidate_pool_k": candidate_pool_k,
            "final_evidence_k": final_evidence_k,
        },
    }


class ChromaPageProvider:
    """Read exact same-source pages; it performs no semantic search."""

    COLLECTION_NAMES = ("context_1024", "concept_512", "detail_128")

    def __init__(self, persist_dir: str | Path):
        import chromadb

        self._client = chromadb.PersistentClient(path=str(persist_dir))

    @staticmethod
    def _usable(content: str, source_file: str) -> bool:
        normalized = _normalize_text(content)
        source_title = _normalize_text(Path(source_file).stem)
        return bool(normalized) and normalized != source_title and len(normalized) >= 20

    def get_page(self, source_file: str, page_number: int):
        candidates: list[dict[str, Any]] = []
        where = {
            "$and": [
                {"source_file": {"$eq": source_file}},
                {"page_number": {"$eq": int(page_number)}},
            ]
        }
        for collection_name in self.COLLECTION_NAMES:
            collection = self._client.get_collection(collection_name)
            payload = collection.get(where=where, include=["documents", "metadatas"])
            documents = payload.get("documents") or []
            metadatas = payload.get("metadatas") or []
            for content, metadata in zip(documents, metadatas):
                if not self._usable(str(content), source_file):
                    continue
                candidates.append(
                    {
                        "content": str(content).strip(),
                        "granularity": int(metadata.get("granularity", 0)),
                        "source_file": str(metadata.get("source_file", source_file)),
                        "page_number": int(metadata.get("page_number", page_number)),
                        "distance": 1.0,
                        "relevance_score": 0.0,
                        "authority_weight": 0.0,
                        "final_score": 0.0,
                        "chapter_title": str(metadata.get("chapter_title", "")),
                        "block_type": str(metadata.get("block_type", "text")),
                    }
                )
        return max(candidates, key=lambda item: len(item["content"]), default=None)


def _validate_run_config(config: dict[str, Any]) -> None:
    required = (
        "recovery_version",
        "runtime_projection_path",
        "output_dir",
        "chroma_persist_dir",
        "index_status_path",
        "expected_runtime_projection_sha256",
        "expected_index_status_sha256",
        "expected_sample_count",
        "expected_dataset_version",
        "expected_kb_version",
    )
    missing = [field for field in required if not str(config.get(field, "")).strip()]
    if missing:
        raise ValueError(f"config fields missing: {', '.join(missing)}")
    if config.get("execute_retrieval") is not True:
        raise ValueError("execute_retrieval must be true")
    if config.get("execute_model_calls") is not False:
        raise ValueError("execute_model_calls must remain false")
    if config.get("query_rewrite_enabled") is not False:
        raise ValueError("query_rewrite_enabled must remain disabled")
    if config.get("cross_encoder_enabled") is not False:
        raise ValueError("cross_encoder_enabled must remain disabled")


def _verify_hash(path: Path, expected: object, label: str) -> str:
    observed = compute_sha256(path)
    if observed != str(expected):
        raise ValueError(f"{label} SHA-256 mismatch")
    return observed


def _normalize_source_name(source: object) -> str:
    return Path(str(source)).stem.replace(" ", "").casefold()


def evaluate_recovery_results(
    results: list[dict[str, Any]], gold_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gold_by_id = {str(row["candidate_id"]): row for row in gold_rows}
    metrics: list[dict[str, Any]] = []
    for result in results:
        gold = gold_by_id[str(result["sample_id"])]
        gold_source = _normalize_source_name(gold["source_filename"])
        gold_page = int(gold["page_number"])
        source_ranks: list[int] = []
        exact_ranks: list[int] = []
        adjacent_ranks: list[int] = []
        for rank, item in enumerate(result["evidence"], start=1):
            if _normalize_source_name(item["source_file"]) != gold_source:
                continue
            source_ranks.append(rank)
            page = int(item["page_number"])
            if page == gold_page:
                exact_ranks.append(rank)
            if abs(page - gold_page) <= 1:
                adjacent_ranks.append(rank)
        metrics.append(
            {
                "sample_id": result["sample_id"],
                "retrieval_status": result["status"],
                "gold_source_hit": bool(source_ranks),
                "gold_source_page_hit": bool(exact_ranks),
                "adjacent_gold_source_page_hit": bool(adjacent_ranks),
                "gold_source_page_rank": exact_ranks[0] if exact_ranks else None,
                "gold_source_page_reciprocal_rank": (
                    1.0 / exact_ranks[0] if exact_ranks else 0.0
                ),
                "retrieved_source_pages": [
                    {
                        "rank": rank,
                        "source_file": item["source_file"],
                        "page_number": item["page_number"],
                        "is_neighbor": bool(item.get("is_neighbor", False)),
                    }
                    for rank, item in enumerate(result["evidence"], start=1)
                ],
            }
        )
    count = len(metrics)
    status_counts = Counter(row["retrieval_status"] for row in metrics)
    summary = {
        "sample_count": count,
        "gold_source_recall_at_k": sum(row["gold_source_hit"] for row in metrics) / count,
        "gold_source_page_recall_at_k": sum(
            row["gold_source_page_hit"] for row in metrics
        )
        / count,
        "adjacent_gold_source_page_recall_at_k": sum(
            row["adjacent_gold_source_page_hit"] for row in metrics
        )
        / count,
        "gold_source_page_mrr": sum(
            row["gold_source_page_reciprocal_rank"] for row in metrics
        )
        / count,
        "insufficient_evidence_rate": status_counts.get("insufficient_evidence", 0)
        / count,
        "status_counts": dict(sorted(status_counts.items())),
    }
    return metrics, summary


def _summary_markdown(audit: dict[str, Any], profile: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Validation40 retrieval recovery v0.1",
            "",
            f"- Recovery version: `{audit['recovery_version']}`",
            f"- Samples: `{audit['sample_count']}`",
            f"- Candidate pool per granularity: `{audit['candidate_pool_k']}`",
            f"- Final evidence window: `{audit['final_evidence_k']}`",
            f"- Gold source recall@K: `{profile['gold_source_recall_at_k']:.4f}`",
            f"- Exact source-page recall@K: `{profile['gold_source_page_recall_at_k']:.4f}`",
            f"- Adjacent source-page recall@K: `{profile['adjacent_gold_source_page_recall_at_k']:.4f}`",
            f"- Exact source-page MRR: `{profile['gold_source_page_mrr']:.4f}`",
            f"- Insufficient evidence rate: `{profile['insufficient_evidence_rate']:.4f}`",
            "- External model calls: `0`",
            "- Query rewrite executed: `false`",
            "- Cross-encoder executed: `false`",
            "- Pilot Test80 accessed: `false`",
            "",
            "Gold is joined only after retrieval results are built. These are anchor-level",
            "retrieval diagnostics, not answer correctness or clinical effectiveness results.",
            "",
        ]
    )


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"output conflict: refusing to overwrite {path}")
        return
    with tempfile.NamedTemporaryFile(
        mode="wb", delete=False, dir=path.parent, prefix=f".{path.name}."
    ) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _create_default_retriever(repo_root: Path, persist_dir: Path) -> object:
    backend_path = str(repo_root / "backend")
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    from app.knowledge.retriever import MultiGranularityRetriever

    return MultiGranularityRetriever(persist_dir=str(persist_dir))


def run(config_path: str | Path, *, repo_root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config = _load_json(Path(config_path))
    _validate_run_config(config)
    runtime_path = root / config["runtime_projection_path"]
    index_status_path = root / config["index_status_path"]
    _verify_hash(
        runtime_path, config["expected_runtime_projection_sha256"], "runtime projection"
    )
    _verify_hash(index_status_path, config["expected_index_status_sha256"], "index status")
    runtime_rows = _read_jsonl(runtime_path)
    if len(runtime_rows) != int(config["expected_sample_count"]):
        raise ValueError("runtime sample count mismatch")
    for row in runtime_rows:
        _validate_runtime_sample(row)
        if row.get("dataset_version") != config["expected_dataset_version"]:
            raise ValueError("dataset version mismatch")
        if row.get("kb_version") != config["expected_kb_version"]:
            raise ValueError("KB version mismatch")

    persist_dir = (root / config["chroma_persist_dir"]).resolve()
    retriever = _create_default_retriever(root, persist_dir)
    page_provider = ChromaPageProvider(persist_dir)
    results = [
        build_recovery_result(
            sample=row,
            retriever=retriever,
            page_provider=page_provider,
            config=config,
        )
        for row in runtime_rows
    ]
    audit = {
        "recovery_version": config["recovery_version"],
        "sample_count": len(results),
        "candidate_pool_k": int(config["candidate_pool_k"]),
        "final_evidence_k": int(config["final_evidence_k"]),
        "dataset_version": config["expected_dataset_version"],
        "kb_version": config["expected_kb_version"],
        "seed": int(config["seed"]),
        "external_model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
        "query_rewrite_executed": False,
        "cross_encoder_executed": False,
        "gold_fields_accessed_during_retrieval": 0,
        "pilot_test_accessed": False,
    }

    evaluation = config["post_retrieval_evaluation"]
    gold_path = root / evaluation["gold_path"]
    _verify_hash(gold_path, evaluation["expected_gold_sha256"], "Validation40 Gold")
    gold_rows = _read_jsonl(gold_path)
    metrics, profile = evaluate_recovery_results(results, gold_rows)
    output_dir = root / config["output_dir"]
    payloads = {
        OUTPUT_FILENAMES["retrieval_results"]: _canonical_jsonl_bytes(results),
        OUTPUT_FILENAMES["retrieval_audit"]: _canonical_json_bytes(audit),
        OUTPUT_FILENAMES["sample_metrics"]: _canonical_jsonl_bytes(metrics),
        OUTPUT_FILENAMES["profile_summary"]: _canonical_json_bytes(profile),
        OUTPUT_FILENAMES["summary_markdown"]: _summary_markdown(audit, profile).encode(
            "utf-8"
        ),
    }
    for filename, content in payloads.items():
        _write_immutable(output_dir / filename, content)
    return {
        **audit,
        "profile_summary": profile,
        "output_dir": str(output_dir),
        "output_sha256": {
            filename: hashlib.sha256(content).hexdigest()
            for filename, content in payloads.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Validation40 retrieval recovery without external models."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root", default=str(ROOT))
    args = parser.parse_args()
    print(json.dumps(run(args.config, repo_root=args.repo_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
