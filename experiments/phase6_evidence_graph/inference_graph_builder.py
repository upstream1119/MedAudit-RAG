"""Deterministic runtime-only inference graph builder for Phase 6-A."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from .graph_contract import (
    assert_no_gold_only_content,
    evidence_node_id,
    normalize_id_text,
    question_node_id,
    source_node_id,
    validate_inference_graph,
)


BUILDER_VERSION = "phase6a-runtime-builder-v0.1"
_REQUIRED_VERSION_FIELDS = ("schema_version", "dataset_version", "kb_version")


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    return float(value)


def _admitted_source_indexes(
    source_registry: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    sources = source_registry.get("sources")
    if not isinstance(sources, list):
        raise ValueError("source_registry.sources must be a list")

    by_filename: dict[str, dict[str, Any]] = {}
    admitted_source_ids: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("every source registry entry must be a dictionary")
        if source.get("status") != "indexed" or source.get("included_in_kb") is not True:
            continue

        source_id = _require_text(source.get("source_id"), "source.source_id")
        filename = _require_text(source.get("filename"), f"{source_id}.filename")
        normalized_filename = normalize_id_text(filename)
        if source_id in admitted_source_ids:
            raise ValueError(f"duplicate admitted source_id: {source_id}")
        if normalized_filename in by_filename:
            raise ValueError(f"duplicate admitted source filename: {filename}")

        admitted_source_ids.add(source_id)
        by_filename[normalized_filename] = source

    return by_filename, admitted_source_ids


def _runtime_router_properties(router_output: dict[str, Any]) -> dict[str, str]:
    normalized_query = _require_text(
        router_output.get("normalized_query"),
        "router_output.normalized_query",
    )
    raw_intent = router_output.get("intent")
    intent = getattr(raw_intent, "value", raw_intent)
    return {
        "normalized_query": normalized_query,
        "intent": _require_text(intent, "router_output.intent"),
    }


def _runtime_versions(versions: dict[str, Any]) -> dict[str, str]:
    runtime_versions = {
        field: _require_text(versions.get(field), f"versions.{field}")
        for field in _REQUIRED_VERSION_FIELDS
    }
    runtime_versions["builder_version"] = BUILDER_VERSION
    return runtime_versions


def _retrieval_observation(
    chunk: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    page_number = chunk.get("page_number")
    if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:
        raise ValueError("retrieved_evidence.page_number must be a positive integer")

    granularity = chunk.get("granularity")
    if isinstance(granularity, bool) or not isinstance(granularity, int) or granularity < 1:
        raise ValueError("retrieved_evidence.granularity must be a positive integer")

    return {
        "source_id": _require_text(source.get("source_id"), "source.source_id"),
        "source_file": _require_text(chunk.get("source_file"), "retrieved_evidence.source_file"),
        "page_number": page_number,
        "content": _require_text(chunk.get("content"), "retrieved_evidence.content"),
        "granularity": granularity,
        "distance": _number(chunk.get("distance"), "retrieved_evidence.distance"),
        "relevance_score": _number(
            chunk.get("relevance_score"),
            "retrieved_evidence.relevance_score",
        ),
        "authority_weight": _number(
            chunk.get("authority_weight"),
            "retrieved_evidence.authority_weight",
        ),
        "final_score": _number(chunk.get("final_score"), "retrieved_evidence.final_score"),
        "chapter_title": str(chunk.get("chapter_title") or ""),
        "block_type": str(chunk.get("block_type") or "text"),
    }


def _observation_rank(observation: dict[str, Any]) -> tuple[float, float, float, int, str]:
    stable_tiebreaker = json.dumps(observation, ensure_ascii=False, sort_keys=True)
    return (
        observation["final_score"],
        observation["relevance_score"],
        observation["authority_weight"],
        -observation["granularity"],
        stable_tiebreaker,
    )


def _source_node(source: dict[str, Any]) -> dict[str, Any]:
    source_id = source["source_id"]
    return {
        "id": source_node_id(source_id),
        "type": "SourceDocument",
        "properties": {
            "source_id": source_id,
            "source_title": str(source.get("title") or ""),
            "filename": source["filename"],
            "source_type": str(source.get("source_type") or ""),
            "authority_level": str(source.get("authority_level") or ""),
            "year": source.get("year"),
            "status": source["status"],
        },
        "provenance": "source_registry",
    }


def build_inference_graph(
    *,
    sample_id: str,
    question: str,
    router_output: dict[str, Any],
    retrieved_evidence: list[dict[str, Any]],
    source_registry: dict[str, Any],
    versions: dict[str, Any],
    empty_evidence_reason: str = "retrieval_returned_no_admitted_evidence",
) -> dict[str, Any]:
    """Build and validate a graph without reading benchmark gold annotations."""
    sample_id = _require_text(sample_id, "sample_id")
    question = _require_text(question, "question")
    if not isinstance(router_output, dict):
        raise ValueError("router_output must be a dictionary")
    if not isinstance(retrieved_evidence, list):
        raise ValueError("retrieved_evidence must be a list")
    if not isinstance(source_registry, dict):
        raise ValueError("source_registry must be a dictionary")
    if not isinstance(versions, dict):
        raise ValueError("versions must be a dictionary")

    assert_no_gold_only_content(
        {
            "router_output": router_output,
            "retrieved_evidence": retrieved_evidence,
        }
    )
    source_by_filename, admitted_source_ids = _admitted_source_indexes(source_registry)

    observations_by_evidence_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sources_used: dict[str, dict[str, Any]] = {}
    for chunk in retrieved_evidence:
        if not isinstance(chunk, dict):
            raise ValueError("every retrieved evidence item must be a dictionary")

        source_file = _require_text(
            chunk.get("source_file"),
            "retrieved_evidence.source_file",
        )
        source = source_by_filename.get(normalize_id_text(source_file))
        if source is None:
            raise ValueError(f"retrieved evidence source is not admitted: {source_file}")

        observation = _retrieval_observation(chunk, source)
        node_id = evidence_node_id(
            observation["source_id"],
            observation["page_number"],
            observation["content"],
        )
        observations_by_evidence_id[node_id].append(observation)
        sources_used[observation["source_id"]] = source

    evidence_nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    for node_id in sorted(observations_by_evidence_id):
        observations = observations_by_evidence_id[node_id]
        best = max(observations, key=_observation_rank)
        retrieved_granularities = sorted(
            {observation["granularity"] for observation in observations}
        )
        evidence_nodes.append(
            {
                "id": node_id,
                "type": "EvidenceSpan",
                "properties": {
                    **best,
                    "retrieved_granularities": retrieved_granularities,
                },
                "provenance": "runtime_retrieval",
            }
        )
        edges.append(
            {
                "source": node_id,
                "target": source_node_id(best["source_id"]),
                "type": "EVIDENCE_FROM_SOURCE",
            }
        )

    has_evidence = bool(evidence_nodes)
    graph = {
        "graph_id": f"inference_graph::{sample_id}",
        "graph_type": "inference_graph",
        "sample_id": sample_id,
        "versions": _runtime_versions(versions),
        "build_status": "success" if has_evidence else "empty_evidence",
        "failure_reason": (
            None if has_evidence else _require_text(empty_evidence_reason, "empty_evidence_reason")
        ),
        "nodes": [
            {
                "id": question_node_id(sample_id),
                "type": "Question",
                "properties": {
                    "text": question,
                    "router_output": _runtime_router_properties(router_output),
                },
                "provenance": "runtime_question",
            },
            *[
                _source_node(sources_used[source_id])
                for source_id in sorted(sources_used)
            ],
            *evidence_nodes,
        ],
        "edges": sorted(
            edges,
            key=lambda edge: (edge["source"], edge["target"], edge["type"]),
        ),
    }
    validate_inference_graph(graph, admitted_source_ids)
    return graph
