"""Graph-guided evidence reranking and query-constraint auditing for Phase 6-B."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any

from .graph_contract import (
    assert_no_gold_only_content,
    constraint_node_id,
    validate_inference_graph,
)
from .runtime_constraint_extractor import (
    RULESET_VERSION,
    extract_runtime_constraints,
)


ARTIFACT_SCHEMA_VERSION = "phase6b-reranking-artifact-v0.2"
_SCORE_KEYS = ("relevance", "authority", "constraint_type_coverage")
_EXCLUSIVE_CONSTRAINT_TYPES = frozenset(
    {
        "dose",
        "frequency",
        "route",
        "evidence_scope",
    }
)
_INTRAVENOUS_ROUTE_VALUES = frozenset({"iv_unspecified", "iv_infusion"})


def canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    return float(value)


def _clamp01(value: object, field_name: str) -> float:
    return min(1.0, max(0.0, _number(value, field_name)))


def _validated_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("reranking config must be a dictionary")

    top_k = config.get("top_k")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    score_weights = config.get("score_weights")
    if not isinstance(score_weights, dict) or set(score_weights) != set(_SCORE_KEYS):
        raise ValueError(f"score_weights must contain exactly {_SCORE_KEYS}")
    normalized_score_weights = {
        key: _number(score_weights[key], f"score_weights.{key}")
        for key in _SCORE_KEYS
    }
    if any(value < 0 for value in normalized_score_weights.values()):
        raise ValueError("score weights must be non-negative")
    if abs(sum(normalized_score_weights.values()) - 1.0) > 1e-9:
        raise ValueError("score weights must sum to 1")

    constraint_weights = config.get("constraint_type_weights")
    if not isinstance(constraint_weights, dict) or not constraint_weights:
        raise ValueError("constraint_type_weights must be a non-empty dictionary")
    normalized_constraint_weights = {}
    for raw_key, raw_value in constraint_weights.items():
        key = _require_text(raw_key, "constraint_type")
        value = _number(raw_value, f"constraint_type_weights.{key}")
        if value <= 0:
            raise ValueError("constraint type weights must be positive")
        normalized_constraint_weights[key] = value

    return {
        "config_version": _require_text(
            config.get("config_version"),
            "config_version",
        ),
        "method_id": _require_text(config.get("method_id"), "method_id"),
        "method_version": _require_text(
            config.get("method_version"),
            "method_version",
        ),
        "top_k": top_k,
        "score_weights": normalized_score_weights,
        "constraint_type_weights": normalized_constraint_weights,
    }


def _source_ids(graph: dict[str, Any]) -> set[str]:
    return {
        _require_text(node["properties"].get("source_id"), "source_id")
        for node in graph.get("nodes", [])
        if node.get("type") == "SourceDocument"
    }


def _question_node(graph: dict[str, Any]) -> dict[str, Any]:
    question_nodes = [
        node for node in graph["nodes"] if node.get("type") == "Question"
    ]
    if len(question_nodes) != 1:
        raise ValueError("inference graph must contain exactly one Question node")
    return question_nodes[0]


def _evidence_nodes(graph: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        node for node in graph["nodes"] if node.get("type") == "EvidenceSpan"
    ]


def _coverage(
    query_types: set[str],
    evidence_types: set[str],
    constraint_weights: dict[str, float],
) -> float:
    denominator = sum(constraint_weights.get(name, 1.0) for name in query_types)
    if denominator == 0:
        return 0.0
    numerator = sum(
        constraint_weights.get(name, 1.0)
        for name in query_types & evidence_types
    )
    return numerator / denominator


def _dose_values_match(query_value: str, evidence_value: str) -> bool:
    return query_value.removesuffix("/day") == evidence_value.removesuffix("/day")


def _values_match(
    constraint_type: str,
    query_value: str,
    evidence_value: str,
) -> bool:
    if query_value == evidence_value:
        return True
    if constraint_type == "dose":
        return _dose_values_match(query_value, evidence_value)
    if constraint_type == "route":
        # Specific infusion evidence supports a general intravenous query,
        # but general intravenous evidence cannot prove the specific route.
        return query_value == "iv_unspecified" and evidence_value == "iv_infusion"
    return False


def _values_conflict(
    constraint_type: str,
    query_value: str,
    evidence_value: str,
) -> bool:
    if _values_match(constraint_type, query_value, evidence_value):
        return False
    if constraint_type == "route" and {
        query_value,
        evidence_value,
    }.issubset(_INTRAVENOUS_ROUTE_VALUES):
        return False
    return constraint_type in _EXCLUSIVE_CONSTRAINT_TYPES


def _constraint_audit(
    query_constraints: list[dict],
    top_evidence_constraints: list[list[dict]],
) -> list[dict[str, Any]]:
    evidence_values_by_type: dict[str, set[str]] = defaultdict(set)
    for constraints in top_evidence_constraints:
        for constraint in constraints:
            evidence_values_by_type[constraint["constraint_type"]].add(
                constraint["normalized_value"]
            )

    rows = []
    for query_constraint in query_constraints:
        constraint_type = query_constraint["constraint_type"]
        query_value = query_constraint["normalized_value"]
        evidence_values = sorted(evidence_values_by_type.get(constraint_type, set()))
        if any(
            _values_match(constraint_type, query_value, value)
            for value in evidence_values
        ):
            status = "matched"
        elif any(
            _values_conflict(constraint_type, query_value, value)
            for value in evidence_values
        ):
            status = "conflict"
        else:
            status = "unsupported"
        rows.append(
            {
                "constraint_type": constraint_type,
                "query_value": query_value,
                "evidence_values": evidence_values,
                "status": status,
            }
        )
    return rows


def _runtime_constraint_graph(
    *,
    sample_id: str,
    question_id: str,
    query_constraints: list[dict],
    ranked_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    nodes_by_id: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    query_nodes_by_type: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for constraint in query_constraints:
        node_id = constraint_node_id(
            sample_id,
            constraint["constraint_type"],
            constraint["normalized_value"],
        )
        nodes_by_id[node_id] = {
            "id": node_id,
            "type": "MedicationConstraint",
            "properties": dict(constraint),
            "provenance": "runtime_question_constraint",
        }
        query_nodes_by_type[constraint["constraint_type"]].append(
            (node_id, constraint["normalized_value"])
        )
        edges.append(
            {
                "source": question_id,
                "target": node_id,
                "type": "QUESTION_REQUIRES_EVIDENCE_TYPE",
            }
        )

    for row in ranked_rows:
        for constraint in row["runtime_constraints"]:
            node_id = constraint_node_id(
                sample_id,
                constraint["constraint_type"],
                constraint["normalized_value"],
            )
            nodes_by_id.setdefault(
                node_id,
                {
                    "id": node_id,
                    "type": "MedicationConstraint",
                    "properties": dict(constraint),
                    "provenance": "runtime_evidence_constraint",
                },
            )
            edges.append(
                {
                    "source": row["evidence_id"],
                    "target": node_id,
                    "type": "EVIDENCE_MATCHES_CONSTRAINT_TYPE",
                }
            )

            for query_node_id, query_value in query_nodes_by_type.get(
                constraint["constraint_type"],
                [],
            ):
                evidence_value = constraint["normalized_value"]
                if _values_match(
                    constraint["constraint_type"],
                    query_value,
                    evidence_value,
                ):
                    relation = "EVIDENCE_MATCHES_CONSTRAINT_VALUE"
                elif _values_conflict(
                    constraint["constraint_type"],
                    query_value,
                    evidence_value,
                ):
                    relation = "EVIDENCE_CONFLICTS_WITH_CONSTRAINT"
                else:
                    continue
                edge = {
                    "source": row["evidence_id"],
                    "target": query_node_id,
                    "type": relation,
                }
                if relation == "EVIDENCE_CONFLICTS_WITH_CONSTRAINT":
                    edge["properties"] = {
                        "evidence_value": evidence_value
                    }
                edges.append(edge)

    unique_edges = {
        json.dumps(edge, ensure_ascii=False, sort_keys=True): edge for edge in edges
    }
    return {
        "nodes": [nodes_by_id[node_id] for node_id in sorted(nodes_by_id)],
        "edges": [
            unique_edges[key]
            for key in sorted(unique_edges)
        ],
    }


def build_reranking_artifact(
    inference_graph: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build a deterministic method artifact without reading evaluation data."""
    if not isinstance(inference_graph, dict):
        raise ValueError("Phase 6-B accepts inference_graph dictionaries only")
    if inference_graph.get("graph_type") != "inference_graph":
        raise ValueError("Phase 6-B accepts inference_graph only")
    assert_no_gold_only_content(inference_graph)
    validate_inference_graph(inference_graph, _source_ids(inference_graph))
    validated_config = _validated_config(config)

    sample_id = _require_text(inference_graph.get("sample_id"), "sample_id")
    parent_sha256 = canonical_sha256(inference_graph)
    question_node = _question_node(inference_graph)
    question_properties = question_node.get("properties", {})
    router_output = question_properties.get("router_output") or {}
    query_constraints = extract_runtime_constraints(
        _require_text(question_properties.get("text"), "question.text"),
        str(router_output.get("normalized_query") or ""),
    )

    artifact_status = "success"
    if inference_graph.get("build_status") == "empty_evidence":
        artifact_status = (
            "boundary_refusal"
            if inference_graph.get("failure_reason")
            == "prescription_boundary_detected"
            else "insufficient_graph_evidence"
        )

    evidence_nodes = _evidence_nodes(inference_graph)
    vector_ranked = sorted(
        evidence_nodes,
        key=lambda node: (
            -_number(node["properties"].get("final_score"), "final_score"),
            node["id"],
        ),
    )
    rank_before = {
        node["id"]: index for index, node in enumerate(vector_ranked, 1)
    }

    query_types = {
        constraint["constraint_type"] for constraint in query_constraints
    }
    scored_rows: list[dict[str, Any]] = []
    for node in evidence_nodes:
        properties = node["properties"]
        runtime_constraints = extract_runtime_constraints(
            _require_text(properties.get("content"), "evidence.content")
        )
        evidence_types = {
            constraint["constraint_type"] for constraint in runtime_constraints
        }
        coverage = _coverage(
            query_types,
            evidence_types,
            validated_config["constraint_type_weights"],
        )
        relevance = _clamp01(
            properties.get("relevance_score"),
            "evidence.relevance_score",
        )
        authority = _clamp01(
            properties.get("authority_weight"),
            "evidence.authority_weight",
        )
        weights = validated_config["score_weights"]
        graph_score = (
            weights["relevance"] * relevance
            + weights["authority"] * authority
            + weights["constraint_type_coverage"] * coverage
        )
        scored_rows.append(
            {
                "evidence_id": node["id"],
                "source_id": properties["source_id"],
                "source_file": properties["source_file"],
                "page_number": properties["page_number"],
                "content": properties["content"],
                "rank_before": rank_before[node["id"]],
                "relevance_score": relevance,
                "authority_weight": authority,
                "original_final_score": properties["final_score"],
                "runtime_constraints": runtime_constraints,
                "matched_constraint_types": sorted(query_types & evidence_types),
                "score_components": {
                    "relevance": round(weights["relevance"] * relevance, 10),
                    "authority": round(weights["authority"] * authority, 10),
                    "constraint_type_coverage": round(
                        weights["constraint_type_coverage"] * coverage,
                        10,
                    ),
                },
                "constraint_type_coverage": round(coverage, 10),
                "graph_score": round(graph_score, 10),
            }
        )

    has_constraint_coverage = any(
        row["constraint_type_coverage"] > 0 for row in scored_rows
    )
    rerank_applied = (
        artifact_status == "success"
        and bool(query_constraints)
        and has_constraint_coverage
    )
    rerank_skip_reason = None
    if not rerank_applied:
        if artifact_status != "success":
            rerank_skip_reason = f"artifact_status::{artifact_status}"
        elif not query_constraints:
            rerank_skip_reason = "no_runtime_query_constraints"
        else:
            rerank_skip_reason = "no_runtime_constraint_coverage"
    reranked = (
        sorted(
            scored_rows,
            key=lambda row: (
                -row["graph_score"],
                row["rank_before"],
                row["evidence_id"],
            ),
        )
        if rerank_applied
        else sorted(
            scored_rows,
            key=lambda row: (
                row["rank_before"],
                row["evidence_id"],
            ),
        )
    )
    top_rows = reranked[: validated_config["top_k"]]
    for rank_after, row in enumerate(top_rows, 1):
        row["rank_after"] = rank_after
    vector_top_k_evidence_ids = [
        node["id"] for node in vector_ranked[: validated_config["top_k"]]
    ]
    graph_top_k_evidence_ids = [row["evidence_id"] for row in top_rows]
    vector_top_k_set = set(vector_top_k_evidence_ids)
    graph_top_k_set = set(graph_top_k_evidence_ids)

    constraint_audit = _constraint_audit(
        query_constraints,
        [row["runtime_constraints"] for row in top_rows],
    )
    audit_counts = Counter(row["status"] for row in constraint_audit)
    runtime_graph = (
        {"nodes": [], "edges": []}
        if artifact_status == "boundary_refusal"
        else _runtime_constraint_graph(
            sample_id=sample_id,
            question_id=question_node["id"],
            query_constraints=query_constraints,
            ranked_rows=reranked,
        )
    )

    artifact: dict[str, Any] = {
        "artifact_type": "phase6b_reranking_artifact",
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_status": artifact_status,
        "sample_id": sample_id,
        "parent_inference_graph_id": inference_graph["graph_id"],
        "parent_inference_graph_sha256": parent_sha256,
        "method_id": validated_config["method_id"],
        "method_version": validated_config["method_version"],
        "config_version": validated_config["config_version"],
        "versions": dict(inference_graph["versions"]),
        "ruleset_version": RULESET_VERSION,
        "top_k": validated_config["top_k"],
        "score_weights": validated_config["score_weights"],
        "constraint_type_weights": validated_config["constraint_type_weights"],
        "query_constraints": query_constraints,
        "rerank_applied": rerank_applied,
        "rerank_skip_reason": rerank_skip_reason,
        "ranked_evidence": top_rows,
        "ranking_baseline": {
            "vector_top_k_evidence_ids": vector_top_k_evidence_ids,
            "graph_top_k_evidence_ids": graph_top_k_evidence_ids,
            "top1_changed": bool(
                vector_top_k_evidence_ids
                and graph_top_k_evidence_ids
                and vector_top_k_evidence_ids[0] != graph_top_k_evidence_ids[0]
            ),
            "moved_in_evidence_ids": sorted(
                graph_top_k_set - vector_top_k_set
            ),
            "moved_out_evidence_ids": sorted(
                vector_top_k_set - graph_top_k_set
            ),
        },
        "constraint_audit": (
            [] if artifact_status == "boundary_refusal" else constraint_audit
        ),
        "audit_summary": dict(sorted(audit_counts.items())),
        "runtime_constraint_graph": runtime_graph,
        "external_model_calls": 0,
        "estimated_cost": 0,
    }
    assert_no_gold_only_content(artifact)
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return artifact
