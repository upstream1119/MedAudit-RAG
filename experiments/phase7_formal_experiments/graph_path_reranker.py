"""Symmetric, Gold-free graph-path reranking for frozen G1 candidates."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

from experiments.phase6_evidence_graph.graph_contract import (
    assert_no_gold_only_content,
)
from experiments.phase7_formal_experiments import runtime_graph_path_router as router
from experiments.phase7_formal_experiments.runtime_graph_constraint_extractor import (
    assess_constraint_path,
    extract_graph_runtime_constraints,
)


RERANKER_VERSION = "phase7-g2-symmetric-graph-path-reranker-v0.1"
TRACE_VERSION = "phase7-g2-symmetric-graph-path-trace-v0.1"
_TRACE_LIST_FIELDS = (
    "query_constraints",
    "matched_constraints",
    "content_matched_constraints",
    "source_matched_constraints",
)


def _candidate_identity(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_key": str(candidate.get("candidate_key", "")),
        "source_file": str(candidate.get("source_file", "")),
        "page_number": int(candidate.get("page_number", 0) or 0),
    }


def _selected_path_map(selected_paths: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not isinstance(selected_paths, list):
        raise TypeError("selected_paths must be a list")
    mapping: dict[str, dict[str, Any]] = {}
    for raw_path in selected_paths:
        if not isinstance(raw_path, dict):
            raise TypeError("selected path must be a dictionary")
        key = str(raw_path.get("candidate_key", ""))
        if not key or key in mapping:
            raise ValueError("selected path candidate keys must be unique and non-empty")
        mapping[key] = raw_path
    return mapping


def _build_symmetric_trace(
    *,
    question: str,
    candidate: dict[str, Any],
    route_record: dict[str, Any],
    graph_index: dict[str, Any],
    lexicon: dict[str, Any],
    allow_specific_condition_class_path: bool,
) -> dict[str, Any]:
    candidate_key = str(candidate.get("candidate_key", ""))
    graph_candidates = graph_index.get("candidates")
    if not isinstance(graph_candidates, dict) or candidate_key not in graph_candidates:
        raise ValueError("path-eligible candidate is missing from graph index")
    graph_candidate = graph_candidates[candidate_key]
    if not isinstance(graph_candidate, dict):
        raise TypeError("graph index candidate must be a dictionary")
    if _candidate_identity(candidate) != _candidate_identity(graph_candidate):
        raise ValueError("candidate identity does not match graph index")
    if str(candidate.get("content", "")) != str(graph_candidate.get("content", "")):
        raise ValueError("candidate content does not match graph index")
    if _candidate_identity(candidate) != {
        "candidate_key": str(route_record.get("candidate_key", "")),
        "source_file": str(route_record.get("source_file", "")),
        "page_number": int(route_record.get("page_number", 0) or 0),
    }:
        raise ValueError("candidate identity does not match routed path")

    query_constraints = extract_graph_runtime_constraints(question, lexicon=lexicon)
    content_constraints = extract_graph_runtime_constraints(
        str(candidate.get("content", "")), lexicon=lexicon
    )
    source_constraints = extract_graph_runtime_constraints(
        str(candidate.get("source_file", "")),
        str(candidate.get("chapter_title", "")),
        lexicon=lexicon,
    )
    context_constraints = extract_graph_runtime_constraints(
        str(candidate.get("content", "")),
        str(candidate.get("source_file", "")),
        str(candidate.get("chapter_title", "")),
        lexicon=lexicon,
    )
    assessment = assess_constraint_path(
        query_constraints,
        context_constraints,
        content_constraints,
        minimum_matched_constraint_types=2,
        allow_specific_condition_class_path=allow_specific_condition_class_path,
    )
    if not assessment["qualified"]:
        raise ValueError("routed path no longer satisfies graph path constraints")

    query_pairs = router._constraint_pairs(query_constraints)
    content_pairs = router._constraint_pairs(content_constraints)
    source_pairs = router._constraint_pairs(source_constraints)
    context_pairs = router._constraint_pairs(context_constraints)
    matched_pairs = query_pairs & context_pairs
    tier, tier_label = router._source_condition_tier(
        query_pairs, content_pairs, source_pairs
    )
    expected_types = sorted(str(value) for value in assessment["matched_constraint_types"])
    routed_types = sorted(
        str(value) for value in route_record.get("graph_path_constraint_types", [])
    )
    if expected_types != routed_types:
        raise ValueError("routed path constraint types do not match runtime evidence")
    if tier != int(route_record.get("graph_source_condition_tier", -1)):
        raise ValueError("routed path source tier does not match runtime evidence")
    if tier_label != str(route_record.get("graph_source_condition_tier_label", "")):
        raise ValueError("routed path source tier label does not match runtime evidence")

    trace = {
        "trace_version": TRACE_VERSION,
        "router_version": router.ROUTER_VERSION,
        "route_decision": "selected",
        "route_rank": int(route_record.get("graph_route_rank", 0) or 0),
        "source_condition_tier": tier,
        "source_condition_tier_label": tier_label,
        "query_constraints": router._sorted_constraint_records(query_pairs),
        "matched_constraints": router._sorted_constraint_records(matched_pairs),
        "content_matched_constraints": router._sorted_constraint_records(
            matched_pairs & content_pairs
        ),
        "source_matched_constraints": router._sorted_constraint_records(
            matched_pairs & source_pairs
        ),
        "candidate": _candidate_identity(candidate),
    }
    if trace["route_rank"] < 1:
        raise ValueError("routed path rank must be positive")
    assert_no_gold_only_content(trace)
    return trace


def _validate_existing_trace(candidate: dict[str, Any], rebuilt: dict[str, Any]) -> bool:
    existing = candidate.get("graph_path_trace")
    if existing is None:
        return False
    if not isinstance(existing, dict):
        raise ValueError("existing graph path trace must be a dictionary")
    expected = {
        "router_version": rebuilt["router_version"],
        "route_decision": rebuilt["route_decision"],
        "route_rank": rebuilt["route_rank"],
        "source_condition_tier": rebuilt["source_condition_tier"],
        "source_condition_tier_label": rebuilt["source_condition_tier_label"],
        "candidate": rebuilt["candidate"],
        **{field: rebuilt[field] for field in _TRACE_LIST_FIELDS},
    }
    observed = {key: existing.get(key) for key in expected}
    if observed != expected:
        raise ValueError("existing graph path trace does not match symmetric rebuild")
    return True


def _path_score(
    trace: dict[str, Any], tier_weights: dict[str, float]
) -> tuple[float, dict[str, float]]:
    query_types = {
        str(record["constraint_type"]) for record in trace["query_constraints"]
    }
    if not query_types:
        raise ValueError("path trace query constraints are empty")
    matched_types = {
        str(record["constraint_type"]) for record in trace["matched_constraints"]
    }
    content_types = {
        str(record["constraint_type"])
        for record in trace["content_matched_constraints"]
    }
    tier_label = str(trace["source_condition_tier_label"])
    if tier_label not in tier_weights:
        raise ValueError(f"missing source tier weight: {tier_label}")
    source_specificity = float(tier_weights[tier_label])
    if not 0.0 <= source_specificity <= 1.0:
        raise ValueError("source tier weights must be within [0, 1]")
    route_rank = int(trace["route_rank"])
    components = {
        "source_specificity": source_specificity,
        "matched_type_coverage": len(matched_types) / len(query_types),
        "content_type_coverage": len(content_types) / len(query_types),
        "route_reciprocal": 1.0 / route_rank,
    }
    components = {
        key: round(min(1.0, max(0.0, value)), 12)
        for key, value in components.items()
    }
    return round(sum(components.values()) / len(components), 12), components


def graph_rerank_candidates(
    *,
    question: str,
    candidates: list[dict[str, Any]],
    selected_paths: list[dict[str, Any]],
    graph_index: dict[str, Any],
    lexicon: dict[str, Any],
    tier_weights: dict[str, float],
    max_rank_shift: float,
    allow_specific_condition_class_path: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rerank one frozen G1 pool without changing its candidate identities."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty text")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a non-empty list")
    if max_rank_shift < 0.0:
        raise ValueError("max_rank_shift must be non-negative")
    assert_no_gold_only_content(candidates)
    assert_no_gold_only_content(selected_paths)
    assert_no_gold_only_content(graph_index)
    assert_no_gold_only_content(lexicon)

    routed = _selected_path_map(selected_paths)
    original = deepcopy(candidates)
    original_keys = [str(row.get("candidate_key", "")) for row in original]
    if not all(original_keys) or len(set(original_keys)) != len(original_keys):
        raise ValueError("candidate keys must be unique and non-empty")
    original_ranks = [int(row.get("post_rerank_rank", 0) or 0) for row in original]
    if original_ranks != list(range(1, len(original) + 1)):
        raise ValueError("candidates must follow contiguous Cross-Encoder ranks")

    enriched: list[dict[str, Any]] = []
    origin_counts: Counter[str] = Counter()
    existing_trace_count = 0
    for candidate in original:
        row = deepcopy(candidate)
        key = str(row["candidate_key"])
        route_record = routed.get(key)
        trace = None
        score = 0.0
        components = {
            "source_specificity": 0.0,
            "matched_type_coverage": 0.0,
            "content_type_coverage": 0.0,
            "route_reciprocal": 0.0,
        }
        if route_record is not None:
            trace = _build_symmetric_trace(
                question=question,
                candidate=row,
                route_record=route_record,
                graph_index=graph_index,
                lexicon=lexicon,
                allow_specific_condition_class_path=allow_specific_condition_class_path,
            )
            existing_trace_count += int(_validate_existing_trace(row, trace))
            score, components = _path_score(trace, tier_weights)
            origin_counts[str(row.get("candidate_origin", "missing"))] += 1
        before_rank = int(row["post_rerank_rank"])
        row.update(
            {
                "g2_path_eligible": trace is not None,
                "g2_graph_path_trace": trace,
                "graph_path_score": score,
                "graph_path_score_components": components,
                "reranker_rank_before_graph": before_rank,
                "graph_adjusted_rank_value": round(
                    before_rank - max_rank_shift * score, 12
                ),
            }
        )
        enriched.append(row)

    enriched.sort(
        key=lambda row: (
            float(row["graph_adjusted_rank_value"]),
            int(row["reranker_rank_before_graph"]),
            str(row["candidate_key"]),
        )
    )
    for rank, row in enumerate(enriched, start=1):
        row["graph_rerank_rank"] = rank
    if {str(row["candidate_key"]) for row in enriched} != set(original_keys):
        raise RuntimeError("graph reranking changed the candidate set")

    audit = {
        "reranker_version": RERANKER_VERSION,
        "trace_version": TRACE_VERSION,
        "candidate_count": len(enriched),
        "selected_path_count": len(routed),
        "path_eligible_count": sum(bool(row["g2_path_eligible"]) for row in enriched),
        "existing_trace_verified_count": existing_trace_count,
        "path_eligible_origin_counts": dict(sorted(origin_counts.items())),
        "candidate_set_preserved": True,
        "max_rank_shift": float(max_rank_shift),
        "candidate_origin_used_for_score": False,
    }
    return enriched, audit
