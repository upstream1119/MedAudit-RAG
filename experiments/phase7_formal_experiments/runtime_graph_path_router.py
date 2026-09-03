"""Gold-free source-aware routing for Phase 7 graph candidate paths."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

from experiments.phase6_evidence_graph.graph_contract import (
    assert_no_gold_only_content,
)
from experiments.phase7_formal_experiments.runtime_graph_constraint_extractor import (
    assess_constraint_path,
    extract_graph_runtime_constraints,
)


ROUTER_VERSION = "phase7-runtime-graph-path-router-v0.1"
PATH_TRACE_VERSION = "phase7-runtime-graph-path-trace-v0.1"


def _constraint_pairs(constraints: list[dict]) -> set[tuple[str, str]]:
    return {
        (
            str(constraint["constraint_type"]),
            str(constraint["normalized_value"]),
        )
        for constraint in constraints
    }


def _sorted_constraint_records(
    pairs: set[tuple[str, str]],
) -> list[dict[str, str]]:
    return [
        {
            "constraint_type": constraint_type,
            "normalized_value": normalized_value,
        }
        for constraint_type, normalized_value in sorted(pairs)
    ]


def build_runtime_path_catalog(graph_index: dict, lexicon: dict) -> dict:
    """Precompute runtime-safe constraints for every graph candidate."""
    if not isinstance(graph_index, dict) or not isinstance(lexicon, dict):
        raise TypeError("graph_index and lexicon must be dictionaries")
    assert_no_gold_only_content(graph_index)
    assert_no_gold_only_content(lexicon)
    raw_candidates = graph_index.get("candidates")
    if not isinstance(raw_candidates, dict):
        raise ValueError("graph index candidates must be a dictionary")

    candidates: dict[str, dict] = {}
    for raw_key, raw_row in sorted(raw_candidates.items()):
        if not isinstance(raw_row, dict):
            raise TypeError("graph candidate must be a dictionary")
        key = str(raw_key)
        row = deepcopy(raw_row)
        row["candidate_key"] = str(row.get("candidate_key", key))
        content = str(row.get("content", ""))
        source_file = str(row.get("source_file", ""))
        chapter_title = str(row.get("chapter_title", ""))
        candidates[key] = {
            "candidate": row,
            "content_constraints": extract_graph_runtime_constraints(
                content,
                lexicon=lexicon,
            ),
            "source_constraints": extract_graph_runtime_constraints(
                source_file,
                chapter_title,
                lexicon=lexicon,
            ),
            "context_constraints": extract_graph_runtime_constraints(
                content,
                source_file,
                chapter_title,
                lexicon=lexicon,
            ),
        }
    return {
        "router_version": ROUTER_VERSION,
        "runtime_safe": True,
        "graph_index_version": str(graph_index.get("graph_index_version", "")),
        "lexicon_version": str(lexicon.get("lexicon_version", "")),
        "candidates": candidates,
    }


def _source_condition_tier(
    query_pairs: set[tuple[str, str]],
    content_pairs: set[tuple[str, str]],
    source_pairs: set[tuple[str, str]],
) -> tuple[int, str]:
    query_conditions = {pair for pair in query_pairs if pair[0] == "clinical_condition"}
    content_conditions = {
        pair for pair in content_pairs if pair[0] == "clinical_condition"
    }
    source_conditions = {
        pair for pair in source_pairs if pair[0] == "clinical_condition"
    }
    if not query_conditions:
        return 3, "query_without_condition"
    if query_conditions & source_conditions:
        return 0, "source_condition_exact"
    if query_conditions & content_conditions and not source_conditions:
        return 1, "content_condition_generic_source"
    if query_conditions & content_conditions:
        return 2, "content_condition_other_source"
    return 4, "context_only_condition"


def route_graph_paths(
    question: str,
    *,
    catalog: dict,
    lexicon: dict,
    allow_specific_condition_class_path: bool,
    max_total_paths: int,
    max_paths_per_source: int,
    max_paths_per_source_page: int,
) -> dict:
    """Rank qualified paths by source specificity before budget enforcement."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty text")
    for name, value in {
        "max_total_paths": max_total_paths,
        "max_paths_per_source": max_paths_per_source,
        "max_paths_per_source_page": max_paths_per_source_page,
    }.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if catalog.get("router_version") != ROUTER_VERSION or not catalog.get(
        "runtime_safe"
    ):
        raise ValueError("catalog must be built by the runtime-safe path router")
    assert_no_gold_only_content(lexicon)
    query_constraints = extract_graph_runtime_constraints(question, lexicon=lexicon)
    query_pairs = _constraint_pairs(query_constraints)

    qualified: list[dict] = []
    for key, entry in sorted(catalog.get("candidates", {}).items()):
        assessment = assess_constraint_path(
            query_constraints,
            entry["context_constraints"],
            entry["content_constraints"],
            minimum_matched_constraint_types=2,
            allow_specific_condition_class_path=(
                allow_specific_condition_class_path
            ),
        )
        if not assessment["qualified"]:
            continue
        content_pairs = _constraint_pairs(entry["content_constraints"])
        source_pairs = _constraint_pairs(entry["source_constraints"])
        context_pairs = _constraint_pairs(entry["context_constraints"])
        matched_pairs = query_pairs & context_pairs
        tier, tier_label = _source_condition_tier(
            query_pairs,
            content_pairs,
            source_pairs,
        )
        candidate = deepcopy(entry["candidate"])
        candidate.update(
            {
                "candidate_key": str(candidate.get("candidate_key", key)),
                "graph_source_condition_tier": tier,
                "graph_source_condition_tier_label": tier_label,
                "graph_path_match_count": assessment[
                    "matched_constraint_count"
                ],
                "graph_path_constraint_types": assessment[
                    "matched_constraint_types"
                ],
                "graph_content_match_count": assessment[
                    "content_matched_constraint_count"
                ],
                "graph_content_match_types": assessment[
                    "content_matched_constraint_types"
                ],
                "_graph_path_trace_input": {
                    "query_constraints": _sorted_constraint_records(query_pairs),
                    "matched_constraints": _sorted_constraint_records(
                        matched_pairs
                    ),
                    "content_matched_constraints": _sorted_constraint_records(
                        matched_pairs & content_pairs
                    ),
                    "source_matched_constraints": _sorted_constraint_records(
                        matched_pairs & source_pairs
                    ),
                },
            }
        )
        qualified.append(candidate)

    qualified.sort(
        key=lambda row: (
            int(row["graph_source_condition_tier"]),
            -len(row["graph_content_match_types"]),
            -int(row["graph_content_match_count"]),
            -len(row["graph_path_constraint_types"]),
            -int(row["graph_path_match_count"]),
            str(row.get("source_file", "")),
            int(row.get("page_number", 0) or 0),
            str(row["candidate_key"]),
        )
    )
    selected: list[dict] = []
    source_counts: Counter[str] = Counter()
    source_page_counts: Counter[tuple[str, int]] = Counter()
    drop_reason_counts: Counter[str] = Counter()
    path_audit: list[dict] = []
    for raw_rank, row in enumerate(qualified, start=1):
        source = str(row.get("source_file", ""))
        page = int(row.get("page_number", 0) or 0)
        source_page = (source, page)
        if source_page_counts[source_page] >= max_paths_per_source_page:
            drop_reason = "source_page_quota"
        elif source_counts[source] >= max_paths_per_source:
            drop_reason = "source_quota"
        elif len(selected) >= max_total_paths:
            drop_reason = "total_quota"
        else:
            drop_reason = "selected"
            selected_row = deepcopy(row)
            route_rank = len(selected) + 1
            selected_row["graph_route_rank"] = route_rank
            trace_input = selected_row.pop("_graph_path_trace_input")
            selected_row["graph_path_trace"] = {
                "trace_version": PATH_TRACE_VERSION,
                "router_version": ROUTER_VERSION,
                "route_decision": "selected",
                "raw_rank": raw_rank,
                "route_rank": route_rank,
                "source_condition_tier": int(
                    selected_row["graph_source_condition_tier"]
                ),
                "source_condition_tier_label": str(
                    selected_row["graph_source_condition_tier_label"]
                ),
                **trace_input,
                "candidate": {
                    "candidate_key": str(selected_row["candidate_key"]),
                    "source_file": source,
                    "page_number": page,
                },
            }
            assert_no_gold_only_content(selected_row["graph_path_trace"])
            selected.append(selected_row)
            source_counts[source] += 1
            source_page_counts[source_page] += 1
        if drop_reason != "selected":
            drop_reason_counts[drop_reason] += 1
        path_audit.append(
            {
                "candidate_key": str(row["candidate_key"]),
                "source_file": source,
                "page_number": page,
                "graph_raw_rank": raw_rank,
                "graph_source_condition_tier": int(
                    row["graph_source_condition_tier"]
                ),
                "graph_route_drop_reason": drop_reason,
            }
        )
    return {
        "router_version": ROUTER_VERSION,
        "query_constraints": query_constraints,
        "raw_path_count": len(qualified),
        "selected_path_count": len(selected),
        "selected_paths": selected,
        "drop_reason_counts": dict(sorted(drop_reason_counts.items())),
        "path_audit": path_audit,
    }
