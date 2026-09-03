"""Deterministic, Gold-free graph-guided candidate expansion for Phase 7."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

from experiments.phase6_evidence_graph.graph_contract import (
    assert_no_gold_only_content,
)
from experiments.phase6_evidence_graph.runtime_constraint_extractor import (
    RULESET_VERSION,
    extract_runtime_constraints,
)
from experiments.phase7_formal_experiments.runtime_graph_path_router import (
    route_graph_paths,
)


GRAPH_INDEX_VERSION = "phase7-c1c4e2-candidate-graph-v0.1"
MINIMUM_MATCHED_CONSTRAINT_TYPES = 2
_REQUIRED_CANDIDATE_FIELDS = (
    "candidate_key",
    "collection",
    "document_id",
    "content",
    "source_file",
    "page_number",
)


def _require_candidate(row: dict) -> dict:
    if not isinstance(row, dict):
        raise TypeError("candidate row must be a dictionary")
    for field in _REQUIRED_CANDIDATE_FIELDS:
        value = row.get(field)
        if value is None or value == "":
            raise ValueError(f"candidate row is missing {field}")
    if not isinstance(row["page_number"], int) or row["page_number"] < 1:
        raise ValueError("candidate page_number must be a positive integer")
    return row


def _constraint_key(constraint: dict) -> str:
    return (
        f"{constraint['constraint_type']}::"
        f"{constraint['normalized_value']}"
    )


def _page_key(row: dict) -> str:
    return f"{row['source_file']}::p{row['page_number']}"


def build_candidate_graph_index(candidate_rows: list[dict]) -> dict:
    """Build a runtime-only constraint/page/evidence graph from KB candidates."""
    if not isinstance(candidate_rows, list):
        raise TypeError("candidate_rows must be a list")
    assert_no_gold_only_content(candidate_rows)

    rows_by_key: dict[str, dict] = {}
    constraints_by_candidate: dict[str, list[dict]] = {}
    constraint_postings: dict[str, list[str]] = defaultdict(list)
    page_postings: dict[str, list[str]] = defaultdict(list)

    for raw_row in sorted(
        candidate_rows,
        key=lambda item: str(item.get("candidate_key", "")),
    ):
        row = deepcopy(_require_candidate(raw_row))
        candidate_key = str(row["candidate_key"])
        existing = rows_by_key.get(candidate_key)
        if existing is not None:
            if existing != row:
                raise ValueError(f"conflicting duplicate candidate_key: {candidate_key}")
            continue

        constraints = extract_runtime_constraints(row["content"])
        rows_by_key[candidate_key] = row
        constraints_by_candidate[candidate_key] = constraints
        page_postings[_page_key(row)].append(candidate_key)
        for constraint in constraints:
            constraint_postings[_constraint_key(constraint)].append(candidate_key)

    source_pages: dict[str, set[int]] = defaultdict(set)
    for row in rows_by_key.values():
        source_pages[row["source_file"]].add(row["page_number"])

    adjacent_pages: dict[str, list[str]] = {}
    for source_file, pages in sorted(source_pages.items()):
        for page_number in sorted(pages):
            neighbours = [
                f"{source_file}::p{candidate_page}"
                for candidate_page in (page_number - 1, page_number + 1)
                if candidate_page in pages
            ]
            adjacent_pages[f"{source_file}::p{page_number}"] = neighbours

    graph_index = {
        "graph_index_version": GRAPH_INDEX_VERSION,
        "constraint_ruleset_version": RULESET_VERSION,
        "minimum_matched_constraint_types": MINIMUM_MATCHED_CONSTRAINT_TYPES,
        "candidates": {
            key: rows_by_key[key]
            for key in sorted(rows_by_key)
        },
        "candidate_constraints": {
            key: constraints_by_candidate[key]
            for key in sorted(constraints_by_candidate)
        },
        "constraint_postings": {
            key: sorted(set(values))
            for key, values in sorted(constraint_postings.items())
        },
        "page_postings": {
            key: sorted(set(values))
            for key, values in sorted(page_postings.items())
        },
        "adjacent_pages": adjacent_pages,
    }
    assert_no_gold_only_content(graph_index)
    return graph_index


def _unique_baseline_candidates(candidates: list[dict]) -> list[dict]:
    unique: list[dict] = []
    seen: set[str] = set()
    for raw_row in candidates:
        row = deepcopy(_require_candidate(raw_row))
        candidate_key = str(row["candidate_key"])
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        row["candidate_origin"] = "baseline"
        row.pop("graph_path_match_count", None)
        row.pop("graph_path_constraint_types", None)
        row.pop("graph_path_trace", None)
        unique.append(row)
    return unique


def _rank_graph_candidates(question: str, graph_index: dict) -> list[dict]:
    query_constraints = extract_runtime_constraints(question)
    if not query_constraints:
        return []

    matches: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for constraint in query_constraints:
        key = _constraint_key(constraint)
        for candidate_key in graph_index["constraint_postings"].get(key, []):
            matches[candidate_key].add(
                (
                    constraint["constraint_type"],
                    constraint["normalized_value"],
                )
            )

    ranked: list[dict] = []
    minimum_types = graph_index["minimum_matched_constraint_types"]
    for candidate_key, matched_constraints in matches.items():
        matched_types = sorted({item[0] for item in matched_constraints})
        if len(matched_types) < minimum_types:
            continue
        row = deepcopy(graph_index["candidates"][candidate_key])
        row["candidate_origin"] = "graph_expansion"
        row["graph_path_match_count"] = len(matched_constraints)
        row["graph_path_constraint_types"] = matched_types
        ranked.append(row)

    return sorted(
        ranked,
        key=lambda row: (
            -row["graph_path_match_count"],
            row["source_file"],
            row["page_number"],
            row["candidate_key"],
        ),
    )


def expand_candidates(
    question: str,
    baseline_candidates: list[dict],
    graph_index: dict,
    *,
    total_budget: int = 20,
    graph_quota: int = 4,
    runtime_path_catalog: dict | None = None,
    runtime_lexicon: dict | None = None,
    routing_policy: dict | None = None,
) -> list[dict]:
    """Reserve a fixed budget slice for reliable graph-path candidates."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(total_budget, int) or total_budget < 1:
        raise ValueError("total_budget must be a positive integer")
    if not isinstance(graph_quota, int) or not 0 <= graph_quota <= total_budget:
        raise ValueError("graph_quota must be between 0 and total_budget")
    if not isinstance(graph_index, dict):
        raise TypeError("graph_index must be a dictionary")

    assert_no_gold_only_content(baseline_candidates)
    assert_no_gold_only_content(graph_index)
    if graph_index.get("graph_index_version") != GRAPH_INDEX_VERSION:
        raise ValueError("unsupported graph_index_version")

    baseline = _unique_baseline_candidates(baseline_candidates)
    if graph_quota == 0:
        return baseline[:total_budget]

    routing_inputs = (runtime_path_catalog, runtime_lexicon, routing_policy)
    if any(value is not None for value in routing_inputs) and not all(
        value is not None for value in routing_inputs
    ):
        raise ValueError("runtime routing inputs must be provided together")

    baseline_keys = {row["candidate_key"] for row in baseline}
    if runtime_path_catalog is not None:
        assert_no_gold_only_content(runtime_path_catalog)
        assert_no_gold_only_content(runtime_lexicon)
        assert_no_gold_only_content(routing_policy)
        routed = route_graph_paths(
            question,
            catalog=runtime_path_catalog,
            lexicon=runtime_lexicon,
            allow_specific_condition_class_path=bool(
                routing_policy["allow_specific_condition_class_path"]
            ),
            max_total_paths=int(routing_policy["max_total_paths"]),
            max_paths_per_source=int(routing_policy["max_paths_per_source"]),
            max_paths_per_source_page=int(
                routing_policy["max_paths_per_source_page"]
            ),
        )
        ranked_graph_candidates = []
        for raw_row in routed["selected_paths"]:
            row = deepcopy(_require_candidate(raw_row))
            row["candidate_origin"] = "graph_expansion"
            ranked_graph_candidates.append(row)
    else:
        ranked_graph_candidates = _rank_graph_candidates(question, graph_index)
    graph_candidates = [
        row
        for row in ranked_graph_candidates
        if row["candidate_key"] not in baseline_keys
    ]
    if not graph_candidates:
        return baseline[:total_budget]

    selected_graph = graph_candidates[: min(graph_quota, total_budget)]
    baseline_slots = total_budget - len(selected_graph)
    return [*baseline[:baseline_slots], *selected_graph]
