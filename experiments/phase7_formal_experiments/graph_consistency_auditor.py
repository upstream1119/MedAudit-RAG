"""Deterministic Gold-free consistency auditing over frozen G2 evidence."""

from __future__ import annotations

from itertools import combinations

from experiments.phase6_evidence_graph.graph_contract import (
    assert_no_gold_only_content,
)
from experiments.phase7_formal_experiments.runtime_graph_constraint_extractor import (
    extract_graph_runtime_constraints,
)


AUDITOR_VERSION = "phase7-g3-graph-consistency-auditor-v0.1"


def _constraint_map(constraints: list[dict]) -> dict[str, set[str]]:
    mapped: dict[str, set[str]] = {}
    for constraint in constraints:
        constraint_type = str(constraint.get("constraint_type", ""))
        normalized_value = str(constraint.get("normalized_value", ""))
        if constraint_type and normalized_value:
            mapped.setdefault(constraint_type, set()).add(normalized_value)
    return mapped


def _constraint_records(
    constraint_type: str,
    values: set[str],
) -> list[dict[str, str]]:
    return [
        {"constraint_type": constraint_type, "normalized_value": value}
        for value in sorted(values)
    ]


def extract_evidence_constraints(evidence: dict, *, lexicon: dict) -> dict:
    """Extract content and metadata constraints without evaluation annotations."""
    if not isinstance(evidence, dict):
        raise TypeError("evidence must be a dictionary")
    assert_no_gold_only_content(evidence)
    assert_no_gold_only_content(lexicon)
    content = str(evidence.get("content", ""))
    source_file = str(evidence.get("source_file", ""))
    chapter_title = str(evidence.get("chapter_title", ""))
    return {
        "candidate_key": str(evidence.get("candidate_key", "")),
        "source_file": source_file,
        "page_number": int(evidence.get("page_number", 0) or 0),
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


def compare_query_to_evidence(
    query_constraints: list[dict],
    evidence_constraints: list[dict],
    *,
    exclusive_constraint_types: set[str],
    high_risk_coverage_types: set[str],
    compatible_context_types: set[str],
) -> dict:
    """Compare requested runtime constraints with the selected evidence set."""
    query_map = _constraint_map(query_constraints)
    evidence_content_map: dict[str, set[str]] = {}
    evidence_context_map: dict[str, set[str]] = {}
    for evidence in evidence_constraints:
        for constraint_type, values in _constraint_map(
            evidence["content_constraints"]
        ).items():
            evidence_content_map.setdefault(constraint_type, set()).update(values)
        for constraint_type, values in _constraint_map(
            evidence["context_constraints"]
        ).items():
            evidence_context_map.setdefault(constraint_type, set()).update(values)

    matches: list[dict[str, str]] = []
    mismatches: list[dict] = []
    coverage_gaps: list[dict[str, str]] = []
    scope_mismatches: list[dict] = []
    for constraint_type, query_values in sorted(query_map.items()):
        content_values = evidence_content_map.get(constraint_type, set())
        if query_values & content_values:
            matches.extend(
                _constraint_records(constraint_type, query_values & content_values)
            )
        if constraint_type in high_risk_coverage_types and not content_values:
            coverage_gaps.extend(
                _constraint_records(constraint_type, query_values)
            )
        if (
            constraint_type in exclusive_constraint_types
            and content_values
            and len(content_values) == 1
            and query_values.isdisjoint(content_values)
        ):
            mismatches.append(
                {
                    "constraint_type": constraint_type,
                    "query_values": sorted(query_values),
                    "evidence_values": sorted(content_values),
                }
            )

        context_values = evidence_context_map.get(constraint_type, set())
        if (
            constraint_type in compatible_context_types
            and context_values
            and query_values.isdisjoint(context_values)
        ):
            scope_mismatches.append(
                {
                    "constraint_type": constraint_type,
                    "query_values": sorted(query_values),
                    "evidence_values": sorted(context_values),
                }
            )

    labels: set[str] = set()
    if matches:
        labels.add("supported_match")
    if mismatches:
        labels.add("corrective_value_mismatch")
    if coverage_gaps:
        labels.add("coverage_gap")
    if scope_mismatches:
        labels.add("scope_mismatch")
    return {
        "labels": sorted(labels),
        "matched_constraints": matches,
        "corrective_mismatches": mismatches,
        "coverage_gaps": coverage_gaps,
        "scope_mismatches": scope_mismatches,
    }


def compare_evidence_pair(
    left: dict,
    right: dict,
    *,
    exclusive_constraint_types: set[str],
    strong_anchor_types: set[str],
    compatible_context_types: set[str],
    missing_scope_is_not_comparable: bool,
) -> dict:
    """Compare two evidence items only when anchors and scope are compatible."""
    left_content = _constraint_map(left["content_constraints"])
    right_content = _constraint_map(right["content_constraints"])
    left_context = _constraint_map(left["context_constraints"])
    right_context = _constraint_map(right["context_constraints"])
    shared_anchors: list[dict[str, str]] = []
    for constraint_type in sorted(strong_anchor_types):
        shared_anchors.extend(
            _constraint_records(
                constraint_type,
                left_context.get(constraint_type, set())
                & right_context.get(constraint_type, set()),
            )
        )

    shared_scope: list[dict[str, str]] = []
    disjoint_scope: list[dict] = []
    for constraint_type in sorted(compatible_context_types):
        left_values = left_context.get(constraint_type, set())
        right_values = right_context.get(constraint_type, set())
        if not left_values or not right_values:
            continue
        intersection = left_values & right_values
        if intersection:
            shared_scope.extend(_constraint_records(constraint_type, intersection))
        else:
            disjoint_scope.append(
                {
                    "constraint_type": constraint_type,
                    "left_values": sorted(left_values),
                    "right_values": sorted(right_values),
                }
            )

    comparable = bool(shared_anchors) and not disjoint_scope
    if missing_scope_is_not_comparable:
        comparable = comparable and bool(shared_scope)
    conflicts: list[dict] = []
    if comparable:
        for constraint_type in sorted(exclusive_constraint_types):
            left_values = left_content.get(constraint_type, set())
            right_values = right_content.get(constraint_type, set())
            if left_values and right_values and left_values.isdisjoint(right_values):
                conflicts.append(
                    {
                        "constraint_type": constraint_type,
                        "left_values": sorted(left_values),
                        "right_values": sorted(right_values),
                    }
                )
    labels = ["evidence_evidence_conflict"] if conflicts else []
    if not comparable:
        labels = ["not_comparable"]
    return {
        "left_candidate_key": str(left["candidate_key"]),
        "right_candidate_key": str(right["candidate_key"]),
        "comparable": comparable,
        "shared_strong_anchors": shared_anchors,
        "shared_scope": shared_scope,
        "disjoint_scope": disjoint_scope,
        "conflicts": conflicts,
        "labels": labels,
    }


def resolve_route_action(
    summary_labels: list[str],
    *,
    upstream_boundary_refusal: bool,
) -> tuple[str, list[str]]:
    """Apply the preregistered route precedence."""
    labels = set(summary_labels)
    if upstream_boundary_refusal:
        return "boundary_refusal_passthrough", [
            "existing_upstream_boundary_refusal"
        ]
    review_reasons = sorted(labels & {"evidence_evidence_conflict", "scope_mismatch"})
    if review_reasons:
        return "review_required", review_reasons
    if "coverage_gap" in labels:
        return "insufficient_evidence", ["coverage_gap"]
    if "corrective_value_mismatch" in labels:
        return "allow_corrective_answer", ["corrective_value_mismatch"]
    if "supported_match" in labels:
        return "allow_supported_answer", ["supported_match"]
    return "insufficient_evidence", ["no_comparable_evidence"]


def audit_graph_consistency(
    question: str,
    *,
    evidence_top4: list[dict],
    lexicon: dict,
    contract: dict,
    upstream_boundary_refusal: bool = False,
) -> dict:
    """Produce a deterministic annotation-only audit for one question."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty text")
    if not isinstance(evidence_top4, list):
        raise TypeError("evidence_top4 must be a list")
    assert_no_gold_only_content(evidence_top4)
    assert_no_gold_only_content(lexicon)
    query_constraints = extract_graph_runtime_constraints(
        question,
        lexicon=lexicon,
    )
    evidence_constraints = [
        extract_evidence_constraints(evidence, lexicon=lexicon)
        for evidence in evidence_top4
    ]
    exclusive_types = set(contract["exclusive_constraint_types"])
    high_risk_types = set(contract["high_risk_coverage_types"])
    scope_config = contract["scope_compatibility"]
    compatible_context_types = set(scope_config["compatible_context_types"])
    query_comparison = compare_query_to_evidence(
        query_constraints,
        evidence_constraints,
        exclusive_constraint_types=exclusive_types,
        high_risk_coverage_types=high_risk_types,
        compatible_context_types=compatible_context_types,
    )
    pairwise = [
        compare_evidence_pair(
            left,
            right,
            exclusive_constraint_types=exclusive_types,
            strong_anchor_types=set(scope_config["strong_anchor_types"]),
            compatible_context_types=compatible_context_types,
            missing_scope_is_not_comparable=bool(
                scope_config["missing_scope_is_not_comparable"]
            ),
        )
        for left, right in combinations(evidence_constraints, 2)
    ]
    labels = set(query_comparison["labels"])
    for comparison in pairwise:
        labels.update(comparison["labels"])
    if not evidence_constraints:
        labels.add("not_comparable")
    summary_labels = sorted(labels)
    route_action, route_reasons = resolve_route_action(
        summary_labels,
        upstream_boundary_refusal=upstream_boundary_refusal,
    )
    return {
        "auditor_version": AUDITOR_VERSION,
        "query_constraints": query_constraints,
        "evidence_constraints": evidence_constraints,
        "query_evidence_comparison": query_comparison,
        "pairwise_comparisons": pairwise,
        "summary_labels": summary_labels,
        "route_action": route_action,
        "route_reasons": route_reasons,
    }
