"""Phase 7 runtime-only entity constraints for graph candidate expansion."""

from __future__ import annotations

import re
import unicodedata
from copy import deepcopy

from experiments.phase6_evidence_graph.runtime_constraint_extractor import (
    extract_runtime_constraints,
)


RULESET_VERSION = "phase7-runtime-graph-entity-rules-v0.2"
STRONG_ANCHOR_TYPES = {
    "audit_domain",
    "clinical_condition",
    "medication",
    "medication_class",
}
_BROAD_CONTENT_CONSTRAINTS = {
    ("medication_class", "antimicrobial"),
}
_SPECIFIC_ANCHOR_TYPES = {
    "audit_domain",
    "clinical_condition",
    "medication",
}


def _normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("runtime graph constraint input must be text")
    return unicodedata.normalize("NFKC", text).casefold().strip()


def _validate_lexicon(lexicon: dict) -> list[dict]:
    if not isinstance(lexicon, dict):
        raise TypeError("lexicon must be a dictionary")
    if not lexicon.get("lexicon_version"):
        raise ValueError("lexicon_version is required")
    entries = lexicon.get("entries")
    if not isinstance(entries, list):
        raise ValueError("lexicon entries must be a list")

    validated: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise TypeError("lexicon entry must be a dictionary")
        constraint_type = str(raw_entry.get("constraint_type", "")).strip()
        normalized_value = str(raw_entry.get("normalized_value", "")).strip()
        aliases = raw_entry.get("aliases")
        if not constraint_type or not normalized_value:
            raise ValueError("lexicon entry type and value are required")
        if not isinstance(aliases, list) or not aliases:
            raise ValueError("lexicon entry aliases must be a non-empty list")
        key = (constraint_type, normalized_value)
        if key in seen:
            raise ValueError(f"duplicate lexicon entry: {constraint_type}::{normalized_value}")
        seen.add(key)
        validated.append(
            {
                "constraint_type": constraint_type,
                "normalized_value": normalized_value,
                "aliases": sorted(
                    {
                        _normalize_text(alias)
                        for alias in aliases
                        if isinstance(alias, str) and alias.strip()
                    }
                ),
                "strong_anchor": bool(raw_entry.get("strong_anchor", False)),
            }
        )
    return validated


def _contains_alias(text: str, alias: str) -> bool:
    if not alias:
        return False
    if alias.isascii() and re.fullmatch(r"[a-z0-9_-]+", alias):
        return re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text) is not None
    return alias in text


def _deduplicate_constraints(constraints: list[dict]) -> list[dict]:
    merged: dict[tuple[str, str], dict] = {}
    for raw_constraint in constraints:
        constraint = deepcopy(raw_constraint)
        key = (
            str(constraint["constraint_type"]),
            str(constraint["normalized_value"]),
        )
        existing = merged.get(key)
        if existing is None:
            constraint["surface_forms"] = sorted(
                set(constraint.get("surface_forms", []))
            )
            constraint["strong_anchor"] = bool(
                constraint.get("strong_anchor", False)
                or constraint["constraint_type"] in STRONG_ANCHOR_TYPES
            )
            merged[key] = constraint
            continue
        existing["surface_forms"] = sorted(
            set(existing.get("surface_forms", []))
            | set(constraint.get("surface_forms", []))
        )
        existing["strong_anchor"] = bool(
            existing.get("strong_anchor", False)
            or constraint.get("strong_anchor", False)
            or existing["constraint_type"] in STRONG_ANCHOR_TYPES
        )
    return [merged[key] for key in sorted(merged)]


def extract_graph_runtime_constraints(
    *texts: str,
    lexicon: dict,
) -> list[dict]:
    """Extend Phase 6 constraints with deterministic runtime entity aliases."""
    entries = _validate_lexicon(lexicon)
    constraints = extract_runtime_constraints(*texts)
    lexicon_version = str(lexicon["lexicon_version"])

    for raw_text in texts:
        text = _normalize_text(raw_text)
        if not text:
            continue
        for entry in entries:
            matched_aliases = [
                alias
                for alias in entry["aliases"]
                if _contains_alias(text, alias)
            ]
            if not matched_aliases:
                continue
            constraints.append(
                {
                    "constraint_type": entry["constraint_type"],
                    "normalized_value": entry["normalized_value"],
                    "surface_forms": matched_aliases,
                    "ruleset_version": RULESET_VERSION,
                    "lexicon_version": lexicon_version,
                    "strong_anchor": entry["strong_anchor"],
                }
            )
    return _deduplicate_constraints(constraints)


def has_strong_anchor(constraints: list[dict]) -> bool:
    return any(
        bool(constraint.get("strong_anchor", False))
        or constraint.get("constraint_type") in STRONG_ANCHOR_TYPES
        for constraint in constraints
    )


def has_condition_conflict(
    query_constraints: list[dict],
    candidate_constraints: list[dict],
) -> bool:
    """Reject candidates that name only disjoint, explicit clinical conditions."""
    query_conditions = {
        constraint["normalized_value"]
        for constraint in query_constraints
        if constraint.get("constraint_type") == "clinical_condition"
    }
    candidate_conditions = {
        constraint["normalized_value"]
        for constraint in candidate_constraints
        if constraint.get("constraint_type") == "clinical_condition"
    }
    if not query_conditions or not candidate_conditions:
        return False
    return query_conditions.isdisjoint(candidate_conditions)


def _constraint_pairs(constraints: list[dict]) -> set[tuple[str, str]]:
    return {
        (
            str(constraint["constraint_type"]),
            str(constraint["normalized_value"]),
        )
        for constraint in constraints
    }


def assess_constraint_path(
    query_constraints: list[dict],
    candidate_context_constraints: list[dict],
    candidate_content_constraints: list[dict],
    *,
    minimum_matched_constraint_types: int = 2,
    allow_specific_condition_class_path: bool = False,
) -> dict:
    """Qualify a graph path without letting source metadata replace evidence."""
    if minimum_matched_constraint_types < 1:
        raise ValueError("minimum_matched_constraint_types must be positive")

    query_pairs = _constraint_pairs(query_constraints)
    context_pairs = _constraint_pairs(candidate_context_constraints)
    content_pairs = _constraint_pairs(candidate_content_constraints)
    matched_pairs = query_pairs & context_pairs
    matched_types = sorted({pair[0] for pair in matched_pairs})
    content_matched_pairs = matched_pairs & content_pairs
    content_matched_types = sorted({pair[0] for pair in content_matched_pairs})

    result = {
        "qualified": False,
        "reason": "insufficient_types",
        "matched_constraint_count": len(matched_pairs),
        "matched_constraint_types": matched_types,
        "content_matched_constraint_count": len(content_matched_pairs),
        "content_matched_constraint_types": content_matched_types,
    }
    if has_condition_conflict(
        query_constraints,
        candidate_context_constraints,
    ):
        result["reason"] = "condition_conflict"
        return result
    if len(matched_types) < minimum_matched_constraint_types:
        return result

    matched_constraints = [
        constraint
        for constraint in query_constraints
        if (
            str(constraint["constraint_type"]),
            str(constraint["normalized_value"]),
        )
        in matched_pairs
    ]
    if not has_strong_anchor(matched_constraints):
        result["reason"] = "missing_strong_anchor"
        return result
    has_specific_anchor = any(
        pair[0] in _SPECIFIC_ANCHOR_TYPES
        or (
            pair[0] == "medication_class"
            and pair not in _BROAD_CONTENT_CONSTRAINTS
        )
        for pair in matched_pairs
    )
    if not has_specific_anchor:
        result["reason"] = "missing_specific_anchor"
        return result
    if not content_matched_pairs:
        result["reason"] = "no_content_supported_match"
        return result
    if content_matched_pairs <= _BROAD_CONTENT_CONSTRAINTS:
        result["reason"] = "broad_content_only"
        return result
    if set(matched_types) == {"clinical_condition", "medication_class"}:
        matched_class_pairs = {
            pair for pair in matched_pairs if pair[0] == "medication_class"
        }
        if (
            not allow_specific_condition_class_path
            or matched_class_pairs <= _BROAD_CONTENT_CONSTRAINTS
        ):
            result["reason"] = "condition_class_only"
            return result

    result["qualified"] = True
    result["reason"] = "qualified"
    return result
