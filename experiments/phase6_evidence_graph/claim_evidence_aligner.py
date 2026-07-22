"""Deterministic claim-evidence alignment for Phase 6-B4."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from .graph_contract import assert_no_gold_only_content
from .graph_reranker import canonical_sha256
from .runtime_constraint_extractor import (
    RULESET_VERSION as CONSTRAINT_RULESET_VERSION,
)
from .runtime_constraint_extractor import extract_runtime_constraints


ARTIFACT_SCHEMA_VERSION = "phase6b-claim-alignment-artifact-v0.2"
ALIGNMENT_RULESET_VERSION = "phase6b-claim-alignment-rules-v0.2"

_CLAIM_SPLIT_PATTERN = re.compile(r"[。！？!?；;\n]+")
_REJECTION_PATTERN = re.compile(
    r"(?:不足以支持|不可以|不能|不应|不得|不支持|无依据|不推荐|避免|"
    r"禁用|\bdo\s+not\b|\bshould\s+not\b|\bmust\s+not\b|"
    r"\bnot\s+recommended\b)",
    re.IGNORECASE,
)
_EXCLUSIVE_CONSTRAINT_TYPES = frozenset(
    {
        "dose",
        "frequency",
        "route",
        "monitoring_window",
        "monitoring_action",
        "dose_adjustment",
        "contraindication_action",
        "evidence_scope",
    }
)
_OVERALL_STATE_PRECEDENCE = (
    "contradicted",
    "unsupported",
    "insufficient_evidence",
    "supported",
)


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")
    validated = {
        "config_version": _require_text(
            config.get("config_version"), "config_version"
        ),
        "method_id": _require_text(config.get("method_id"), "method_id"),
        "method_version": _require_text(
            config.get("method_version"), "method_version"
        ),
        "candidate_output_origin": str(
            config.get("candidate_output_origin")
            or "unspecified_runtime_output"
        ).strip(),
    }
    evidence_budget = config.get("evidence_budget")
    if not isinstance(evidence_budget, int) or evidence_budget < 1:
        raise ValueError("evidence_budget must be a positive integer")
    validated["evidence_budget"] = evidence_budget
    return validated


def _validate_parent_artifact(artifact: dict[str, Any]) -> None:
    if not isinstance(artifact, dict):
        raise ValueError("reranking artifact must be a dictionary")
    assert_no_gold_only_content(artifact)
    if artifact.get("artifact_type") != "phase6b_reranking_artifact":
        raise ValueError("artifact_type must be phase6b_reranking_artifact")
    if artifact.get("artifact_schema_version") != (
        "phase6b-reranking-artifact-v0.2"
    ):
        raise ValueError("unsupported reranking artifact schema")
    if artifact.get("artifact_status") not in {
        "success",
        "boundary_refusal",
        "insufficient_graph_evidence",
    }:
        raise ValueError("unsupported reranking artifact status")
    _require_text(artifact.get("sample_id"), "sample_id")
    if not isinstance(artifact.get("versions"), dict):
        raise ValueError("versions must be a dictionary")
    if not isinstance(artifact.get("ranked_evidence"), list):
        raise ValueError("ranked_evidence must be a list")


def split_atomic_claims(answer_text: str) -> list[str]:
    """Split a short answer into deterministic sentence-level claims."""
    normalized = unicodedata.normalize("NFKC", _require_text(answer_text, "answer_text"))
    return [
        part.strip(" ，,")
        for part in _CLAIM_SPLIT_PATTERN.split(normalized)
        if part.strip(" ，,")
    ]


def detect_claim_stance(claim_text: str) -> str:
    """Classify an atomic claim as an assertion or explicit rejection."""
    return "reject" if _REJECTION_PATTERN.search(claim_text) else "assert"


def _constraint_stance(text: str, constraint: dict[str, Any]) -> str:
    """Determine polarity near a constraint instead of for the whole row."""
    if constraint["constraint_type"] in {
        "contraindication_action",
        "contraindication_check",
    }:
        # The normalized values already encode the safety action/check.
        return "assert"

    normalized_text = unicodedata.normalize("NFKC", text)
    for surface_form in constraint.get("surface_forms") or []:
        for surface_part in re.split(r"\s*/\s*", surface_form):
            surface_part = surface_part.strip()
            if not surface_part:
                continue
            position = normalized_text.lower().find(surface_part.lower())
            if position < 0:
                continue
            start = max(0, position - 96)
            end = min(
                len(normalized_text), position + len(surface_part) + 64
            )
            return detect_claim_stance(normalized_text[start:end])
    return detect_claim_stance(normalized_text)


def _normalized_constraint_value(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    if normalized.endswith("/day"):
        return normalized[:-4]
    return normalized


def _values_match(first: str, second: str) -> bool:
    return _normalized_constraint_value(first) == _normalized_constraint_value(
        second
    )


def _evidence_constraint_index(
    evidence_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, str]]]:
    indexed: dict[str, list[dict[str, str]]] = {}
    for row in evidence_rows:
        evidence_id = _require_text(row.get("evidence_id"), "evidence_id")
        content = _require_text(row.get("content"), "content")
        parent_constraints = row.get("runtime_constraints")
        if not isinstance(parent_constraints, list):
            raise ValueError("runtime_constraints must be a list")

        merged_constraints: dict[tuple[str, str], dict[str, Any]] = {}
        for constraint in [
            *parent_constraints,
            *extract_runtime_constraints(content),
        ]:
            if not isinstance(constraint, dict):
                raise ValueError("runtime constraint must be a dictionary")
            constraint_type = _require_text(
                constraint.get("constraint_type"), "constraint_type"
            )
            normalized_value = _require_text(
                constraint.get("normalized_value"), "normalized_value"
            )
            key = (constraint_type, normalized_value)
            merged = merged_constraints.setdefault(
                key,
                {
                    "constraint_type": constraint_type,
                    "normalized_value": normalized_value,
                    "surface_forms": [],
                },
            )
            merged["surface_forms"] = sorted(
                {
                    *merged["surface_forms"],
                    *(constraint.get("surface_forms") or []),
                }
            )

        for constraint in merged_constraints.values():
            constraint_type = constraint["constraint_type"]
            indexed.setdefault(constraint_type, []).append(
                {
                    "normalized_value": constraint["normalized_value"],
                    "evidence_id": evidence_id,
                    "stance": _constraint_stance(content, constraint),
                }
            )
    return indexed


def _source_aliases(source_file: str) -> list[str]:
    stem = unicodedata.normalize("NFKC", Path(source_file).stem)
    without_version = re.sub(
        r"[（(]\s*\d{4}\s*年?版?\s*[）)]", "", stem
    )
    aliases = {
        re.sub(r"\s+", "", stem),
        re.sub(r"\s+", "", without_version),
    }
    return sorted(alias for alias in aliases if len(alias) >= 4)


def _bind_evidence_to_named_source(
    claim_text: str,
    evidence_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized_claim = re.sub(
        r"\s+", "", unicodedata.normalize("NFKC", claim_text)
    )
    matched_source_ids: set[str] = set()
    matched_aliases: set[str] = set()
    for row in evidence_rows:
        source_id = _require_text(row.get("source_id"), "source_id")
        source_file = _require_text(row.get("source_file"), "source_file")
        for alias in _source_aliases(source_file):
            if alias in normalized_claim:
                matched_source_ids.add(source_id)
                matched_aliases.add(alias)

    if matched_source_ids:
        admitted_rows = [
            row
            for row in evidence_rows
            if row.get("source_id") in matched_source_ids
        ]
        binding_status = "bound"
    else:
        admitted_rows = evidence_rows
        binding_status = "not_explicitly_named"

    return admitted_rows, {
        "binding_status": binding_status,
        "matched_aliases": sorted(matched_aliases),
        "admitted_source_ids": sorted(
            {
                _require_text(row.get("source_id"), "source_id")
                for row in admitted_rows
            }
        ),
    }


def _evaluate_constraint(
    constraint: dict[str, Any],
    evidence_index: dict[str, list[dict[str, str]]],
    claim_stance: str,
) -> dict[str, Any]:
    constraint_type = constraint["constraint_type"]
    claim_value = constraint["normalized_value"]
    candidates = evidence_index.get(constraint_type, [])
    matching_ids = sorted(
        {
            candidate["evidence_id"]
            for candidate in candidates
            if _values_match(claim_value, candidate["normalized_value"])
            and candidate["stance"] == claim_stance
        }
    )
    opposite_matching_ids = sorted(
        {
            candidate["evidence_id"]
            for candidate in candidates
            if _values_match(claim_value, candidate["normalized_value"])
            and candidate["stance"] != claim_stance
        }
    )
    alternative_ids = sorted(
        {
            candidate["evidence_id"]
            for candidate in candidates
            if not _values_match(claim_value, candidate["normalized_value"])
            and candidate["stance"] == "assert"
        }
    )

    if matching_ids:
        state = "matched"
        evidence_ids = matching_ids
    elif opposite_matching_ids:
        state = "conflict"
        evidence_ids = opposite_matching_ids
    elif claim_stance == "assert":
        if constraint_type == "evidence_scope" and alternative_ids:
            state = "scope_mismatch"
            evidence_ids = alternative_ids
        elif constraint_type in _EXCLUSIVE_CONSTRAINT_TYPES and alternative_ids:
            state = "conflict"
            evidence_ids = alternative_ids
        else:
            state = "unsupported"
            evidence_ids = []
    else:
        if constraint_type in _EXCLUSIVE_CONSTRAINT_TYPES and alternative_ids:
            state = "matched"
            evidence_ids = alternative_ids
        else:
            state = "unsupported"
            evidence_ids = []

    return {
        "constraint_type": constraint_type,
        "claim_value": claim_value,
        "claim_stance": claim_stance,
        "state": state,
        "evidence_ids": evidence_ids,
    }


def _align_atomic_claim(
    *,
    sample_id: str,
    claim_index: int,
    claim_text: str,
    evidence_rows: list[dict[str, Any]],
    parent_has_evidence: bool,
) -> dict[str, Any]:
    stance = detect_claim_stance(claim_text)
    constraints = extract_runtime_constraints(claim_text)
    claim_id = f"claim::{sample_id}::runtime::{claim_index}"
    bound_evidence_rows, source_binding = _bind_evidence_to_named_source(
        claim_text, evidence_rows
    )
    evidence_index = _evidence_constraint_index(bound_evidence_rows)

    if not parent_has_evidence:
        return {
            "claim_id": claim_id,
            "claim_text": claim_text,
            "stance": stance,
            "support_state": "insufficient_evidence",
            "runtime_constraints": constraints,
            "supporting_evidence_ids": [],
            "contradicting_evidence_ids": [],
            "reason_codes": ["parent_has_no_admitted_evidence"],
            "constraint_checks": [],
            "source_binding": source_binding,
        }
    if not constraints:
        return {
            "claim_id": claim_id,
            "claim_text": claim_text,
            "stance": stance,
            "support_state": "insufficient_evidence",
            "runtime_constraints": [],
            "supporting_evidence_ids": [],
            "contradicting_evidence_ids": [],
            "reason_codes": ["no_auditable_constraint_extracted"],
            "constraint_checks": [],
            "source_binding": source_binding,
        }

    checks = [
        _evaluate_constraint(
            constraint,
            evidence_index,
            _constraint_stance(claim_text, constraint),
        )
        for constraint in constraints
    ]
    check_states = {check["state"] for check in checks}
    supporting_ids = sorted(
        {
            evidence_id
            for check in checks
            if check["state"] == "matched"
            for evidence_id in check["evidence_ids"]
        }
    )
    contradicting_ids = sorted(
        {
            evidence_id
            for check in checks
            if check["state"] == "conflict"
            for evidence_id in check["evidence_ids"]
        }
    )

    if stance == "reject" and len(checks) > 1 and "unsupported" in check_states:
        support_state = "unsupported"
        reason_codes = ["compound_rejection_not_fully_evidenced"]
    elif "conflict" in check_states:
        support_state = "contradicted"
        reason_codes = ["exclusive_constraint_conflict"]
    elif "scope_mismatch" in check_states:
        support_state = "unsupported"
        reason_codes = ["evidence_scope_mismatch"]
    elif "unsupported" in check_states:
        support_state = "unsupported"
        reason_codes = ["no_supporting_constraint_path"]
    else:
        support_state = "supported"
        reason_codes = ["all_runtime_constraints_supported"]

    return {
        "claim_id": claim_id,
        "claim_text": claim_text,
        "stance": stance,
        "support_state": support_state,
        "runtime_constraints": constraints,
        "supporting_evidence_ids": supporting_ids,
        "contradicting_evidence_ids": contradicting_ids,
        "reason_codes": reason_codes,
        "constraint_checks": checks,
        "source_binding": source_binding,
    }


def _overall_support_state(claims: list[dict[str, Any]]) -> str:
    states = {claim["support_state"] for claim in claims}
    for state in _OVERALL_STATE_PRECEDENCE:
        if state in states:
            return state
    return "insufficient_evidence"


def _claim_summary(claims: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(claim["support_state"] for claim in claims)
    requires_human_review = any(
        state in {"contradicted", "unsupported", "insufficient_evidence"}
        for state in counts
    )
    return {
        "claim_count": len(claims),
        "supported": counts.get("supported", 0),
        "contradicted": counts.get("contradicted", 0),
        "unsupported": counts.get("unsupported", 0),
        "insufficient_evidence": counts.get("insufficient_evidence", 0),
        "requires_human_review": requires_human_review,
    }


def _claim_evidence_paths(claims: list[dict[str, Any]]) -> list[dict[str, str]]:
    paths: list[dict[str, str]] = []
    for claim in claims:
        for evidence_id in claim["supporting_evidence_ids"]:
            paths.append(
                {
                    "source": evidence_id,
                    "target": claim["claim_id"],
                    "relation_type": "EVIDENCE_SUPPORTS_RUNTIME_CLAIM",
                }
            )
        for evidence_id in claim["contradicting_evidence_ids"]:
            paths.append(
                {
                    "source": evidence_id,
                    "target": claim["claim_id"],
                    "relation_type": "EVIDENCE_CONTRADICTS_RUNTIME_CLAIM",
                }
            )
    return paths


def align_claims(
    reranking_artifact: dict[str, Any],
    answer_text: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Align answer claims against admitted runtime evidence only."""
    _validate_parent_artifact(reranking_artifact)
    validated_config = _validate_config(config)
    answer = _require_text(answer_text, "answer_text")
    sample_id = reranking_artifact["sample_id"]
    parent_status = reranking_artifact["artifact_status"]
    evidence_rows = reranking_artifact["ranked_evidence"][
        : validated_config["evidence_budget"]
    ]

    if parent_status == "boundary_refusal":
        claims: list[dict[str, Any]] = []
        overall_state = "not_applicable"
    else:
        claims = [
            _align_atomic_claim(
                sample_id=sample_id,
                claim_index=index,
                claim_text=claim_text,
                evidence_rows=evidence_rows,
                parent_has_evidence=bool(evidence_rows),
            )
            for index, claim_text in enumerate(split_atomic_claims(answer), start=1)
        ]
        overall_state = _overall_support_state(claims)

    summary = _claim_summary(claims)
    support_coverage = (
        summary["supported"] / summary["claim_count"]
        if summary["claim_count"]
        else 0.0
    )
    result = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "phase6b_claim_alignment_artifact",
        "artifact_status": parent_status,
        "sample_id": sample_id,
        "parent_reranking_artifact_sha256": canonical_sha256(
            reranking_artifact
        ),
        "parent_inference_graph_id": reranking_artifact.get(
            "parent_inference_graph_id"
        ),
        "parent_inference_graph_sha256": reranking_artifact.get(
            "parent_inference_graph_sha256"
        ),
        "versions": dict(reranking_artifact["versions"]),
        "method_id": validated_config["method_id"],
        "method_version": validated_config["method_version"],
        "config_version": validated_config["config_version"],
        "alignment_ruleset_version": ALIGNMENT_RULESET_VERSION,
        "constraint_ruleset_version": CONSTRAINT_RULESET_VERSION,
        "parent_constraint_ruleset_version": reranking_artifact.get(
            "ruleset_version", "unknown"
        ),
        "evidence_constraint_extraction_mode": (
            "parent_plus_runtime_reextraction"
        ),
        "candidate_output_origin": validated_config[
            "candidate_output_origin"
        ],
        "answer_text": answer,
        "evidence_budget": validated_config["evidence_budget"],
        "admitted_evidence_ids": [
            row["evidence_id"] for row in evidence_rows
        ],
        "overall_support_state": overall_state,
        "claim_support_coverage": support_coverage,
        "claim_summary": summary,
        "claims": claims,
        "claim_evidence_paths": _claim_evidence_paths(claims),
        "external_model_calls": 0,
        "estimated_cost": 0,
    }
    assert_no_gold_only_content(result)
    result["artifact_sha256"] = canonical_sha256(result)
    return result
