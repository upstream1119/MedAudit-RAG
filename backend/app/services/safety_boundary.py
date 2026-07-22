"""Shared medical safety-boundary rules."""

from __future__ import annotations

import re


PRESCRIPTION_ACTION_TERMS = (
    "开处方",
    "帮我开处方",
    "开药",
    "帮我开药",
    "处方",
    "用药方案",
)

PATIENT_CONTEXT_TERMS = (
    "这个孩子",
    "患儿",
    "宝宝",
    "婴儿",
    "儿童",
    "发热",
    "咳嗽",
)


def is_direct_prescription_request(query: str) -> bool:
    """Return whether a query asks for patient-specific prescription generation."""
    compact_query = re.sub(r"\s+", "", query)
    has_prescription_action = any(
        term in compact_query for term in PRESCRIPTION_ACTION_TERMS
    )
    has_patient_context = any(
        term in compact_query for term in PATIENT_CONTEXT_TERMS
    )
    return has_prescription_action and has_patient_context
