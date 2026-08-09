import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "benchmark_anchor_review.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "Benchmark anchor review module is not implemented"
    spec = importlib.util.spec_from_file_location(
        "benchmark_anchor_review",
        MODULE_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _coverage(source_id: str = "SRC-101") -> dict:
    return {
        "source_id": source_id,
        "title": f"测试指南 {source_id}",
        "filename": f"{source_id}.pdf",
        "actual_sha256": f"sha256-{source_id}",
        "evidence_types": ["pediatric_dosing", "monitoring"],
        "can_support": ["儿科剂量与监测原则"],
        "cannot_support": ["替代患者个体化处方"],
    }


def _candidate(
    candidate_id: str,
    *,
    source_id: str = "SRC-101",
    page_number: int = 3,
    granularity: int = 512,
    topics: list[str] | None = None,
    text: str | None = None,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "source_id": source_id,
        "source_title": f"测试指南 {source_id}",
        "source_filename": f"{source_id}.pdf",
        "source_sha256": f"sha256-{source_id}",
        "page_number": page_number,
        "block_type": "text",
        "granularity": granularity,
        "raw_text": text
        or f"第{page_number}页：儿童用药后应依据病情变化进行复核，并记录剂量、频次与监测结果。",
        "matched_topics": topics or ["monitoring"],
        "matched_terms": {"monitoring": ["复核"]},
        "review_status": "candidate_unverified",
        "parser_version": "parser-v1",
        "chunker_version": "chunker-v1",
        "config_version": "candidate-v1",
    }


def _config() -> dict:
    return {
        "config_version": "benchmark-anchor-review-v0.1",
        "expected_source_count": 1,
        "max_pending_per_source": 5,
        "preferred_granularities": [512, 128, 1024],
        "min_verified_text_chars": 10,
        "min_alnum_ratio": 0.25,
        "scope_limited_evidence_types": [
            "medicine_identity",
            "dosage_form",
            "essential_medicine_status",
        ],
    }


def _accepted_review(candidate: dict) -> dict:
    return {
        **candidate,
        "candidate_text": candidate["raw_text"],
        "context_text": candidate["raw_text"],
        "review_status": "pending_author_review",
        "reviewer_id": "author-01",
        "author_reviewed_at": "2026-07-27",
        "author_review_outcome": "accepted",
        "author_review_reason": "逐页核对后可支持该项监测原则。",
        "verified_text_span": candidate["raw_text"],
        "supported_claim_types": '["monitoring"]',
        "evidence_scope": "儿科用药后的复核与监测原则",
        "age_scope": "pediatric_unspecified",
        "applicability_conditions": "仅限指南所述适用人群与场景",
        "scope_check": "within_can_support",
    }


def test_review_queue_caps_pending_rows_and_preserves_page_topic_diversity():
    module = _load_module()
    candidates = [
        _candidate(
            f"CAND-{index}",
            page_number=index,
            topics=["monitoring" if index % 2 else "dose"],
        )
        for index in range(1, 9)
    ]

    queue, source_results = module.build_review_queue(
        candidates,
        [_coverage()],
        {},
        _config(),
    )

    pending = [
        row for row in queue if row["review_status"] == "pending_author_review"
    ]
    assert len(pending) == 5
    assert len({row["page_number"] for row in pending}) == 5
    assert {topic for row in pending for topic in row["matched_topics"]} == {
        "dose",
        "monitoring",
    }
    assert source_results == [
        {
            "source_id": "SRC-101",
            "candidate_count": 8,
            "pending_review_count": 5,
            "dev50_overlap_rejected_count": 0,
            "processing_status": "pending_author_review",
            "scope_notes": "",
        }
    ]


def test_dev50_overlap_is_audited_but_not_added_to_manual_workload():
    module = _load_module()
    candidate = _candidate("CAND-overlap", page_number=3)

    queue, _ = module.build_review_queue(
        [candidate],
        [_coverage()],
        {("SRC-101", 3): ["DEV50-ANCHOR-1"]},
        _config(),
    )

    assert len(queue) == 1
    assert queue[0]["review_status"] == "rejected_dev50_overlap"
    assert queue[0]["dev50_overlap_anchor_ids"] == ["DEV50-ANCHOR-1"]
    assert queue[0]["author_review_outcome"] == ""


def test_table_of_contents_dot_leader_fragments_are_not_shortlisted():
    module = _load_module()
    toc_fragment = _candidate(
        "CAND-toc",
        page_number=1,
        text=(
            "Intravenous treatment ........................................ 22 "
            "Antibiotic choice ............................................ 24 "
            "Monitoring ................................................... 31"
        ),
    )
    valid_candidates = [
        _candidate(f"CAND-valid-{index}", page_number=index)
        for index in range(2, 8)
    ]

    queue, _ = module.build_review_queue(
        [toc_fragment, *valid_candidates],
        [_coverage()],
        {},
        _config(),
    )

    pending_ids = {
        row["candidate_id"]
        for row in queue
        if row["review_status"] == "pending_author_review"
    }
    assert "CAND-toc" not in pending_ids
    assert len(pending_ids) == 5


def test_review_queue_csv_round_trip_preserves_json_field_types(tmp_path: Path):
    module = _load_module()
    queue, _ = module.build_review_queue(
        [_candidate("CAND-round-trip")],
        [_coverage()],
        {},
        _config(),
    )
    output_path = tmp_path / "review_queue.csv"

    module._write_review_queue(queue, output_path)
    restored = module._read_review_queue(output_path)

    assert restored[0]["matched_topics"] == ["monitoring"]
    assert restored[0]["matched_terms"] == {"monitoring": ["复核"]}
    assert restored[0]["dev50_overlap_anchor_ids"] == []


def test_incomplete_author_fields_cannot_be_promoted():
    module = _load_module()
    candidate = _candidate("CAND-incomplete")
    review = _accepted_review(candidate)
    review["author_review_reason"] = ""

    anchors, decisions = module.promote_verified_anchors(
        [review],
        [_coverage()],
        [candidate],
        {"SRC-101": 8},
        {},
        _config(),
    )

    assert anchors == []
    assert decisions[0]["promotion_status"] == "rejected_incomplete_review"


@pytest.mark.parametrize(
    "field",
    [
        "author_review_reason",
        "evidence_scope",
        "age_scope",
        "applicability_conditions",
    ],
)
def test_garbled_author_metadata_cannot_be_promoted(field: str):
    module = _load_module()
    candidate = _candidate("CAND-garbled-metadata")
    review = _accepted_review(candidate)
    review[field] = "????????"

    anchors, decisions = module.promote_verified_anchors(
        [review],
        [_coverage()],
        [candidate],
        {"SRC-101": 8},
        {},
        _config(),
    )

    assert anchors == []
    assert decisions[0]["promotion_status"] == "rejected_metadata_quality"


@pytest.mark.parametrize(
    ("outcome", "scope_check"),
    [
        ("rejected", "within_can_support"),
        ("accepted", "scope_limited"),
        ("scope_limited", "scope_limited"),
    ],
)
def test_rejected_or_scope_limited_rows_cannot_enter_anchor_pool(
    outcome: str,
    scope_check: str,
):
    module = _load_module()
    candidate = _candidate("CAND-scope")
    review = _accepted_review(candidate)
    review["author_review_outcome"] = outcome
    review["scope_check"] = scope_check

    anchors, decisions = module.promote_verified_anchors(
        [review],
        [_coverage()],
        [candidate],
        {"SRC-101": 8},
        {},
        _config(),
    )

    assert anchors == []
    assert decisions[0]["promotion_status"] == "rejected_scope"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_title", "被篡改的标题"),
        ("source_sha256", "bad-sha256"),
        ("page_number", 99),
    ],
)
def test_title_hash_or_page_mismatch_is_rejected(field: str, value):
    module = _load_module()
    candidate = _candidate("CAND-provenance")
    review = _accepted_review(candidate)
    review[field] = value

    anchors, decisions = module.promote_verified_anchors(
        [review],
        [_coverage()],
        [candidate],
        {"SRC-101": 8},
        {},
        _config(),
    )

    assert anchors == []
    assert decisions[0]["promotion_status"] == "rejected_provenance_mismatch"


@pytest.mark.parametrize(
    "verified_text",
    [
        "标题",
        "������������",
        "这段文字并不存在于候选片段或上下文中。",
    ],
)
def test_short_garbled_or_untraceable_text_is_rejected(verified_text: str):
    module = _load_module()
    candidate = _candidate("CAND-text")
    review = _accepted_review(candidate)
    review["verified_text_span"] = verified_text

    anchors, decisions = module.promote_verified_anchors(
        [review],
        [_coverage()],
        [candidate],
        {"SRC-101": 8},
        {},
        _config(),
    )

    assert anchors == []
    assert decisions[0]["promotion_status"] == "rejected_text_quality"


def test_dev50_overlap_cannot_be_promoted_even_after_author_acceptance():
    module = _load_module()
    candidate = _candidate("CAND-leakage")
    review = _accepted_review(candidate)

    anchors, decisions = module.promote_verified_anchors(
        [review],
        [_coverage()],
        [candidate],
        {"SRC-101": 8},
        {("SRC-101", 3): ["DEV50-ANCHOR-1"]},
        _config(),
    )

    assert anchors == []
    assert decisions[0]["promotion_status"] == "rejected_dev50_overlap"


def test_anchor_id_is_stable_and_contains_only_author_verified_fields():
    module = _load_module()
    candidate = _candidate("CAND-stable")
    review = _accepted_review(candidate)

    first, first_decisions = module.promote_verified_anchors(
        [review],
        [_coverage()],
        [candidate],
        {"SRC-101": 8},
        {},
        _config(),
    )
    second, second_decisions = module.promote_verified_anchors(
        [review],
        [_coverage()],
        [candidate],
        {"SRC-101": 8},
        {},
        _config(),
    )

    assert len(first) == len(second) == 1
    assert first[0]["anchor_id"] == second[0]["anchor_id"]
    assert first[0]["verification_status"] == "author_verified_anchor"
    assert first[0]["source_id"] == "SRC-101"
    assert first[0]["page_number"] == 3
    assert first_decisions[0]["promotion_status"] == "promoted"
    assert second_decisions[0]["promotion_status"] == "promoted"


def test_completed_author_review_is_not_counted_as_pending_for_source():
    module = _load_module()
    candidate = _candidate("CAND-reviewed")
    review = _accepted_review(candidate)

    source_results = module._rebuild_source_results(
        [_coverage()],
        [candidate],
        [review],
        _config(),
    )

    assert source_results[0]["pending_review_count"] == 0
    assert source_results[0]["processing_status"] == "author_review_complete"


def test_verification_report_excludes_completed_author_review_from_pending(
    tmp_path: Path,
):
    module = _load_module()
    candidate = _candidate("CAND-report-reviewed")
    review = _accepted_review(candidate)
    report_path = tmp_path / "verification_report.md"

    module._write_report(
        report_path,
        queue=[review],
        source_results=[],
        anchors=[],
        decisions=[],
        config=_config(),
        candidate_sha256="candidate-sha256",
        stage="promotion_validation_complete",
    )

    report = report_path.read_text(encoding="utf-8")
    assert "待作者逐页核验：0" in report
