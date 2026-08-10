import csv
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "benchmark_anchor_expansion_review.py"
)
FORMAL_CONFIG_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "configs"
    / "benchmark_anchor_expansion_review_v0_2.json"
)


def _load_module():
    assert MODULE_PATH.exists(), "Anchor expansion review module is not implemented"
    spec = importlib.util.spec_from_file_location(
        "benchmark_anchor_expansion_review",
        MODULE_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _queue_row(index: int, *, source_id: str = "SRC-101") -> dict:
    text = f"儿童用药证据片段 {index}：应核对剂量、频次和适用人群。"
    return {
        "candidate_id": f"CAND-{index:03d}",
        "source_id": source_id,
        "source_title": f"测试指南 {source_id}",
        "source_filename": f"{source_id}.pdf",
        "source_sha256": f"sha256-{source_id}",
        "page_number": str(index),
        "block_type": "text",
        "granularity": "512",
        "candidate_text": text,
        "context_candidate_id": f"CAND-{index:03d}",
        "context_text": text,
        "matched_topics": '["dose", "frequency"]',
        "matched_terms": '{"dose": ["剂量"], "frequency": ["频次"]}',
        "review_status": "pending_author_review",
        "dev50_overlap_anchor_ids": "[]",
        "selection_rank_within_source": "1",
        "parser_version": "parser-v1",
        "chunker_version": "chunker-v1",
        "candidate_config_version": "candidate-v1",
        "review_config_version": "benchmark-anchor-expansion-v0.2",
        "reviewer_id": "",
        "author_reviewed_at": "",
        "author_review_outcome": "",
        "author_review_reason": "",
        "verified_text_span": "",
        "supported_claim_types": "",
        "evidence_scope": "",
        "age_scope": "",
        "applicability_conditions": "",
        "scope_check": "",
    }


def _config() -> dict:
    return {
        "config_version": "benchmark-anchor-expansion-review-v0.2",
        "dataset_version": "benchmark-v1.0-pre-freeze",
        "kb_version": "KB-medium-v1",
        "expected_queue_size": 4,
        "batch_size": 2,
        "min_verified_text_chars": 10,
        "min_alnum_ratio": 0.25,
        "allowed_assistant_outcomes": [
            "accepted_draft",
            "revision_required",
            "rejected_draft",
        ],
    }


def _draft(row: dict, outcome: str = "accepted_draft") -> dict:
    return {
        **row,
        "assistant_review_outcome": outcome,
        "assistant_review_reason": "已核对来源页，片段可支持儿科剂量与频次核对。",
        "assistant_verified_text_span": row["candidate_text"],
        "assistant_supported_claim_types": '["dose", "frequency"]',
        "assistant_evidence_scope": "儿科用药剂量与频次核对",
        "assistant_age_scope": "pediatric_unspecified",
        "assistant_applicability_conditions": "仅限原指南所述场景",
        "assistant_scope_check": "within_can_support",
    }


def test_prepare_batches_is_deterministic_and_preserves_parent_hash():
    module = _load_module()
    rows = [_queue_row(index) for index in range(1, 5)]

    first = module.prepare_review_batches(rows, _config(), "parent-sha256")
    second = module.prepare_review_batches(rows, _config(), "parent-sha256")

    assert first == second
    assert [len(batch["rows"]) for batch in first] == [2, 2]
    assert {batch["parent_queue_sha256"] for batch in first} == {
        "parent-sha256"
    }
    assert all(
        row["assistant_review_outcome"] == ""
        for batch in first
        for row in batch["rows"]
    )


def test_prepare_rejects_nonempty_author_fields():
    module = _load_module()
    rows = [_queue_row(index) for index in range(1, 5)]
    rows[0]["author_review_outcome"] = "accepted"

    with pytest.raises(ValueError, match="作者字段"):
        module.prepare_review_batches(rows, _config(), "parent-sha256")


def test_validate_drafts_rejects_source_provenance_drift():
    module = _load_module()
    rows = [_queue_row(index) for index in range(1, 5)]
    drafts = [_draft(row) for row in rows]
    drafts[0]["source_sha256"] = "tampered"

    with pytest.raises(ValueError, match="来源字段漂移"):
        module.validate_assistant_drafts(drafts, rows, _config())


def test_accepted_draft_requires_traceable_verified_text():
    module = _load_module()
    rows = [_queue_row(index) for index in range(1, 5)]
    drafts = [_draft(row) for row in rows]
    drafts[0]["assistant_verified_text_span"] = "原候选与上下文中不存在的医学结论"

    with pytest.raises(ValueError, match="不可追溯"):
        module.validate_assistant_drafts(drafts, rows, _config())


def test_formal_config_rejects_heading_level_short_evidence():
    module = _load_module()
    config = module.load_config(FORMAL_CONFIG_PATH)
    row = _queue_row(1)
    row["candidate_text"] = "Fever for 5 days"
    row["context_text"] = row["candidate_text"]
    draft = _draft(row)
    draft["assistant_verified_text_span"] = row["candidate_text"]

    assert config["min_verified_text_chars"] >= 40
    with pytest.raises(ValueError, match="过短或不可读"):
        module.validate_assistant_drafts([draft], [row], {
            **config,
            "expected_queue_size": 1,
        })


def test_write_outputs_keeps_author_fields_empty_and_records_zero_usage(
    tmp_path: Path,
):
    module = _load_module()
    rows = [_queue_row(index) for index in range(1, 5)]
    drafts = [_draft(row) for row in rows]
    validated = module.validate_assistant_drafts(drafts, rows, _config())

    outputs = module.write_review_outputs(
        validated,
        parent_queue_sha256="parent-sha256",
        config=_config(),
        output_dir=tmp_path,
    )

    with Path(outputs["draft_csv"]).open(encoding="utf-8-sig", newline="") as handle:
        saved_rows = list(csv.DictReader(handle))
    summary = json.loads(Path(outputs["summary_json"]).read_text(encoding="utf-8"))

    assert all(row["author_review_outcome"] == "" for row in saved_rows)
    assert summary["assistant_reviewed_count"] == 4
    assert summary["author_confirmed_count"] == 0
    assert summary["external_api_calls"] == 0
    assert summary["input_tokens"] == 0
    assert summary["output_tokens"] == 0
    assert summary["estimated_cost"] == 0
