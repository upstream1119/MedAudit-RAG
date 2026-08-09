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
    / "benchmark_anchor_expansion.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "Benchmark anchor expansion module is not implemented"
    spec = importlib.util.spec_from_file_location(
        "benchmark_anchor_expansion",
        MODULE_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source(source_id: str, title: str = "儿童用药测试指南") -> dict:
    return {
        "source_id": source_id,
        "title": title,
        "filename": f"{source_id}.pdf",
        "actual_sha256": f"sha256-{source_id}",
        "evidence_types": ["pediatric_dosing", "monitoring"],
    }


def _candidate(
    candidate_id: str,
    *,
    source_id: str = "SRC-101",
    page_number: int = 1,
    text: str = "儿童用药后应复核剂量、频次与监测结果。",
    topics: list[str] | None = None,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "source_id": source_id,
        "source_title": "儿童用药测试指南",
        "source_filename": f"{source_id}.pdf",
        "source_sha256": f"sha256-{source_id}",
        "page_number": page_number,
        "block_type": "text",
        "granularity": 512,
        "raw_text": text,
        "matched_topics": topics or ["monitoring"],
        "matched_terms": {"monitoring": ["复核"]},
        "review_status": "candidate_unverified",
        "parser_version": "parser-v1",
        "chunker_version": "chunker-v1",
        "config_version": "candidate-v1",
    }


def _config() -> dict:
    return {
        "config_version": "benchmark-anchor-expansion-v0.2",
        "dataset_version": "benchmark-v1.0-pre-freeze",
        "kb_version": "KB-medium-v1",
        "expected_source_count": 1,
        "max_pending_per_source": 3,
        "preferred_granularities": [512, 128, 1024],
        "scope_limited_evidence_types": [
            "medicine_identity",
            "dosage_form",
            "essential_medicine_status",
        ],
        "require_pediatric_relevance": True,
        "pediatric_terms": ["儿童", "儿科", "children", "pediatric"],
        "reference_heading_terms": ["参考文献", "references", "bibliography"],
        "max_citation_markers": 1,
    }


def test_prepare_excludes_previous_anchor_dev50_and_review_pages(tmp_path: Path):
    module = _load_module()
    candidates = [
        _candidate("CAND-old-review", page_number=1),
        _candidate("CAND-old-anchor", page_number=2),
        _candidate("CAND-dev50", page_number=3),
        _candidate("CAND-fresh", page_number=4),
    ]

    queue, summary = module.build_expansion_queue(
        candidates=candidates,
        coverage_rows=[_source("SRC-101")],
        previous_review_rows=[{"source_id": "SRC-101", "page_number": 1}],
        existing_anchors=[{"source_id": "SRC-101", "page_number": 2}],
        dev50_pairs={
            ("SRC-101", 3): ["DEV50-ANCHOR-1"],
        },
        config=_config(),
    )

    assert [row["candidate_id"] for row in queue] == ["CAND-fresh"]
    assert summary["excluded_source_page_count"] == 3
    assert summary["pending_author_review"] == 1


def test_prepare_filters_reference_and_non_pediatric_fragments():
    module = _load_module()
    source = _source("SRC-101", title="General antimicrobial guideline")
    candidates = [
        _candidate(
            "CAND-reference",
            page_number=1,
            text="参考文献 [12] Example Journal. [13] Another Journal.",
        ),
        _candidate(
            "CAND-adult",
            page_number=2,
            text="Adults should be monitored after antimicrobial treatment.",
        ),
        _candidate(
            "CAND-child",
            page_number=3,
            text="Children should be reassessed after antimicrobial treatment.",
        ),
    ]
    for candidate in candidates:
        candidate["source_title"] = source["title"]

    queue, summary = module.build_expansion_queue(
        candidates=candidates,
        coverage_rows=[source],
        previous_review_rows=[],
        existing_anchors=[],
        dev50_pairs={},
        config=_config(),
    )

    assert [row["candidate_id"] for row in queue] == ["CAND-child"]
    assert summary["noise_filtered_count"] == 2


def test_prepare_selects_distinct_pages_and_is_deterministic():
    module = _load_module()
    candidates = [
        _candidate(f"CAND-{index}", page_number=(index + 1) // 2)
        for index in range(1, 9)
    ]
    kwargs = {
        "candidates": candidates,
        "coverage_rows": [_source("SRC-101")],
        "previous_review_rows": [],
        "existing_anchors": [],
        "dev50_pairs": {},
        "config": _config(),
    }

    first_queue, first_summary = module.build_expansion_queue(**kwargs)
    second_queue, second_summary = module.build_expansion_queue(**kwargs)

    assert first_queue == second_queue
    assert first_summary == second_summary
    assert len(first_queue) == 3
    assert len({row["page_number"] for row in first_queue}) == 3


def test_prepare_prioritizes_configured_risk_topics():
    module = _load_module()
    config = _config()
    config["max_pending_per_source"] = 1
    config["priority_topics"] = ["dose", "monitoring"]
    candidates = [
        _candidate(
            "CAND-monitoring",
            page_number=1,
            topics=["monitoring"],
        ),
        _candidate(
            "CAND-dose",
            page_number=2,
            topics=["dose"],
        ),
    ]

    queue, _ = module.build_expansion_queue(
        candidates=candidates,
        coverage_rows=[_source("SRC-101")],
        previous_review_rows=[],
        existing_anchors=[],
        dev50_pairs={},
        config=config,
    )

    assert [row["candidate_id"] for row in queue] == ["CAND-dose"]


def test_prepare_replaces_high_priority_table_of_contents_noise():
    module = _load_module()
    config = _config()
    config["max_pending_per_source"] = 1
    config["priority_topics"] = ["dose", "monitoring"]
    candidates = [
        _candidate(
            "CAND-toc",
            page_number=1,
            topics=["dose"],
            text=(
                "儿童剂量 ........................................ 12 "
                "儿童监测 ........................................ 18 "
                "儿童疗程 ........................................ 25"
            ),
        ),
        _candidate(
            "CAND-valid",
            page_number=2,
            topics=["monitoring"],
        ),
    ]

    queue, _ = module.build_expansion_queue(
        candidates=candidates,
        coverage_rows=[_source("SRC-101")],
        previous_review_rows=[],
        existing_anchors=[],
        dev50_pairs={},
        config=config,
    )

    assert [row["candidate_id"] for row in queue] == ["CAND-valid"]


def test_write_outputs_refuses_to_overwrite_reviewed_queue(tmp_path: Path):
    module = _load_module()
    queue_path = tmp_path / "anchor_expansion_review_queue_v0_2.csv"
    with queue_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["candidate_id", "author_reviewed_at"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "candidate_id": "CAND-reviewed",
                "author_reviewed_at": "2026-08-09",
            }
        )

    with pytest.raises(ValueError, match="人工核验"):
        module.write_expansion_outputs(
            queue=[],
            summary={"dataset_version": "benchmark-v1.0-pre-freeze"},
            output_dir=tmp_path,
        )


def test_write_outputs_records_versions_and_hashes(tmp_path: Path):
    module = _load_module()
    queue = [
        {
            "candidate_id": "CAND-fresh",
            "source_id": "SRC-101",
            "page_number": 4,
            "review_status": "pending_author_review",
            "author_reviewed_at": "",
        }
    ]
    summary = {
        "config_version": "benchmark-anchor-expansion-v0.2",
        "dataset_version": "benchmark-v1.0-pre-freeze",
        "kb_version": "KB-medium-v1",
        "pending_author_review": 1,
        "input_sha256": {"candidates": "abc"},
        "external_api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
    }

    outputs = module.write_expansion_outputs(
        queue=queue,
        summary=summary,
        output_dir=tmp_path,
    )

    saved = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))
    assert saved["config_version"] == "benchmark-anchor-expansion-v0.2"
    assert saved["dataset_version"] == "benchmark-v1.0-pre-freeze"
    assert saved["kb_version"] == "KB-medium-v1"
    assert saved["input_sha256"] == {"candidates": "abc"}
    assert saved["external_api_calls"] == 0


def test_prepare_reports_quality_filtered_target_shortfall():
    module = _load_module()
    config = _config()
    config["target_review_queue_size"] = 3

    _, summary = module.build_expansion_queue(
        candidates=[_candidate("CAND-only", page_number=1)],
        coverage_rows=[_source("SRC-101")],
        previous_review_rows=[],
        existing_anchors=[],
        dev50_pairs={},
        config=config,
    )

    assert summary["target_review_queue_size"] == 3
    assert summary["target_shortfall"] == 2
    assert summary["target_met"] is False
