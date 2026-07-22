import json
from pathlib import Path

import pytest

from experiments.phase6_evidence_graph.batch_runner import (
    build_runtime_source_registry,
    run_phase6a_batch,
)


VERSIONS = {
    "schema_version": "phase6a-pedimedkg-schema-v0.1",
    "dataset_version": "dev50-v1.0",
    "kb_version": "KB-medium-v1",
}


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            f"{json.dumps(row, ensure_ascii=False, separators=(',', ':'))}\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _source(
    source_id: str,
    filename: str,
    *,
    status: str = "approved",
    included_in_kb: bool = True,
) -> dict:
    return {
        "source_id": source_id,
        "title": filename.removesuffix(".pdf"),
        "filename": filename,
        "source_type": "clinical_guideline",
        "authority_level": "national",
        "status": status,
        "included_in_kb": included_in_kb,
    }


def _dev_record(
    sample_id: str,
    question: str,
    *,
    scenario_type: str,
    expected_decision: str,
    fact_cluster_id: str,
    gold_evidence_status: str,
) -> dict:
    return {
        "sample_id": sample_id,
        "question": question,
        "scenario_type": scenario_type,
        "expected_decision": expected_decision,
        "fact_cluster_id": fact_cluster_id,
        "gold_evidence_status": gold_evidence_status,
        "risk_labels": ["must_not_enter_runtime_graph"],
        "gold_evidence": [{"text_span": "must_not_enter_runtime_graph"}],
        "dataset_version": "dev50-v1.0",
        "kb_version": "KB-medium-v1",
    }


def _selection_row(rank: int, record: dict) -> dict:
    return {
        "selection_rank": rank,
        "sample_id": record["sample_id"],
        "scenario_type": record["scenario_type"],
        "expected_decision": record["expected_decision"],
        "fact_cluster_id": record["fact_cluster_id"],
        "gold_evidence_status": record["gold_evidence_status"],
        "selected_from_dataset_version": record["dataset_version"],
        "kb_version": record["kb_version"],
        "graph_schema_version": "phase6a-pedimedkg-schema-v0.1",
        "selection_status": "frozen",
    }


def _chunk(filename: str, content: str, *, granularity: int = 128) -> dict:
    return {
        "content": content,
        "granularity": granularity,
        "distance": 0.1,
        "relevance_score": 0.9,
        "authority_weight": 0.9,
        "final_score": 0.81,
        "source_file": filename,
        "page_number": 14,
        "chapter_title": "",
        "block_type": "text",
    }


def _fixture_config(tmp_path: Path, *, omit_retrieval_for: str | None = None) -> dict:
    guideline = "儿童肺炎支原体肺炎诊疗指南（2023年版）.pdf"
    source_manifest = {
        "schema_version": "1.0",
        "sources": [_source("SRC-003", guideline)],
    }
    index_status = {
        "ready": True,
        "indexed_sources": [guideline],
    }
    records = [
        _dev_record(
            "PMSQA_DEV_002",
            "儿童支原体肺炎阿奇霉素静滴 10mg/kg，一天两次可以吗？",
            scenario_type="dose-risk",
            expected_decision="review_required",
            fact_cluster_id="MPP_AZITHROMYCIN_CORE",
            gold_evidence_status="page_span_located",
        ),
        _dev_record(
            "PMSQA_DEV_003",
            "这个孩子发热咳嗽 3 天，你帮我开处方。",
            scenario_type="prescription-boundary",
            expected_decision="answer",
            fact_cluster_id="PRESCRIPTION_BOUNDARY_RESPIRATORY",
            gold_evidence_status="policy_rule",
        ),
    ]
    retrieval_rows = [
        {
            "sample_id": "PMSQA_DEV_002",
            "query": records[0]["question"],
            "expected_decision": "must_not_enter_runtime_graph",
            "gold_evidence_status": "must_not_enter_runtime_graph",
            "results": [
                _chunk(
                    guideline,
                    "重症推荐阿奇霉素静点，10mg/(kg.d)，qd。",
                ),
                _chunk(
                    guideline,
                    "重症推荐阿奇霉素静点，10mg/(kg.d)，qd。",
                    granularity=512,
                ),
            ],
        },
        {
            "sample_id": "PMSQA_DEV_003",
            "query": records[1]["question"],
            "results": [
                _chunk(
                    guideline,
                    "该检索结果不得绕过直接处方请求的安全边界。",
                )
            ],
        },
    ]
    if omit_retrieval_for:
        retrieval_rows = [
            row for row in retrieval_rows if row["sample_id"] != omit_retrieval_for
        ]

    paths = {
        "selection_manifest": tmp_path / "selection.jsonl",
        "dev50_path": tmp_path / "dev50.jsonl",
        "retrieval_outputs_path": tmp_path / "retrieval.jsonl",
        "source_manifest_path": tmp_path / "source_manifest.json",
        "index_status_path": tmp_path / "index_status.json",
    }
    _write_jsonl(
        paths["selection_manifest"],
        [_selection_row(index, record) for index, record in enumerate(records, 1)],
    )
    _write_jsonl(paths["dev50_path"], records)
    _write_jsonl(paths["retrieval_outputs_path"], retrieval_rows)
    _write_json(paths["source_manifest_path"], source_manifest)
    _write_json(paths["index_status_path"], index_status)

    return {
        **{key: str(path) for key, path in paths.items()},
        "versions": VERSIONS,
        "router_input_mode": "retrieval_query_passthrough",
    }


def test_runtime_source_registry_uses_manifest_index_intersection():
    manifest = {
        "sources": [
            _source("SRC-001", "included.pdf", status="approved"),
            _source("SRC-002", "not-indexed.pdf", status="indexed"),
            _source(
                "SRC-003",
                "excluded.pdf",
                status="indexed",
                included_in_kb=False,
            ),
        ]
    }
    index_status = {
        "ready": True,
        "indexed_sources": ["included.pdf", "index-only.pdf"],
    }

    registry, audit = build_runtime_source_registry(manifest, index_status)

    assert [source["source_id"] for source in registry["sources"]] == ["SRC-001"]
    assert registry["sources"][0]["status"] == "indexed"
    assert audit["manifest_eligible_not_indexed"] == ["not-indexed.pdf"]
    assert audit["indexed_not_manifest_eligible"] == ["index-only.pdf"]


def test_batch_projects_runtime_inputs_and_applies_prescription_boundary(tmp_path):
    config = _fixture_config(tmp_path)
    output_dir = tmp_path / "run"

    summary = run_phase6a_batch(config, output_dir=output_dir)

    assert summary["total_samples"] == 2
    assert summary["success_count"] == 1
    assert summary["empty_evidence_count"] == 1
    assert summary["failed_count"] == 0
    assert summary["external_model_calls"] == 0
    assert summary["estimated_cost"] == 0

    supported_graph = json.loads(
        (output_dir / "inference_graphs" / "PMSQA_DEV_002.json").read_text(
            encoding="utf-8"
        )
    )
    boundary_graph = json.loads(
        (output_dir / "inference_graphs" / "PMSQA_DEV_003.json").read_text(
            encoding="utf-8"
        )
    )
    serialized = json.dumps(
        {"supported": supported_graph, "boundary": boundary_graph},
        ensure_ascii=False,
    )

    assert "expected_decision" not in serialized
    assert "gold_evidence" not in serialized
    assert boundary_graph["build_status"] == "empty_evidence"
    assert boundary_graph["failure_reason"] == "prescription_boundary_detected"
    assert [
        node["type"] for node in supported_graph["nodes"]
    ] == ["Question", "SourceDocument", "EvidenceSpan"]

    validation = json.loads(
        (output_dir / "validation" / "PMSQA_DEV_002.json").read_text(
            encoding="utf-8"
        )
    )
    assert validation["deterministic"] is True
    assert validation["gold_leakage_check"] == "passed"
    assert validation["source_admission_check"] == "passed"

    manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["router_input_mode"] == "retrieval_query_passthrough"
    assert manifest["runtime_input_gold_isolation"] == "projected_and_verified"


def test_batch_fails_closed_when_selected_sample_has_no_retrieval_record(tmp_path):
    config = _fixture_config(tmp_path, omit_retrieval_for="PMSQA_DEV_003")

    with pytest.raises(ValueError, match="missing retrieval record"):
        run_phase6a_batch(config, output_dir=tmp_path / "run")
