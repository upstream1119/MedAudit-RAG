import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
AUDITOR_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "audit_generation_claims.py"
)


def _load_auditor():
    assert AUDITOR_PATH.exists(), "Phase 7 generation claim auditor is not implemented"
    spec = importlib.util.spec_from_file_location(
        "phase7_generation_claim_auditor", AUDITOR_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _evidence(
    evidence_id: str,
    content: str,
    *,
    rank_before: int,
    rank_after: int,
) -> dict:
    return {
        "evidence_id": evidence_id,
        "source_id": "SRC-002",
        "source_file": "儿童社区获得性肺炎诊疗规范（2019年版）.pdf",
        "page_number": 26,
        "content": content,
        "rank_before": rank_before,
        "rank_after": rank_after,
        "runtime_constraints": [
            {
                "constraint_type": "monitoring_window",
                "normalized_value": "48-72h",
                "surface_forms": ["48-72小时"],
                "ruleset_version": "phase6b-runtime-constraint-rules-v0.3",
            },
            {
                "constraint_type": "monitoring_trigger",
                "normalized_value": "nonresponse",
                "surface_forms": ["症状无改善"],
                "ruleset_version": "phase6b-runtime-constraint-rules-v0.3",
            },
            {
                "constraint_type": "monitoring_action",
                "normalized_value": "reassess",
                "surface_forms": ["再次评估"],
                "ruleset_version": "phase6b-runtime-constraint-rules-v0.3",
            },
        ],
    }


def _parent_artifact() -> dict:
    return {
        "artifact_schema_version": "phase6b-reranking-artifact-v0.2",
        "artifact_type": "phase6b_reranking_artifact",
        "artifact_status": "success",
        "artifact_sha256": "parent-sha",
        "sample_id": "PMSQA_DEV_001",
        "parent_inference_graph_id": "inference_graph::PMSQA_DEV_001",
        "parent_inference_graph_sha256": "graph-sha",
        "versions": {
            "schema_version": "phase6a-pedimedkg-schema-v0.1",
            "dataset_version": "dev50-v1.0",
            "kb_version": "KB-medium-v1",
        },
        "method_id": "runtime_constraint_graph_reranking",
        "method_version": "phase6b-runtime-constraint-v0.4",
        "ruleset_version": "phase6b-runtime-constraint-rules-v0.3",
        "ranked_evidence": [
            _evidence(
                "evidence::support",
                "治疗48-72小时症状无改善，应再次评估。",
                rank_before=2,
                rank_after=1,
            ),
            _evidence(
                "evidence::contradict",
                "治疗48-72小时症状无改善，不应再次评估。",
                rank_before=1,
                rank_after=2,
            ),
        ],
        "external_model_calls": 0,
        "estimated_cost": 0,
    }


def _generation_run(tmp_path: Path) -> tuple[Path, Path]:
    generation_dir = tmp_path / "generation"
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(
        json.dumps(_parent_artifact(), ensure_ascii=False), encoding="utf-8"
    )
    answer = "治疗48-72小时症状无改善，应再次评估。"
    output_rows = []
    metadata_rows = []
    for method_id, cache_key in [
        ("vector_only_rag", "vector-key"),
        ("graph_enhanced_rag", "graph-key"),
    ]:
        output_rows.append(
            {
                "sample_id": "PMSQA_DEV_001",
                "method_id": method_id,
                "model_provider": "zhipu",
                "model_name": "glm-test",
                "prompt_version": "phase6b-generator-v0.1",
                "dataset_version": "dev50-v1.0",
                "kb_version": "KB-medium-v1",
                "inference_profile": "glm-test-nonthinking-v0.1",
                "cache_key": cache_key,
                "status": "success",
                "raw_output": answer,
            }
        )
        metadata_rows.append(
            {
                "sample_id": "PMSQA_DEV_001",
                "method_id": method_id,
                "cache_key": cache_key,
                "parent_reranking_artifact": str(parent_path),
            }
        )
    _write_jsonl(generation_dir / "raw_model_outputs.jsonl", output_rows)
    _write_jsonl(generation_dir / "evaluation_metadata.jsonl", metadata_rows)
    return generation_dir, parent_path


def test_audit_uses_method_specific_evidence_order_without_mutating_parent(
    tmp_path,
):
    auditor = _load_auditor()
    generation_dir, parent_path = _generation_run(tmp_path)
    parent_before = parent_path.read_bytes()
    output_dir = tmp_path / "claim_audit"

    summary = auditor.audit_generation_run(
        generation_dir,
        output_dir=output_dir,
        evidence_budget=1,
    )

    assert summary["total_outputs"] == 2
    assert summary["supported_count"] == 1
    assert summary["contradicted_count"] == 1
    assert summary["failed_count"] == 0
    rows = [
        json.loads(line)
        for line in (output_dir / "claim_audit_rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    states = {row["method_id"]: row["support_state"] for row in rows}
    assert states == {
        "vector_only_rag": "contradicted",
        "graph_enhanced_rag": "supported",
    }
    assert parent_path.read_bytes() == parent_before


def test_failed_generation_is_recorded_without_claim_audit(tmp_path):
    auditor = _load_auditor()
    generation_dir, _ = _generation_run(tmp_path)
    raw_path = generation_dir / "raw_model_outputs.jsonl"
    rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["status"] = "failed"
    rows[0]["raw_output"] = ""
    _write_jsonl(raw_path, rows)
    output_dir = tmp_path / "claim_audit"

    summary = auditor.audit_generation_run(
        generation_dir,
        output_dir=output_dir,
        evidence_budget=1,
    )

    assert summary["total_outputs"] == 2
    assert summary["audited_outputs"] == 1
    assert summary["failed_count"] == 1
    failed_rows = [
        json.loads(line)
        for line in (output_dir / "failed_cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert failed_rows[0]["cache_key"] == "vector-key"
    assert failed_rows[0]["error_type"] == "ValueError"
    assert "not auditable" in failed_rows[0]["error"]


def test_cache_key_mismatch_stops_before_output_creation(tmp_path):
    auditor = _load_auditor()
    generation_dir, _ = _generation_run(tmp_path)
    metadata_path = generation_dir / "evaluation_metadata.jsonl"
    metadata_rows = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
    ]
    _write_jsonl(metadata_path, metadata_rows[:1])
    output_dir = tmp_path / "claim_audit"

    with pytest.raises(ValueError, match="cache_key sets differ"):
        auditor.audit_generation_run(
            generation_dir,
            output_dir=output_dir,
            evidence_budget=1,
        )

    assert not output_dir.exists()


def test_missing_parent_artifact_path_is_reported_explicitly(tmp_path):
    auditor = _load_auditor()
    generation_dir, _ = _generation_run(tmp_path)
    metadata_path = generation_dir / "evaluation_metadata.jsonl"
    metadata_rows = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
    ]
    metadata_rows[0].pop("parent_reranking_artifact")
    _write_jsonl(metadata_path, metadata_rows)
    output_dir = tmp_path / "claim_audit"

    summary = auditor.audit_generation_run(
        generation_dir,
        output_dir=output_dir,
        evidence_budget=1,
    )

    assert summary["failed_count"] == 1
    failed_rows = [
        json.loads(line)
        for line in (output_dir / "failed_cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert failed_rows[0]["error_type"] == "ValueError"
    assert failed_rows[0]["error"] == "missing parent_reranking_artifact"
