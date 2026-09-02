from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "validation40_retrieval_execution.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "Validation40 retrieval execution module is missing"
    spec = importlib.util.spec_from_file_location(
        "validation40_retrieval_execution", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class _Chunk:
    content: str
    granularity: int
    distance: float = 0.1
    relevance_score: float = 0.9
    authority_weight: float = 0.9
    final_score: float = 0.81
    source_file: str = "儿童社区获得性肺炎诊疗规范（2019年版）.pdf"
    page_number: int = 26
    chapter_title: str = "再次评估"
    block_type: str = "text"


class _RecordingRetriever:
    def __init__(self):
        self.calls: list[dict] = []

    def retrieve(self, query: str, top_k: int, granularity: int | None = None):
        self.calls.append(
            {"query": query, "top_k": top_k, "granularity": granularity}
        )
        return [_Chunk(content=f"{query}-{granularity}", granularity=granularity or 128)]


class _FailingNaiveRetriever(_RecordingRetriever):
    def retrieve(self, query: str, top_k: int, granularity: int | None = None):
        if granularity == 512:
            self.calls.append(
                {"query": query, "top_k": top_k, "granularity": granularity}
            )
            raise RuntimeError("local embedding failure")
        return super().retrieve(query, top_k, granularity)


class _NoisyRetriever(_RecordingRetriever):
    def retrieve(self, query: str, top_k: int, granularity: int | None = None):
        self.calls.append(
            {"query": query, "top_k": top_k, "granularity": granularity}
        )
        valid = _Chunk(content="治疗后 48-72 小时无改善时应再次评估。", granularity=512)
        return [
            _Chunk(
                content="儿童社区获得性肺炎诊疗规范（2019年版）",
                granularity=512,
            ),
            valid,
            deepcopy(valid),
            _Chunk(
                content="来源页码缺失的证据。",
                granularity=512,
                page_number=0,
            ),
        ]


def _plan_rows() -> list[dict]:
    rows = []
    for method_id, mode, use_graph in (
        ("naive_rag", "single_granularity", False),
        ("multi_granularity_rag", "multi_granularity", False),
        ("trust_gated_rag", "multi_granularity", False),
        ("graph_enhanced_full", "multi_granularity_graph", True),
    ):
        rows.append(
            {
                "dataset_version": "benchmark-v1.0-guideline-grounded-author-adjudicated",
                "evidence_context_max": 4,
                "evidence_context_min": 2,
                "execution_status": "planned_not_executed",
                "kb_version": "KB-medium-v1",
                "method_id": method_id,
                "method_version": f"{method_id}-v0.1",
                "preflight_version": "validation40-experiment-preflight-v0.1",
                "question": "儿童肺炎治疗后是否需要再次评估？",
                "retrieval_cache_key": f"logical-{method_id}",
                "retrieval_mode": mode,
                "retrieval_task_id": f"RET-{method_id}",
                "retrieval_top_k": 4,
                "sample_id": "PMSQA-VALIDATION-001",
                "seed": 20260817,
                "use_graph": use_graph,
            }
        )
    return rows


def _config() -> dict:
    return {
        "execution_version": "validation40-retrieval-execution-v0.1",
        "expected_retrieval_task_count": 4,
        "expected_retrieval_plan_sha256": "plan-hash",
        "expected_dataset_version": (
            "benchmark-v1.0-guideline-grounded-author-adjudicated"
        ),
        "expected_kb_version": "KB-medium-v1",
        "naive_granularity": 512,
        "evidence_context_min": 2,
        "evidence_context_max": 4,
        "execute_model_calls": False,
        "graph_reranking_enabled": False,
    }


def test_executes_naive_and_shared_multi_profiles_without_model_calls():
    module = _load_module()
    retriever = _RecordingRetriever()

    result = module.build_retrieval_execution(
        _plan_rows(),
        _config(),
        observed_plan_sha256="plan-hash",
        retriever=retriever,
    )

    assert len(retriever.calls) == 2
    assert {call["granularity"] for call in retriever.calls} == {512, None}
    assert len(result["task_results"]) == 4
    assert result["audit"]["logical_retrieval_task_count"] == 4
    assert result["audit"]["physical_retrieval_count"] == 2
    assert result["audit"]["external_model_calls"] == 0
    assert result["audit"]["graph_reranking_executed"] is False


def test_rejects_plan_or_version_drift_before_retrieval():
    module = _load_module()
    rows = _plan_rows()
    rows[0]["kb_version"] = "KB-drifted"
    retriever = _RecordingRetriever()

    with pytest.raises(ValueError, match="KB version mismatch"):
        module.build_retrieval_execution(
            rows,
            deepcopy(_config()),
            observed_plan_sha256="plan-hash",
            retriever=retriever,
        )

    assert retriever.calls == []


def test_records_profile_failure_without_aborting_or_padding_evidence():
    module = _load_module()
    result = module.build_retrieval_execution(
        _plan_rows(),
        _config(),
        observed_plan_sha256="plan-hash",
        retriever=_FailingNaiveRetriever(),
    )

    naive = next(row for row in result["task_results"] if row["method_id"] == "naive_rag")
    multi = next(
        row
        for row in result["task_results"]
        if row["method_id"] == "multi_granularity_rag"
    )
    assert naive["execution_status"] == "failed"
    assert naive["evidence"] == []
    assert multi["execution_status"] == "insufficient_evidence"
    assert len(multi["evidence"]) == 1
    assert len(result["failures"]) == 1
    assert result["audit"]["failed_physical_retrieval_count"] == 1
    failed_physical = next(
        row for row in result["physical_results"] if row["status"] == "failed"
    )
    assert failed_physical["error"] == {
        "error_type": "RuntimeError",
        "error_message": "local embedding failure",
    }


def test_filters_title_only_duplicate_and_invalid_provenance_without_padding():
    module = _load_module()
    result = module.build_retrieval_execution(
        _plan_rows(),
        _config(),
        observed_plan_sha256="plan-hash",
        retriever=_NoisyRetriever(),
    )

    for physical in result["physical_results"]:
        assert physical["status"] == "insufficient_evidence"
        assert len(physical["evidence"]) == 1
        assert physical["evidence"][0]["page_number"] == 26
        assert physical["evidence"][0]["source_file"].endswith(".pdf")
        assert physical["evidence_audit"] == {
            "raw_chunk_count": 4,
            "admitted_evidence_count": 1,
            "duplicate_count": 1,
            "title_only_count": 1,
            "invalid_provenance_count": 1,
        }

    assert result["audit"]["duplicate_evidence_count"] == 2
    assert result["audit"]["title_only_evidence_count"] == 2
    assert result["audit"]["invalid_provenance_count"] == 2


def test_rejects_gold_only_field_leakage_before_retrieval():
    module = _load_module()
    rows = _plan_rows()
    rows[0]["expected_decision"] = "answer"
    retriever = _RecordingRetriever()

    with pytest.raises(ValueError, match="retrieval plan field allowlist mismatch"):
        module.build_retrieval_execution(
            rows,
            _config(),
            observed_plan_sha256="plan-hash",
            retriever=retriever,
        )

    assert retriever.calls == []


def test_reuses_completed_physical_cache_and_retries_failed_entries():
    module = _load_module()
    rows = _plan_rows()
    profiles = [module._physical_profile(row, 512) for row in rows]
    naive_key = module._canonical_sha256(profiles[0])
    multi_key = module._canonical_sha256(profiles[1])
    cached_evidence = [module._project_chunk(_Chunk(content="cached", granularity=512))]
    cached = {
        naive_key: {
            "physical_retrieval_key": naive_key,
            "profile": profiles[0],
            "evidence": cached_evidence,
            "evidence_audit": {
                "raw_chunk_count": 1,
                "admitted_evidence_count": 1,
                "duplicate_count": 0,
                "title_only_count": 0,
                "invalid_provenance_count": 0,
            },
            "status": "insufficient_evidence",
        },
        multi_key: {
            "physical_retrieval_key": multi_key,
            "profile": profiles[1],
            "evidence": [],
            "evidence_audit": {
                "raw_chunk_count": 0,
                "admitted_evidence_count": 0,
                "duplicate_count": 0,
                "title_only_count": 0,
                "invalid_provenance_count": 0,
            },
            "status": "failed",
        },
    }
    retriever = _RecordingRetriever()

    result = module.build_retrieval_execution(
        rows,
        _config(),
        observed_plan_sha256="plan-hash",
        retriever=retriever,
        cached_physical_results=cached,
    )

    assert retriever.calls == [
        {
            "query": "儿童肺炎治疗后是否需要再次评估？",
            "top_k": 4,
            "granularity": None,
        }
    ]
    assert len(result["task_results"]) == 4
    assert {row["method_id"] for row in result["task_results"]} == {
        "naive_rag",
        "multi_granularity_rag",
        "trust_gated_rag",
        "graph_enhanced_full",
    }
    assert result["audit"]["physical_retrieval_count"] == 2


def test_validates_frozen_inputs_and_embedding_identity(tmp_path: Path):
    module = _load_module()
    plan_path = tmp_path / "plan.jsonl"
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    status_path = chroma_dir / "index_status.json"
    summary_path = tmp_path / "rebuild_summary.json"
    pilot_path = tmp_path / "pilot.jsonl"
    plan_path.write_text("{}\n", encoding="utf-8")
    status_path.write_text(
        json.dumps(
            {
                "ready": True,
                "expected_sources": ["a.pdf", "b.pdf"],
                "indexed_sources": ["a.pdf", "b.pdf"],
                "missing_sources": [],
            }
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "pdf_count": 2,
                "index_status": {
                    "embedding_provider": "local",
                    "embedding_model": "BAAI/bge-small-zh-v1.5",
                },
            }
        ),
        encoding="utf-8",
    )
    pilot_path.write_text('{"hidden": true}\n', encoding="utf-8")
    config = {
        "retrieval_plan_path": plan_path.name,
        "chroma_persist_dir": chroma_dir.name,
        "index_status_path": "chroma/index_status.json",
        "index_rebuild_summary_path": summary_path.name,
        "pilot_test_path": pilot_path.name,
        "expected_retrieval_plan_sha256": module.compute_sha256(plan_path),
        "expected_index_status_sha256": module.compute_sha256(status_path),
        "expected_index_rebuild_summary_sha256": module.compute_sha256(summary_path),
        "expected_pilot_test_sha256": module.compute_sha256(pilot_path),
        "expected_index_source_count": 2,
        "expected_embedding_provider": "local",
        "expected_embedding_model": "BAAI/bge-small-zh-v1.5",
    }

    provenance = module.validate_frozen_inputs(
        config,
        repo_root=tmp_path,
        runtime_embedding_provider="local",
        runtime_embedding_model="BAAI/bge-small-zh-v1.5",
        runtime_chroma_persist_dir=chroma_dir,
    )
    assert provenance["pilot_test_accessed"] is False
    assert provenance["pilot_test_sha256_before"] == module.compute_sha256(pilot_path)

    with pytest.raises(ValueError, match="runtime embedding model mismatch"):
        module.validate_frozen_inputs(
            config,
            repo_root=tmp_path,
            runtime_embedding_provider="local",
            runtime_embedding_model="wrong-model",
            runtime_chroma_persist_dir=chroma_dir,
        )

    with pytest.raises(ValueError, match="runtime Chroma persist directory mismatch"):
        module.validate_frozen_inputs(
            config,
            repo_root=tmp_path,
            runtime_embedding_provider="local",
            runtime_embedding_model="BAAI/bge-small-zh-v1.5",
            runtime_chroma_persist_dir=tmp_path / "wrong-chroma",
        )


def test_run_writes_immutable_outputs_and_reuses_disk_cache(tmp_path: Path):
    module = _load_module()
    plan_path = tmp_path / "plan.jsonl"
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    status_path = chroma_dir / "index_status.json"
    summary_path = tmp_path / "rebuild_summary.json"
    pilot_path = tmp_path / "pilot.jsonl"
    config_path = tmp_path / "config.json"
    plan_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in _plan_rows()
        ),
        encoding="utf-8",
    )
    status_path.write_text(
        json.dumps(
            {
                "ready": True,
                "expected_sources": ["a.pdf", "b.pdf"],
                "indexed_sources": ["a.pdf", "b.pdf"],
                "missing_sources": [],
            }
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "pdf_count": 2,
                "index_status": {
                    "embedding_provider": "local",
                    "embedding_model": "BAAI/bge-small-zh-v1.5",
                },
            }
        ),
        encoding="utf-8",
    )
    pilot_path.write_text('{"hidden": true}\n', encoding="utf-8")
    config = {
        **_config(),
        "config_version": "validation40-retrieval-execution-config-v0.1",
        "retrieval_plan_path": plan_path.name,
        "output_dir": "outputs",
        "cache_dir": "cache",
        "chroma_persist_dir": chroma_dir.name,
        "index_status_path": "chroma/index_status.json",
        "index_rebuild_summary_path": summary_path.name,
        "pilot_test_path": pilot_path.name,
        "expected_retrieval_plan_sha256": module.compute_sha256(plan_path),
        "expected_index_status_sha256": module.compute_sha256(status_path),
        "expected_index_rebuild_summary_sha256": module.compute_sha256(summary_path),
        "expected_pilot_test_sha256": module.compute_sha256(pilot_path),
        "expected_index_source_count": 2,
        "expected_embedding_provider": "local",
        "expected_embedding_model": "BAAI/bge-small-zh-v1.5",
        "execute_retrieval": True,
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    first_retriever = _RecordingRetriever()

    first = module.run(
        config_path,
        repo_root=tmp_path,
        retriever=first_retriever,
        runtime_embedding_provider="local",
        runtime_embedding_model="BAAI/bge-small-zh-v1.5",
        runtime_chroma_persist_dir=chroma_dir,
    )
    assert len(first_retriever.calls) == 2
    assert first["logical_retrieval_task_count"] == 4
    assert first["physical_retrieval_count"] == 2
    assert first["pilot_test_sha256_before"] == first["pilot_test_sha256_after"]
    assert len(first["output_sha256"]) == 5

    second_retriever = _RecordingRetriever()
    second = module.run(
        config_path,
        repo_root=tmp_path,
        retriever=second_retriever,
        runtime_embedding_provider="local",
        runtime_embedding_model="BAAI/bge-small-zh-v1.5",
        runtime_chroma_persist_dir=chroma_dir,
    )
    assert second_retriever.calls == []
    assert second["output_sha256"] == first["output_sha256"]
