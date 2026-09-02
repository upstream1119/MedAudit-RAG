from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "validation40_retrieval_evaluation.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "Validation40 retrieval evaluation module is missing"
    spec = importlib.util.spec_from_file_location(
        "validation40_retrieval_evaluation", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gold() -> dict:
    return {
        "candidate_id": "PMSQA-VALIDATION-001",
        "source_filename": "儿童社区获得性肺炎诊疗规范（2019年版）.pdf",
        "page_number": 26,
        "anchor_text_span": "治疗后48至72小时症状无改善时应再次评估。",
    }


def _retrieval() -> dict:
    return {
        "physical_retrieval_key": "physical-key",
        "profile": {
            "sample_id": "PMSQA-VALIDATION-001",
            "profile": "single_granularity",
            "top_k": 4,
        },
        "status": "completed",
        "evidence": [
            {
                "source_file": "其他资料.pdf",
                "page_number": 3,
                "content": "无关证据。",
            },
            {
                "source_file": "儿童社区获得性肺炎诊疗规范（2019年版）.pdf",
                "page_number": 26,
                "content": "治疗后48至72小时症状无改善时应再次评估。",
            },
            {
                "source_file": "儿童社区获得性肺炎诊疗规范（2019年版）.pdf",
                "page_number": 25,
                "content": "相邻页面证据。",
            },
        ],
    }


def test_evaluates_exact_gold_source_page_hit_and_rank():
    module = _load_module()

    result = module.evaluate_physical_result(
        _gold(),
        _retrieval(),
        {"adjacent_page_tolerance": 1, "text_ngram_size": 3},
    )

    assert result["gold_source_hit"] is True
    assert result["gold_source_page_hit"] is True
    assert result["gold_source_page_rank"] == 2
    assert result["gold_source_page_reciprocal_rank"] == 0.5
    assert result["gold_source_precision_at_k"] == 2 / 3
    assert result["gold_source_page_precision_at_k"] == 1 / 3
    assert result["failure_type"] == "gold_source_page_hit"


def test_adjacent_page_hit_is_diagnostic_only():
    module = _load_module()
    retrieval = _retrieval()
    retrieval["evidence"] = [
        {
            "source_file": _gold()["source_filename"],
            "page_number": 25,
            "content": "治疗后48至72小时症状无改善时应再次评估。",
        }
    ]

    result = module.evaluate_physical_result(
        _gold(),
        retrieval,
        {"adjacent_page_tolerance": 1, "text_ngram_size": 3},
    )

    assert result["gold_source_hit"] is True
    assert result["gold_source_page_hit"] is False
    assert result["adjacent_gold_source_page_hit"] is True
    assert result["failure_type"] == "adjacent_gold_page_only"


def test_reports_lexical_anchor_coverage_and_redundant_pair_rate():
    module = _load_module()
    retrieval = _retrieval()
    retrieval["evidence"] = [
        {
            "source_file": _gold()["source_filename"],
            "page_number": 26,
            "content": "治疗后48至72小时症状无改善时，应再次评估。",
        },
        {
            "source_file": _gold()["source_filename"],
            "page_number": 26,
            "content": "治疗后48至72小时症状无改善时应再次评估。",
        },
        {
            "source_file": "其他资料.pdf",
            "page_number": 3,
            "content": "完全无关的文本。",
        },
    ]

    result = module.evaluate_physical_result(
        _gold(),
        retrieval,
        {
            "adjacent_page_tolerance": 1,
            "text_ngram_size": 3,
            "redundancy_jaccard_threshold": 0.8,
        },
    )

    assert result["max_anchor_lexical_coverage"] == 1.0
    assert result["redundant_pair_count"] == 1
    assert result["evidence_pair_count"] == 3
    assert result["redundant_pair_rate"] == 1 / 3


def test_builds_two_profile_summary_without_fake_method_comparison():
    module = _load_module()
    single = _retrieval()
    multi = _retrieval()
    multi["physical_retrieval_key"] = "multi-key"
    multi["profile"] = {
        "sample_id": "PMSQA-VALIDATION-001",
        "profile": "multi_granularity",
        "top_k": 4,
    }
    multi["status"] = "insufficient_evidence"
    multi["evidence"] = []

    result = module.build_retrieval_evaluation(
        [_gold()],
        [single, multi],
        {
            "expected_sample_count": 1,
            "expected_physical_result_count": 2,
            "expected_profiles": ["single_granularity", "multi_granularity"],
            "adjacent_page_tolerance": 1,
            "text_ngram_size": 3,
            "redundancy_jaccard_threshold": 0.8,
        },
    )

    assert len(result["sample_metrics"]) == 2
    summaries = {row["profile"]: row for row in result["profile_summary"]}
    assert summaries["single_granularity"]["gold_source_page_recall_at_k"] == 1.0
    assert summaries["multi_granularity"]["gold_source_page_recall_at_k"] == 0.0
    assert summaries["multi_granularity"]["insufficient_evidence_rate"] == 1.0
    assert result["audit"]["profile_count"] == 2
    assert result["audit"]["method_comparison_generated"] is False
    assert result["audit"]["external_model_calls"] == 0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_run_locks_inputs_and_writes_deterministic_immutable_outputs(tmp_path: Path):
    module = _load_module()
    gold_path = tmp_path / "gold.jsonl"
    physical_path = tmp_path / "physical.jsonl"
    task_path = tmp_path / "tasks.jsonl"
    retrieval_audit_path = tmp_path / "retrieval_audit.json"
    config_path = tmp_path / "config.json"
    output_dir = tmp_path / "outputs"

    single = _retrieval()
    multi = _retrieval()
    multi["physical_retrieval_key"] = "multi-key"
    multi["profile"] = {
        "sample_id": "PMSQA-VALIDATION-001",
        "profile": "multi_granularity",
        "top_k": 4,
    }
    multi["status"] = "insufficient_evidence"
    multi["evidence"] = []
    _write_jsonl(gold_path, [_gold()])
    _write_jsonl(physical_path, [single, multi])
    _write_jsonl(
        task_path,
        [
            {
                "sample_id": "PMSQA-VALIDATION-001",
                "method_id": method,
                "physical_retrieval_key": (
                    "physical-key" if method == "naive_rag" else "multi-key"
                ),
                "graph_reranking_executed": False,
            }
            for method in (
                "naive_rag",
                "multi_granularity_rag",
                "trust_gated_rag",
                "graph_enhanced_full",
            )
        ],
    )
    retrieval_audit_path.write_text(
        json.dumps(
            {
                "sample_count": 1,
                "physical_retrieval_count": 2,
                "logical_retrieval_task_count": 4,
                "external_model_calls": 0,
                "graph_reranking_executed": False,
                "pilot_test_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    config = {
        "evaluation_version": "validation40-retrieval-evaluation-v0.1",
        "gold_path": gold_path.name,
        "physical_results_path": physical_path.name,
        "task_results_path": task_path.name,
        "retrieval_audit_path": retrieval_audit_path.name,
        "output_dir": output_dir.name,
        "expected_gold_sha256": module.compute_sha256(gold_path),
        "expected_physical_results_sha256": module.compute_sha256(physical_path),
        "expected_task_results_sha256": module.compute_sha256(task_path),
        "expected_retrieval_audit_sha256": module.compute_sha256(
            retrieval_audit_path
        ),
        "expected_sample_count": 1,
        "expected_physical_result_count": 2,
        "expected_logical_task_count": 4,
        "expected_profiles": ["single_granularity", "multi_granularity"],
        "expected_methods": [
            "naive_rag",
            "multi_granularity_rag",
            "trust_gated_rag",
            "graph_enhanced_full",
        ],
        "adjacent_page_tolerance": 1,
        "text_ngram_size": 3,
        "redundancy_jaccard_threshold": 0.8,
        "execute_retrieval": False,
        "execute_model_calls": False,
        "graph_reranking_executed": False,
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    first = module.run(config_path, repo_root=tmp_path)
    second = module.run(config_path, repo_root=tmp_path)

    assert first["output_sha256"] == second["output_sha256"]
    assert len(first["output_sha256"]) == 5
    assert first["audit"]["gold_join_stage"] == "post_retrieval_evaluation_only"
    assert first["audit"]["pilot_test_read"] is False
    assert first["audit"]["external_model_calls"] == 0
    assert first["audit"]["logical_method_task_counts"] == {
        "graph_enhanced_full": 1,
        "multi_granularity_rag": 1,
        "naive_rag": 1,
        "trust_gated_rag": 1,
    }
