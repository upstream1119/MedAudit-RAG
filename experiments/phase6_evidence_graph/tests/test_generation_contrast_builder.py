import json
from pathlib import Path

from experiments.phase6_evidence_graph.generation_contrast_builder import (
    run_builder,
)


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _config() -> dict:
    return {
        "config_version": "test-generation-contrast-v0.1",
        "dataset_version": "dev50-v1.0",
        "kb_version": "KB-medium-v1",
        "prompt_version": "phase6b-generator-v0.1",
        "inference_profile": "qwen-test-nonthinking-v0.1",
        "run_mode": "dry_run_no_api_call",
        "execute_model_calls": False,
        "dataset_path": "revision/benchmark/dev50/dev50_v1_0_frozen.jsonl",
        "input_run_dir": "revision/phase6/graph_runs/phase6b_rerank_frozen15_v0_3_20260720_01",
        "output_root": "revision/phase6/test_tmp",
        "run_id_prefix": "test_generation_contrast",
        "sample_ids": [
            "PMSQA_DEV_001",
            "PMSQA_DEV_002",
            "PMSQA_DEV_003",
        ],
        "max_evidence_snippets": 4,
        "max_evidence_chars_per_snippet": 300,
        "max_output_tokens": 300,
        "methods": [
            {"method_id": "vector_only_rag"},
            {"method_id": "graph_enhanced_rag"},
        ],
        "models": [
            {
                "model_provider": "dashscope",
                "model_name": "qwen3.6-flash-2026-04-16",
                "price_input_per_1m_cny": 0.0,
                "price_output_per_1m_cny": 0.0,
            }
        ],
    }


def test_builder_writes_matched_generation_contrast_artifacts(tmp_path):
    output_dir = tmp_path / "run"
    summary = run_builder(_config(), output_dir=output_dir)

    assert summary["sample_count"] == 3
    assert summary["planned_calls"] == 6
    assert summary["external_model_calls"] == 0
    assert (output_dir / "prompts.jsonl").exists()
    assert (output_dir / "call_plan.jsonl").exists()
    assert (output_dir / "evaluation_metadata.jsonl").exists()
    assert (output_dir / "token_usage_estimate.csv").exists()


def test_prompts_keep_methods_paired_and_gold_labels_out(tmp_path):
    output_dir = tmp_path / "run"
    run_builder(_config(), output_dir=output_dir)

    prompts = _load_jsonl(output_dir / "prompts.jsonl")
    by_sample: dict[str, set[str]] = {}
    for row in prompts:
        by_sample.setdefault(row["sample_id"], set()).add(row["method_id"])
        lowered = row["prompt"].lower()
        assert "expected_decision" not in lowered
        assert "gold" not in lowered
        assert "forbidden_claim" not in lowered
        assert "risk_labels" not in lowered

    assert by_sample == {
        "PMSQA_DEV_001": {"vector_only_rag", "graph_enhanced_rag"},
        "PMSQA_DEV_002": {"vector_only_rag", "graph_enhanced_rag"},
        "PMSQA_DEV_003": {"vector_only_rag", "graph_enhanced_rag"},
    }


def test_boundary_sample_remains_evidence_empty_and_fail_closed(tmp_path):
    output_dir = tmp_path / "run"
    run_builder(_config(), output_dir=output_dir)

    prompts = [
        row for row in _load_jsonl(output_dir / "prompts.jsonl")
        if row["sample_id"] == "PMSQA_DEV_003"
    ]
    assert len(prompts) == 2
    assert all("本轮没有可用证据片段" in row["prompt"] for row in prompts)
    assert all("处方" in row["prompt"] for row in prompts)


def test_call_plan_records_versions_cache_and_token_estimates(tmp_path):
    output_dir = tmp_path / "run"
    run_builder(_config(), output_dir=output_dir)

    rows = _load_jsonl(output_dir / "call_plan.jsonl")
    assert rows
    for row in rows:
        assert row["dataset_version"] == "dev50-v1.0"
        assert row["kb_version"] == "KB-medium-v1"
        assert row["prompt_version"] == "phase6b-generator-v0.1"
        assert row["inference_profile"] == "qwen-test-nonthinking-v0.1"
        assert len(row["cache_key"]) == 64
        assert row["estimated_input_tokens"] > 0
        assert row["estimated_output_tokens"] == 300
        assert row["should_call_model"] is False
        assert row["skip_reason"] == "dry_run_no_api_call"


def test_inference_profile_is_part_of_cache_identity(tmp_path):
    first_config = _config()
    second_config = _config()
    second_config["inference_profile"] = "qwen-test-thinking-v0.1"

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    run_builder(first_config, output_dir=first_dir)
    run_builder(second_config, output_dir=second_dir)

    first_rows = _load_jsonl(first_dir / "prompts.jsonl")
    second_rows = _load_jsonl(second_dir / "prompts.jsonl")
    assert [row["cache_key"] for row in first_rows] != [
        row["cache_key"] for row in second_rows
    ]
    assert {row["inference_profile"] for row in first_rows} == {
        "qwen-test-nonthinking-v0.1"
    }
