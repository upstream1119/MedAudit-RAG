from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "validation40_experiment_preflight.py"
)
CONFIG_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "configs"
    / "validation40_experiment_preflight_v0_1.json"
)


def _load_module():
    assert MODULE_PATH.exists(), "Validation40 experiment preflight module is missing"
    spec = importlib.util.spec_from_file_location(
        "validation40_experiment_preflight", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runtime_row(index: int) -> dict:
    return {
        "sample_id": f"PMSQA-VALIDATION-{index:03d}",
        "question": f"第 {index} 条儿科用药证据审计问题",
        "dataset_version": "benchmark-v1.0-guideline-grounded-author-adjudicated",
        "kb_version": "KB-medium-v1",
    }


def _methods() -> list[dict]:
    return [
        {
            "method_id": "vanilla_llm",
            "method_version": "vanilla-llm-v0.1",
            "retrieval_required": False,
            "retrieval_mode": "none",
            "retrieval_top_k": 0,
            "use_trust_gate": False,
            "use_graph": False,
        },
        {
            "method_id": "naive_rag",
            "method_version": "naive-rag-v0.1",
            "retrieval_required": True,
            "retrieval_mode": "single_granularity",
            "retrieval_top_k": 4,
            "use_trust_gate": False,
            "use_graph": False,
        },
        {
            "method_id": "multi_granularity_rag",
            "method_version": "multi-granularity-rag-v0.1",
            "retrieval_required": True,
            "retrieval_mode": "multi_granularity",
            "retrieval_top_k": 4,
            "use_trust_gate": False,
            "use_graph": False,
        },
        {
            "method_id": "trust_gated_rag",
            "method_version": "trust-gated-rag-v0.1",
            "retrieval_required": True,
            "retrieval_mode": "multi_granularity",
            "retrieval_top_k": 4,
            "use_trust_gate": True,
            "use_graph": False,
        },
        {
            "method_id": "graph_enhanced_full",
            "method_version": "graph-enhanced-full-v0.1",
            "retrieval_required": True,
            "retrieval_mode": "multi_granularity_graph",
            "retrieval_top_k": 4,
            "use_trust_gate": True,
            "use_graph": True,
        },
    ]


def _config(count: int = 3) -> dict:
    return {
        "config_version": "validation40-experiment-preflight-config-v0.1",
        "preflight_version": "validation40-experiment-preflight-v0.1",
        "expected_runtime_count": count,
        "expected_runtime_sha256": "runtime-projection-hash",
        "expected_dataset_version": (
            "benchmark-v1.0-guideline-grounded-author-adjudicated"
        ),
        "expected_kb_version": "KB-medium-v1",
        "prompt_version": "pediatric-medication-audit-v0.1",
        "seed": 20260817,
        "evidence_context_min": 2,
        "evidence_context_max": 4,
        "execute_retrieval": False,
        "execute_model_calls": False,
        "methods": _methods(),
        "models": [
            {
                "model_provider": "zhipu",
                "model_name": "glm-4.5-air",
                "model_version": "glm-4.5-air",
                "role": "generator",
            }
        ],
    }


def test_expands_five_methods_and_four_retrieval_tracks():
    module = _load_module()
    result = module.build_experiment_preflight(
        [_runtime_row(index) for index in range(1, 4)],
        _config(),
        observed_runtime_sha256="runtime-projection-hash",
    )

    assert len(result["retrieval_plan"]) == 12
    assert len(result["method_call_plan"]) == 15
    assert result["audit"]["runtime_record_count"] == 3
    assert result["audit"]["retrieval_task_count"] == 12
    assert result["audit"]["method_call_count"] == 15
    assert result["audit"]["external_model_calls"] == 0
    assert result["audit"]["pilot_test_accessed"] is False


def test_cache_keys_are_unique_deterministic_and_runtime_safe():
    module = _load_module()
    rows = [_runtime_row(index) for index in range(1, 4)]
    first = module.build_experiment_preflight(
        rows,
        _config(),
        observed_runtime_sha256="runtime-projection-hash",
    )
    second = module.build_experiment_preflight(
        deepcopy(rows),
        deepcopy(_config()),
        observed_runtime_sha256="runtime-projection-hash",
    )

    assert first == second
    retrieval_keys = [row["retrieval_cache_key"] for row in first["retrieval_plan"]]
    call_keys = [row["cache_key"] for row in first["method_call_plan"]]
    assert len(retrieval_keys) == len(set(retrieval_keys))
    assert len(call_keys) == len(set(call_keys))
    vanilla = [
        row for row in first["method_call_plan"] if row["method_id"] == "vanilla_llm"
    ]
    assert all(row["retrieval_cache_key"] is None for row in vanilla)
    assert first["audit"]["gold_field_leakage_count"] == 0


def test_rejects_runtime_rows_with_gold_or_extra_fields():
    module = _load_module()
    rows = [_runtime_row(index) for index in range(1, 4)]
    rows[0]["expected_decision"] = "answer"

    with pytest.raises(ValueError, match="runtime field allowlist"):
        module.build_experiment_preflight(
            rows,
            _config(),
            observed_runtime_sha256="runtime-projection-hash",
        )


@pytest.mark.parametrize(
    ("mutation", "observed_hash", "match"),
    [
        (lambda rows: rows.pop(), "runtime-projection-hash", "record count"),
        (
            lambda rows: rows.__setitem__(1, deepcopy(rows[0])),
            "runtime-projection-hash",
            "duplicate sample_id",
        ),
        (
            lambda rows: rows[0].__setitem__("kb_version", "KB-drifted"),
            "runtime-projection-hash",
            "kb_version",
        ),
        (lambda rows: None, "wrong-hash", "SHA-256"),
    ],
)
def test_rejects_runtime_drift(mutation, observed_hash, match):
    module = _load_module()
    rows = [_runtime_row(index) for index in range(1, 4)]
    mutation(rows)

    with pytest.raises(ValueError, match=match):
        module.build_experiment_preflight(
            rows,
            _config(),
            observed_runtime_sha256=observed_hash,
        )


def test_rejects_incomplete_method_matrix_or_executable_config():
    module = _load_module()
    rows = [_runtime_row(index) for index in range(1, 4)]
    incomplete = _config()
    incomplete["methods"] = incomplete["methods"][:-1]
    executable = _config()
    executable["execute_model_calls"] = True

    with pytest.raises(ValueError, match="method matrix"):
        module.build_experiment_preflight(
            rows,
            incomplete,
            observed_runtime_sha256="runtime-projection-hash",
        )
    with pytest.raises(ValueError, match="non-executing"):
        module.build_experiment_preflight(
            rows,
            executable,
            observed_runtime_sha256="runtime-projection-hash",
        )


def test_write_is_idempotent_and_conflicts_fail_closed(tmp_path):
    module = _load_module()
    result = module.build_experiment_preflight(
        [_runtime_row(index) for index in range(1, 4)],
        _config(),
        observed_runtime_sha256="runtime-projection-hash",
    )

    first = module.write_preflight_outputs(result, tmp_path)
    second = module.write_preflight_outputs(result, tmp_path)
    assert first == second
    assert set(first) == set(module.OUTPUT_FILENAMES.values())

    conflict_path = tmp_path / module.OUTPUT_FILENAMES["method_call_plan"]
    conflict_path.write_text("conflict\n", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable output conflict"):
        module.write_preflight_outputs(result, tmp_path)


def test_repository_config_is_fixed_nonexecuting_and_pilot_free():
    assert CONFIG_PATH.exists(), "Validation40 experiment preflight config is missing"
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["expected_runtime_count"] == 40
    assert config["execute_retrieval"] is False
    assert config["execute_model_calls"] is False
    assert {row["method_id"] for row in config["methods"]} == {
        "vanilla_llm",
        "naive_rag",
        "multi_granularity_rag",
        "trust_gated_rag",
        "graph_enhanced_full",
    }
    assert "pilot" not in json.dumps(config, ensure_ascii=False).lower()
