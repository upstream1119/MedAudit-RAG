import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "run_generation_calls.py"
)


def _load_runner():
    assert RUNNER_PATH.exists(), "Phase 7 generation runner is not implemented"
    spec = importlib.util.spec_from_file_location("phase7_generation_runner", RUNNER_PATH)
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


def _source_bundle(tmp_path: Path) -> tuple[Path, list[dict]]:
    source_dir = tmp_path / "source"
    rows = []
    for method_id, cache_key in [
        ("vector_only_rag", "vector-key"),
        ("graph_enhanced_rag", "graph-key"),
    ]:
        rows.append(
            {
                "sample_id": "PMSQA_DEV_001",
                "method_id": method_id,
                "model_provider": "dashscope",
                "model_name": "qwen-test",
                "prompt_version": "phase6b-generator-v0.1",
                "dataset_version": "dev50-v1.0",
                "kb_version": "KB-medium-v1",
                "cache_key": cache_key,
                "evidence_context_source": method_id,
                "evidence_snippet_count": 2,
                "question": "测试问题",
                "prompt": f"{method_id} 测试提示",
            }
        )

    _write_jsonl(source_dir / "prompts.jsonl", rows)
    _write_jsonl(
        source_dir / "call_plan.jsonl",
        [
            {
                **{
                    key: row[key]
                    for key in [
                        "sample_id",
                        "method_id",
                        "model_provider",
                        "model_name",
                        "prompt_version",
                        "dataset_version",
                        "kb_version",
                        "cache_key",
                        "evidence_context_source",
                        "evidence_snippet_count",
                    ]
                },
                "estimated_input_tokens": 100,
                "estimated_output_tokens": 30,
                "estimated_cost_cny": 0.0,
            }
            for row in rows
        ],
    )
    _write_jsonl(
        source_dir / "evaluation_metadata.jsonl",
        [
            {
                "sample_id": row["sample_id"],
                "method_id": row["method_id"],
                "cache_key": row["cache_key"],
                "dataset_version": row["dataset_version"],
                "kb_version": row["kb_version"],
                "prompt_version": row["prompt_version"],
            }
            for row in rows
        ],
    )
    return source_dir, rows


def _config(tmp_path: Path, source_dir: Path) -> dict:
    return {
        "config_version": "phase7-test-v0.1",
        "run_id_prefix": "phase7_test",
        "source_run_dir": str(source_dir),
        "output_root": str(tmp_path / "runs"),
        "cache_dir": str(tmp_path / "cache"),
        "execute_model_calls": False,
        "api_key_env": "PHASE7_TEST_API_KEY",
        "api_base_url": "https://example.invalid/v1/chat/completions",
        "request_timeout_seconds": 1,
        "max_retries": 0,
        "temperature": 0.0,
        "max_output_tokens": 30,
        "max_planned_calls": 2,
        "max_estimated_input_tokens": 300,
        "max_estimated_output_tokens": 60,
        "models": [
            {
                "model_provider": "dashscope",
                "model_name": "qwen-test",
                "price_input_per_1m_cny": 0.0,
                "price_output_per_1m_cny": 0.0,
            }
        ],
    }


def test_default_run_is_non_executing_and_preserves_all_artifacts(tmp_path):
    runner = _load_runner()
    source_dir, _ = _source_bundle(tmp_path)

    summary = runner.run_generation_calls(_config(tmp_path, source_dir))

    assert summary["selected_calls"] == 2
    assert summary["external_model_calls"] == 0
    assert summary["status_counts"] == {"planned_not_executed": 2}
    output_dir = Path(summary["output_dir"])
    assert (output_dir / "raw_model_outputs.jsonl").exists()
    assert (output_dir / "evaluation_metadata.jsonl").exists()
    assert (output_dir / "token_usage_actual.csv").exists()
    assert (output_dir / "run_config_effective.json").exists()


def test_bundle_cache_keys_must_match_before_any_call(tmp_path):
    runner = _load_runner()
    source_dir, _ = _source_bundle(tmp_path)
    metadata_path = source_dir / "evaluation_metadata.jsonl"
    metadata = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    metadata.pop()
    _write_jsonl(metadata_path, metadata)

    with pytest.raises(ValueError, match="cache_key sets differ"):
        runner.run_generation_calls(_config(tmp_path, source_dir))


def test_execute_mode_reuses_cache_without_external_call(tmp_path, monkeypatch):
    runner = _load_runner()
    source_dir, rows = _source_bundle(tmp_path)
    config = _config(tmp_path, source_dir)
    cache_dir = Path(config["cache_dir"])
    cache_dir.mkdir(parents=True)
    for row in rows:
        (cache_dir / f"{row['cache_key']}.json").write_text(
            json.dumps(
                {
                    "cache_key": row["cache_key"],
                    "raw_output": "缓存回答",
                    "raw_response": {
                        "choices": [{"message": {"content": "缓存回答"}}],
                        "usage": {
                            "prompt_tokens": 12,
                            "completion_tokens": 4,
                            "total_tokens": 16,
                        },
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _forbidden_call(*args, **kwargs):
        raise AssertionError("cache hit must not call the external API")

    monkeypatch.setattr(runner.phase5_runtime, "call_chat_completion", _forbidden_call)
    summary = runner.run_generation_calls(config, execute=True)

    assert summary["external_model_calls"] == 0
    assert summary["status_counts"] == {"cache_hit": 2}
    assert summary["input_tokens_total"] == 24
    assert summary["output_tokens_total"] == 8
    output_rows = [
        json.loads(line)
        for line in (Path(summary["output_dir"]) / "raw_model_outputs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["latency_ms"] for row in output_rows] == [None, None]
    assert {row["latency_source"] for row in output_rows} == {
        "unavailable_legacy_cache"
    }


def test_retry_failed_selects_only_failed_cache_keys(tmp_path):
    runner = _load_runner()
    source_dir, rows = _source_bundle(tmp_path)
    failed_dir = tmp_path / "failed_run"
    _write_jsonl(
        failed_dir / "failed_cases.jsonl",
        [{"cache_key": rows[1]["cache_key"], "status": "failed"}],
    )

    summary = runner.run_generation_calls(
        _config(tmp_path, source_dir),
        retry_failed_from=failed_dir,
    )

    assert summary["selected_calls"] == 1
    assert summary["status_counts"] == {"planned_not_executed": 1}


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return b'{"choices":[{"message":{"content":"ok"}}]}'


def test_provider_request_body_extra_is_forwarded_safely(tmp_path):
    runner = _load_runner()
    source_dir, _ = _source_bundle(tmp_path)
    config = _config(tmp_path, source_dir)
    config["request_body_extra"] = {"thinking": {"type": "disabled"}}
    model = config["models"][0]

    with patch.object(
        runner.phase5_runtime.urllib.request,
        "urlopen",
        return_value=_FakeResponse(),
    ) as mocked_urlopen:
        runner.phase5_runtime.call_chat_completion(config, model, "test", "secret")

    request = mocked_urlopen.call_args.args[0]
    payload = json.loads(request.data.decode("utf-8"))
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["model"] == "qwen-test"


def test_provider_request_body_extra_cannot_override_core_fields(tmp_path):
    runner = _load_runner()
    source_dir, _ = _source_bundle(tmp_path)
    config = _config(tmp_path, source_dir)
    config["request_body_extra"] = {"model": "unexpected-model"}

    with pytest.raises(ValueError, match="request_body_extra cannot override"):
        runner.phase5_runtime.call_chat_completion(
            config,
            config["models"][0],
            "test",
            "secret",
        )


def test_external_calls_record_latency_attempts_and_csv(tmp_path, monkeypatch):
    runner = _load_runner()
    source_dir, rows = _source_bundle(tmp_path)
    config = _config(tmp_path, source_dir)
    monkeypatch.setenv(config["api_key_env"], "test-key")
    monkeypatch.setattr(
        runner.phase5_runtime,
        "call_chat_completion",
        lambda *args, **kwargs: {
            "choices": [{"message": {"content": "测试回答"}}],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 4,
                "total_tokens": 16,
            },
        },
    )
    perf_values = iter([10.0, 10.125, 20.0, 20.25])
    monkeypatch.setattr(
        runner.phase5_runtime.time,
        "perf_counter",
        lambda: next(perf_values),
    )

    summary = runner.run_generation_calls(
        config,
        execute=True,
        confirm_external_call=True,
    )

    output_dir = Path(summary["output_dir"])
    raw_rows = [
        json.loads(line)
        for line in (output_dir / "raw_model_outputs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["latency_ms"] for row in raw_rows] == [125.0, 250.0]
    assert {row["latency_source"] for row in raw_rows} == {
        "measured_external_call"
    }
    assert [row["attempt_count"] for row in raw_rows] == [1, 1]

    cached = json.loads(
        (Path(config["cache_dir"]) / f"{rows[0]['cache_key']}.json").read_text(
            encoding="utf-8"
        )
    )
    assert cached["latency_ms"] == 125.0
    assert cached["attempt_count"] == 1

    csv_text = (output_dir / "token_usage_actual.csv").read_text(
        encoding="utf-8-sig"
    )
    assert "latency_ms" in csv_text.splitlines()[0]
    assert "latency_source" in csv_text.splitlines()[0]
    assert "attempt_count" in csv_text.splitlines()[0]
