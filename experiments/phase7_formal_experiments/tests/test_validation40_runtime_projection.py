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
    / "validation40_runtime_projection.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "Validation40 runtime projection module is missing"
    spec = importlib.util.spec_from_file_location(
        "validation40_runtime_projection", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(index: int) -> dict:
    return {
        "candidate_id": f"PMSQA-VALIDATION-{index:03d}",
        "question": f"第 {index} 条指南约束审计问题",
        "dataset_split": "validation",
        "dataset_version": "benchmark-v1.0-guideline-grounded-author-adjudicated",
        "kb_version": "KB-medium-v1",
        "freeze_version": "benchmark-v1.0-freeze-v1.0",
        "freeze_status": "frozen",
        "split_status": "frozen",
        "expected_decision": "answer",
        "required_claims": ["受证据支持的结论"],
        "allowed_claims": ["仅回答证据范围"],
        "forbidden_claims": ["不得外推"],
        "risk_labels": ["evidence_supported"],
        "source_title": "测试指南",
        "page_number": index,
        "anchor_text_span": "仅用于离线评测的 Gold 证据。",
    }


def _config(count: int = 3) -> dict:
    return {
        "config_version": "validation40-runtime-projection-config-v0.1",
        "projection_version": "validation40-runtime-projection-v0.1",
        "expected_count": count,
        "expected_dataset_split": "validation",
        "expected_dataset_version": (
            "benchmark-v1.0-guideline-grounded-author-adjudicated"
        ),
        "expected_kb_version": "KB-medium-v1",
        "expected_freeze_version": "benchmark-v1.0-freeze-v1.0",
        "expected_source_sha256": "validation-source-hash",
        "external_model_calls": 0,
    }


def test_build_projects_exactly_the_runtime_allowlist():
    module = _load_module()
    rows = [_row(index) for index in range(1, 4)]

    result = module.build_runtime_projection(
        rows,
        _config(),
        observed_source_sha256="validation-source-hash",
    )

    assert len(result["runtime_rows"]) == 3
    assert len(result["selection_rows"]) == 3
    assert set(result["runtime_rows"][0]) == {
        "sample_id",
        "question",
        "dataset_version",
        "kb_version",
    }
    assert result["runtime_rows"][0]["sample_id"] == rows[0]["candidate_id"]
    assert result["selection_rows"][0]["selection_rank"] == 1
    assert result["audit"]["gold_field_leakage_count"] == 0
    assert result["audit"]["pilot_test_accessed"] is False
    assert result["audit"]["usage"] == module.zero_usage()


def test_runtime_projection_contains_no_gold_or_evidence_fields():
    module = _load_module()
    result = module.build_runtime_projection(
        [_row(index) for index in range(1, 4)],
        _config(),
        observed_source_sha256="validation-source-hash",
    )

    serialized = json.dumps(result["runtime_rows"], ensure_ascii=False)
    for field in module.GOLD_ONLY_FIELDS:
        assert f'"{field}"' not in serialized
    assert "anchor_text_span" not in serialized
    assert "page_number" not in serialized


def test_build_fails_closed_on_hash_count_split_or_version_drift():
    module = _load_module()
    rows = [_row(index) for index in range(1, 4)]
    config = _config()

    with pytest.raises(ValueError, match="hash"):
        module.build_runtime_projection(
            rows,
            config,
            observed_source_sha256="wrong-hash",
        )
    with pytest.raises(ValueError, match="count"):
        module.build_runtime_projection(
            rows[:-1],
            config,
            observed_source_sha256="validation-source-hash",
        )

    split_drift = deepcopy(rows)
    split_drift[0]["dataset_split"] = "pilot_test"
    with pytest.raises(ValueError, match="dataset_split"):
        module.build_runtime_projection(
            split_drift,
            config,
            observed_source_sha256="validation-source-hash",
        )

    version_drift = deepcopy(rows)
    version_drift[0]["kb_version"] = "KB-drifted"
    with pytest.raises(ValueError, match="kb_version"):
        module.build_runtime_projection(
            version_drift,
            config,
            observed_source_sha256="validation-source-hash",
        )


def test_build_rejects_duplicate_or_unfrozen_rows():
    module = _load_module()
    rows = [_row(index) for index in range(1, 4)]
    duplicate = deepcopy(rows)
    duplicate[1]["candidate_id"] = duplicate[0]["candidate_id"]
    with pytest.raises(ValueError, match="duplicate|重复"):
        module.build_runtime_projection(
            duplicate,
            _config(),
            observed_source_sha256="validation-source-hash",
        )

    unfrozen = deepcopy(rows)
    unfrozen[0]["freeze_status"] = "draft"
    with pytest.raises(ValueError, match="freeze_status"):
        module.build_runtime_projection(
            unfrozen,
            _config(),
            observed_source_sha256="validation-source-hash",
        )


def test_write_outputs_are_deterministic_immutable_and_idempotent(tmp_path: Path):
    module = _load_module()
    result = module.build_runtime_projection(
        [_row(index) for index in range(1, 4)],
        _config(),
        observed_source_sha256="validation-source-hash",
    )

    hashes_first = module.write_projection_outputs(result, tmp_path)
    bytes_first = {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }
    hashes_second = module.write_projection_outputs(result, tmp_path)
    bytes_second = {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }

    assert hashes_first == hashes_second
    assert bytes_first == bytes_second

    runtime_path = tmp_path / "validation40_runtime_projection_v0_1.jsonl"
    runtime_path.write_text("conflicting bytes\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflict|冲突"):
        module.write_projection_outputs(result, tmp_path)


def test_run_reads_only_configured_validation_source(tmp_path: Path, monkeypatch):
    module = _load_module()
    validation_path = tmp_path / "validation40.jsonl"
    pilot_path = tmp_path / "pilot_test80.jsonl"
    validation_path.write_text(
        "".join(
            json.dumps(_row(index), ensure_ascii=False) + "\n"
            for index in range(1, 4)
        ),
        encoding="utf-8",
    )
    pilot_path.write_text("must not be read", encoding="utf-8")

    source_hash = module.compute_sha256(validation_path)
    config = {
        **_config(),
        "expected_source_sha256": source_hash,
        "source_path": validation_path.name,
        "output_dir": "outputs",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    original_read_text = Path.read_text
    observed_paths: list[Path] = []

    def tracked_read_text(path: Path, *args, **kwargs):
        observed_paths.append(path.resolve())
        assert path.resolve() != pilot_path.resolve()
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", tracked_read_text)
    summary = module.run(config_path, repo_root=tmp_path)

    assert summary["record_count"] == 3
    assert summary["pilot_test_accessed"] is False
    assert validation_path.resolve() in observed_paths
    assert pilot_path.resolve() not in observed_paths
