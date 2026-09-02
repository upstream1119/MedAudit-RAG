from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "validate_local_retrieval_environment.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "Local retrieval environment validator is missing"
    spec = importlib.util.spec_from_file_location(
        "validate_local_retrieval_environment", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeSparseEncoder:
    def encode(self, texts, **kwargs):
        assert texts == ["阿奇霉素每日一次", "治疗后需要再次评估"]
        assert kwargs["return_dense"] is False
        assert kwargs["return_sparse"] is True
        assert kwargs["return_colbert_vecs"] is False
        return {
            "dense_vecs": None,
            "lexical_weights": [{"11": 0.8}, {"17": 0.6, "23": 0.4}],
            "colbert_vecs": None,
        }


def test_builds_auditable_sparse_smoke_report_without_network_calls():
    module = _load_module()

    report = module.build_sparse_smoke_report(
        encoder=_FakeSparseEncoder(),
        texts=["阿奇霉素每日一次", "治疗后需要再次评估"],
        model_path="D:/AI_Models/huggingface/BAAI/bge-m3",
        device="cuda:0",
        versions={
            "torch": "2.12.1+cu130",
            "transformers": "5.12.1",
            "flag_embedding": "1.4.0",
        },
        cuda_available=True,
        gpu_name="NVIDIA GeForce RTX 4060 Laptop GPU",
        offline_environment={
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )

    assert report["status"] == "passed"
    assert report["text_count"] == 2
    assert report["non_empty_sparse_vector_count"] == 2
    assert report["lexical_weight_counts"] == [1, 2]
    assert report["cuda_available"] is True
    assert report["offline_loading_verified"] is True
    assert report["external_model_calls"] == 0
    assert report["input_tokens"] == 0
    assert report["output_tokens"] == 0
    assert report["estimated_cost"] == 0.0


def test_fails_when_any_sparse_vector_is_empty():
    module = _load_module()

    class _EmptyEncoder:
        def encode(self, texts, **kwargs):
            return {"lexical_weights": [{}, {"5": 0.3}]}

    report = module.build_sparse_smoke_report(
        encoder=_EmptyEncoder(),
        texts=["a", "b"],
        model_path="local",
        device="cuda:0",
        versions={},
        cuda_available=True,
        gpu_name="gpu",
        offline_environment={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )

    assert report["status"] == "failed"
    assert report["non_empty_sparse_vector_count"] == 1


def test_report_preview_converts_numpy_weights_to_json_numbers():
    module = _load_module()

    class _NumpyEncoder:
        def encode(self, texts, **kwargs):
            return {"lexical_weights": [{"5": np.float16(0.25)}]}

    report = module.build_sparse_smoke_report(
        encoder=_NumpyEncoder(),
        texts=["a"],
        model_path="local",
        device="cuda:0",
        versions={},
        cuda_available=True,
        gpu_name="gpu",
        offline_environment={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )

    assert json.loads(json.dumps(report))["lexical_weight_preview"] == [{"5": 0.25}]
