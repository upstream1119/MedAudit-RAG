from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "exact_dense_retrieval.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "Exact dense retrieval module is missing"
    spec = importlib.util.spec_from_file_location("exact_dense_retrieval", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_top_k_is_stable_across_input_order_and_distance_ties():
    module = _load_module()
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    keys = np.asarray(["candidate-b", "candidate-a", "candidate-c"])
    query = np.asarray([1.0, 1.0], dtype=np.float32)

    first = module.rank_exact_top_k(
        embeddings=embeddings,
        candidate_keys=keys,
        query_embedding=query,
        top_k=2,
    )
    permutation = np.asarray([2, 0, 1])
    second = module.rank_exact_top_k(
        embeddings=embeddings[permutation],
        candidate_keys=keys[permutation],
        query_embedding=query,
        top_k=2,
    )

    assert [item["candidate_key"] for item in first] == [
        "candidate-a",
        "candidate-b",
    ]
    assert second == first


def test_retrieve_exact_dense_returns_auditable_candidates_per_collection():
    module = _load_module()
    assets = {
        "detail_128": {
            "embeddings": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            "rows": [
                {
                    "candidate_key": "detail_128::d2",
                    "document_id": "d2",
                    "content": "second",
                    "source_file": "B.pdf",
                    "page_number": 2,
                    "granularity": 128,
                },
                {
                    "candidate_key": "detail_128::d1",
                    "document_id": "d1",
                    "content": "first",
                    "source_file": "A.pdf",
                    "page_number": 1,
                    "granularity": 128,
                },
            ],
        },
        "concept_512": {
            "embeddings": np.asarray([[0.5, 0.5]], dtype=np.float32),
            "rows": [
                {
                    "candidate_key": "concept_512::c1",
                    "document_id": "c1",
                    "content": "concept",
                    "source_file": "C.pdf",
                    "page_number": 3,
                    "granularity": 512,
                }
            ],
        },
    }

    candidates = module.retrieve_exact_dense(
        assets=assets,
        query_embedding=np.asarray([0.0, 1.0], dtype=np.float32),
        top_k=1,
    )

    assert [item["candidate_key"] for item in candidates] == [
        "detail_128::d1",
        "concept_512::c1",
    ]
    assert candidates[0]["route"] == "dense_exact:detail_128"
    assert candidates[0]["route_rank"] == 1
    assert candidates[0]["source_file"] == "A.pdf"
    assert candidates[0]["page_number"] == 1
