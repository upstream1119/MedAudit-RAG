from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "validation40_hybrid_retrieval.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "Validation40 hybrid retrieval module is missing"
    spec = importlib.util.spec_from_file_location(
        "validation40_hybrid_retrieval", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeDenseCollection:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    def query(self, **kwargs):
        assert kwargs["n_results"] == 2
        assert kwargs["include"] == ["documents", "metadatas", "distances"]
        return {
            "ids": [["doc-b", "doc-a"]],
            "documents": [["second", "first"]],
            "metadatas": [[
                {"source_file": "B.pdf", "page_number": 2},
                {"source_file": "A.pdf", "page_number": 1},
            ]],
            "distances": [[0.1, 0.2]],
        }


def test_dense_retrieval_keeps_collection_scoped_identity_and_rank():
    module = _load_module()

    candidates = module.retrieve_dense(
        collections={"detail_128": _FakeDenseCollection("detail_128")},
        query_embedding=np.asarray([0.1, 0.2], dtype=np.float32),
        top_k=2,
    )

    assert [item["candidate_key"] for item in candidates] == [
        "detail_128::doc-b",
        "detail_128::doc-a",
    ]
    assert candidates[0]["route"] == "dense:detail_128"
    assert candidates[0]["route_rank"] == 1
    assert candidates[0]["raw_score"] == pytest.approx(0.9)


def test_sparse_retrieval_scores_query_against_each_collection():
    module = _load_module()
    matrix = sparse.csr_matrix(
        np.asarray([[1.0, 0.0, 0.5], [0.0, 2.0, 0.0], [0.5, 0.0, 0.0]])
    )
    query = sparse.csr_matrix(np.asarray([[1.0, 0.0, 1.0]]))
    rows = [
        {"document_id": "a", "content": "A", "source_file": "A.pdf", "page_number": 1},
        {"document_id": "b", "content": "B", "source_file": "B.pdf", "page_number": 2},
        {"document_id": "c", "content": "C", "source_file": "C.pdf", "page_number": 3},
    ]

    candidates = module.retrieve_sparse(
        matrices={"detail_128": matrix},
        documents={"detail_128": rows},
        query_vector=query,
        top_k=2,
    )

    assert [item["candidate_key"] for item in candidates] == [
        "detail_128::a",
        "detail_128::c",
    ]
    assert candidates[0]["route"] == "sparse:detail_128"
    assert candidates[0]["raw_score"] == pytest.approx(1.5)


def test_rrf_merges_only_same_collection_document_and_preserves_route_trace():
    module = _load_module()
    candidates = [
        {
            "candidate_key": "detail_128::a",
            "document_id": "a",
            "collection": "detail_128",
            "content": "alpha",
            "source_file": "A.pdf",
            "page_number": 1,
            "route": "dense:detail_128",
            "route_rank": 1,
            "raw_score": 0.8,
        },
        {
            "candidate_key": "detail_128::a",
            "document_id": "a",
            "collection": "detail_128",
            "content": "alpha",
            "source_file": "A.pdf",
            "page_number": 1,
            "route": "sparse:detail_128",
            "route_rank": 2,
            "raw_score": 3.0,
        },
        {
            "candidate_key": "concept_512::a",
            "document_id": "a",
            "collection": "concept_512",
            "content": "different granularity",
            "source_file": "A.pdf",
            "page_number": 1,
            "route": "dense:concept_512",
            "route_rank": 1,
            "raw_score": 0.9,
        },
    ]

    fused = module.reciprocal_rank_fusion(candidates, rrf_k=60, top_k=20)

    assert len(fused) == 2
    assert fused[0]["candidate_key"] == "detail_128::a"
    assert fused[0]["rrf_score"] == pytest.approx(1 / 61 + 1 / 62)
    assert {trace["route"] for trace in fused[0]["route_traces"]} == {
        "dense:detail_128",
        "sparse:detail_128",
    }


def test_reranker_sorts_by_cross_encoder_score_and_keeps_pre_rank():
    module = _load_module()

    class _Scorer:
        def predict(self, pairs, **kwargs):
            assert pairs == [["question", "first"], ["question", "second"]]
            return np.asarray([0.2, 0.9], dtype=np.float32)

    candidates = [
        {"candidate_key": "d::1", "content": "first", "rrf_score": 0.5},
        {"candidate_key": "d::2", "content": "second", "rrf_score": 0.4},
    ]

    reranked = module.rerank_candidates(
        question="question",
        candidates=candidates,
        scorer=_Scorer(),
        batch_size=2,
    )

    assert [item["candidate_key"] for item in reranked] == ["d::2", "d::1"]
    assert reranked[0]["pre_rerank_rank"] == 2
    assert reranked[0]["post_rerank_rank"] == 1
    assert reranked[0]["reranker_score"] == pytest.approx(0.9)


def test_dedup_removes_exact_and_same_page_overlap_without_padding():
    module = _load_module()
    candidates = [
        {
            "candidate_key": "d::1",
            "content": "治疗后 48-72 小时症状无改善，应再次评估。",
            "source_file": "A.pdf",
            "page_number": 10,
        },
        {
            "candidate_key": "c::2",
            "content": "治疗后48-72小时症状无改善应再次评估",
            "source_file": "A.pdf",
            "page_number": 10,
        },
        {
            "candidate_key": "x::3",
            "content": "治疗后 48-72 小时症状无改善，应再次评估并分析原因。",
            "source_file": "A.pdf",
            "page_number": 10,
        },
        {
            "candidate_key": "x::4",
            "content": "另一条独立证据。",
            "source_file": "B.pdf",
            "page_number": 3,
        },
    ]

    evidence, audit = module.deduplicate_evidence(
        candidates,
        max_evidence=4,
        ngram_size=3,
        overlap_threshold=0.75,
    )

    assert [item["candidate_key"] for item in evidence] == ["d::1", "x::4"]
    assert audit["input_count"] == 4
    assert audit["exact_duplicate_count"] == 1
    assert audit["same_page_overlap_count"] == 1
    assert audit["output_count"] == 2


def test_sparse_query_vector_uses_token_ids_and_python_numeric_values():
    module = _load_module()

    vector = module.lexical_weights_to_query_csr(
        {"2": np.float16(0.5), 5: 1.25}, vocab_size=8
    )

    assert vector.shape == (1, 8)
    assert vector[0, 2] == pytest.approx(0.5)
    assert vector[0, 5] == pytest.approx(1.25)


def test_saves_real_dense_and_sparse_query_artifacts(tmp_path):
    module = _load_module()
    dense = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    sparse_queries = [
        sparse.csr_matrix(np.asarray([[1.0, 0.0, 0.5]], dtype=np.float32)),
        sparse.csr_matrix(np.asarray([[0.0, 2.0, 0.0]], dtype=np.float32)),
    ]

    artifacts = module.save_query_artifacts(
        output_dir=tmp_path,
        dense_embeddings=dense,
        sparse_queries=sparse_queries,
    )

    loaded_dense = np.load(artifacts["dense_path"])
    loaded_sparse = sparse.load_npz(artifacts["sparse_path"])
    assert np.array_equal(loaded_dense, dense)
    assert loaded_sparse.shape == (2, 3)
    assert loaded_sparse[0, 2] == pytest.approx(0.5)
    assert loaded_sparse[1, 1] == pytest.approx(2.0)


def test_validates_frozen_hashes_without_parsing_pilot_content(tmp_path):
    module = _load_module()
    runtime = tmp_path / "runtime.jsonl"
    dense_status = tmp_path / "index_status.json"
    sparse_manifest = tmp_path / "manifest.json"
    pilot = tmp_path / "pilot.jsonl"
    runtime.write_text('{"sample_id":"x"}\n', encoding="utf-8")
    dense_status.write_text('{"ready":true}\n', encoding="utf-8")
    sparse_manifest.write_text('{"ready":true}\n', encoding="utf-8")
    pilot.write_bytes(b"not-json-and-must-not-be-parsed")
    config = {
        "expected_runtime_projection_sha256": module.sha256_file(runtime),
        "expected_dense_index_status_sha256": module.sha256_file(dense_status),
        "expected_sparse_index_manifest_sha256": module.sha256_file(sparse_manifest),
        "expected_pilot_test_sha256": module.sha256_file(pilot),
    }

    audit = module.validate_frozen_hashes(
        config=config,
        runtime_projection_path=runtime,
        dense_status_path=dense_status,
        sparse_manifest_path=sparse_manifest,
        pilot_test_path=pilot,
    )

    assert audit["pilot_test_accessed"] is False
    assert audit["pilot_test_sha256_before"] == module.sha256_file(pilot)


def test_frozen_hash_validation_fails_closed_on_mismatch(tmp_path):
    module = _load_module()
    paths = []
    for name in ["runtime", "dense", "sparse", "pilot"]:
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        paths.append(path)
    config = {
        "expected_runtime_projection_sha256": "0" * 64,
        "expected_dense_index_status_sha256": module.sha256_file(paths[1]),
        "expected_sparse_index_manifest_sha256": module.sha256_file(paths[2]),
        "expected_pilot_test_sha256": module.sha256_file(paths[3]),
    }

    with pytest.raises(ValueError, match="runtime projection SHA-256 mismatch"):
        module.validate_frozen_hashes(
            config=config,
            runtime_projection_path=paths[0],
            dense_status_path=paths[1],
            sparse_manifest_path=paths[2],
            pilot_test_path=paths[3],
        )
