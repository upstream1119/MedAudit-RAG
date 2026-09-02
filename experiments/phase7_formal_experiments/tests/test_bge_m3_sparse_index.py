from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "bge_m3_sparse_index.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "BGE-M3 sparse index module is missing"
    spec = importlib.util.spec_from_file_location("bge_m3_sparse_index", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeCollection:
    def get(self, include):
        assert include == ["documents", "metadatas"]
        return {
            "ids": ["b", "a"],
            "documents": ["second", "first"],
            "metadatas": [
                {"source_file": "B.pdf", "page_number": 2, "granularity": 128},
                {"source_file": "A.pdf", "page_number": 1, "granularity": 128},
            ],
        }


def test_exports_chroma_rows_in_deterministic_id_order():
    module = _load_module()

    rows = module.export_chroma_rows(_FakeCollection(), "detail_128")

    assert [row["document_id"] for row in rows] == ["a", "b"]
    assert rows[0]["content"] == "first"
    assert rows[0]["collection"] == "detail_128"
    assert rows[0]["source_file"] == "A.pdf"
    assert rows[0]["page_number"] == 1


def test_converts_lexical_weights_to_csr_without_losing_token_columns():
    module = _load_module()

    matrix = module.lexical_weights_to_csr(
        [{"3": 0.5, "7": np.float16(0.25)}, {7: 0.75}], vocab_size=10
    )

    assert matrix.shape == (2, 10)
    assert matrix.dtype == np.float32
    assert matrix.nnz == 3
    assert matrix[0, 3] == pytest.approx(0.5)
    assert matrix[0, 7] == pytest.approx(0.25)
    assert matrix[1, 7] == pytest.approx(0.75)


def test_audit_fails_closed_for_empty_vectors_or_duplicate_ids():
    module = _load_module()
    rows = [
        {"document_id": "dup", "source_file": "A.pdf", "page_number": 1},
        {"document_id": "dup", "source_file": "B.pdf", "page_number": 2},
    ]
    matrix = module.lexical_weights_to_csr([{"1": 0.5}, {}], vocab_size=4)

    audit = module.audit_sparse_index(
        rows_by_collection={"detail_128": rows},
        matrices_by_collection={"detail_128": matrix},
        expected_collections=["detail_128"],
        expected_sources={"A.pdf", "B.pdf"},
    )

    assert audit["ready"] is False
    assert audit["empty_vector_count"] == 1
    assert audit["duplicate_document_id_count"] == 1
    assert audit["missing_sources"] == []


def test_encodes_each_collection_in_one_model_call_and_checks_row_count():
    module = _load_module()

    class _RecordingEncoder:
        def __init__(self):
            self.calls = []

        def encode(self, texts, **kwargs):
            self.calls.append((texts, kwargs))
            return {"lexical_weights": [{"1": 0.2}, {"2": 0.3}]}

    encoder = _RecordingEncoder()
    weights = module.encode_collection_weights(
        encoder=encoder,
        rows=[{"content": "first"}, {"content": "second"}],
        batch_size=4,
        max_length=512,
        collection_name="detail_128",
    )

    assert weights == [{"1": 0.2}, {"2": 0.3}]
    assert len(encoder.calls) == 1
    assert encoder.calls[0][0] == ["first", "second"]
    assert encoder.calls[0][1]["batch_size"] == 4
