from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "phase7_formal_experiments"
BUILDER_PATH = EXPERIMENT_DIR / "build_bge_m3_exact_dense_assets.py"
RETRIEVER_PATH = EXPERIMENT_DIR / "exact_dense_retrieval.py"


def _load(path: Path, name: str):
    assert path.exists(), f"module is missing: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeCollection:
    def __init__(self):
        self._rows = [
            ("z", [1.0, 0.0], "z text", {"source_file": "Z.pdf", "page_number": 2}),
            ("a", [0.0, 1.0], "a text", {"source_file": "A.pdf", "page_number": 1}),
        ]

    def count(self) -> int:
        return len(self._rows)

    def get(self, *, limit: int, offset: int, include: list[str]):
        selected = self._rows[offset : offset + limit]
        return {
            "ids": [row[0] for row in selected],
            "embeddings": [row[1] for row in selected],
            "documents": [row[2] for row in selected],
            "metadatas": [row[3] for row in selected],
        }


def test_build_and_load_exact_dense_assets_preserves_sorted_row_alignment(tmp_path):
    builder = _load(BUILDER_PATH, "build_bge_m3_exact_dense_assets")
    retriever = _load(RETRIEVER_PATH, "exact_dense_retrieval")
    status_path = tmp_path / "index_status.json"
    status_path.write_text('{"ready": true}\n', encoding="utf-8")
    output_dir = tmp_path / "exact-assets"

    manifest = builder.build_exact_dense_assets(
        collections={"detail_128": _FakeCollection()},
        output_dir=output_dir,
        source_status_path=status_path,
        batch_size=1,
    )
    assets, loaded_manifest = retriever.load_exact_dense_assets(output_dir)

    assert manifest["ready"] is True
    assert manifest["distance_metric"] == "squared_l2"
    assert manifest["collections"]["detail_128"]["count"] == 2
    assert loaded_manifest == manifest
    assert [row["candidate_key"] for row in assets["detail_128"]["rows"]] == [
        "detail_128::a",
        "detail_128::z",
    ]
    np.testing.assert_array_equal(
        assets["detail_128"]["embeddings"],
        np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    )
