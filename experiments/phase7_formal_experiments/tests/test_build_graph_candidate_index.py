from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "build_graph_candidate_index.py"
)
COLLECTIONS = ("detail_128", "concept_512", "context_1024")


def _load_module():
    assert MODULE_PATH.exists(), "Graph candidate index builder is missing"
    spec = importlib.util.spec_from_file_location(
        "build_graph_candidate_index",
        MODULE_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _make_exact_assets(root: Path) -> dict:
    collections: dict[str, dict] = {}
    for index, collection in enumerate(COLLECTIONS, start=1):
        row = {
            "candidate_key": f"{collection}::doc-{index}",
            "collection": collection,
            "document_id": f"doc-{index}",
            "content": "治疗48-72小时症状无改善时，应再次进行临床评估。",
            "source_file": "儿童社区获得性肺炎诊疗规范.pdf",
            "page_number": 20 + index,
            "granularity": index * 128,
        }
        rows_path = root / f"{collection}_rows.jsonl"
        rows_path.write_text(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        collections[collection] = {
            "count": 1,
            "rows_file": rows_path.name,
            "rows_sha256": _sha256(rows_path),
        }
    manifest = {
        "asset_version": "bge-m3-exact-dense-assets-v0.1",
        "ready": True,
        "collections": collections,
    }
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "expected_candidate_count": 3,
        "expected_asset_manifest_sha256": _sha256(manifest_path),
        "graph_index_version": "phase7-c1c4e2-candidate-graph-v0.1",
        "expected_collection_rows_sha256": {
            name: values["rows_sha256"]
            for name, values in collections.items()
        },
    }


def _rewrite_collection(
    assets: Path,
    config: dict,
    collection: str,
    rows: list[dict],
) -> None:
    rows_path = assets / f"{collection}_rows.jsonl"
    rows_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )
    manifest_path = assets / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed_hash = _sha256(rows_path)
    manifest["collections"][collection]["count"] = len(rows)
    manifest["collections"][collection]["rows_sha256"] = observed_hash
    _write_json(manifest_path, manifest)
    config["expected_candidate_count"] = sum(
        int(values["count"])
        for values in manifest["collections"].values()
    )
    config["expected_collection_rows_sha256"][collection] = observed_hash
    config["expected_asset_manifest_sha256"] = _sha256(manifest_path)


def test_builds_hash_audited_graph_index_from_three_exact_collections(tmp_path):
    module = _load_module()
    assets = tmp_path / "assets"
    output = tmp_path / "output"
    assets.mkdir()
    config = _make_exact_assets(assets)

    result = module.build_graph_candidate_index_assets(
        exact_asset_dir=assets,
        output_dir=output,
        config=config,
    )

    assert result["audit"]["input_candidate_count"] == 3
    assert result["audit"]["unique_candidate_count"] == 3
    assert result["audit"]["collection_counts"] == {
        "concept_512": 1,
        "context_1024": 1,
        "detail_128": 1,
    }
    assert result["audit"]["source_count"] == 1
    assert result["audit"]["source_page_count"] == 3
    assert result["manifest"]["ready"] is True
    assert result["manifest"]["external_model_calls"] == 0
    assert result["manifest"]["input_tokens"] == 0
    assert result["manifest"]["output_tokens"] == 0
    assert result["manifest"]["estimated_cost"] == 0.0
    for filename in (
        "graph_candidate_index_v0_1.json",
        "graph_candidate_index_audit_v0_1.json",
        "manifest.json",
    ):
        assert (output / filename).is_file()


def test_rejects_exact_asset_manifest_hash_drift(tmp_path):
    module = _load_module()
    assets = tmp_path / "assets"
    assets.mkdir()
    config = _make_exact_assets(assets)
    config["expected_asset_manifest_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        module.build_graph_candidate_index_assets(
            exact_asset_dir=assets,
            output_dir=tmp_path / "output",
            config=config,
        )


def test_rejects_gold_only_fields_before_writing_graph_assets(tmp_path):
    module = _load_module()
    assets = tmp_path / "assets"
    assets.mkdir()
    config = _make_exact_assets(assets)
    rows_path = assets / "detail_128_rows.jsonl"
    row = json.loads(rows_path.read_text(encoding="utf-8"))
    row["gold_evidence"] = "must never enter runtime graph"
    _rewrite_collection(assets, config, "detail_128", [row])
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="gold-only"):
        module.build_graph_candidate_index_assets(
            exact_asset_dir=assets,
            output_dir=output,
            config=config,
        )
    assert not output.exists()


def test_rejects_conflicting_duplicate_candidate_keys(tmp_path):
    module = _load_module()
    assets = tmp_path / "assets"
    assets.mkdir()
    config = _make_exact_assets(assets)
    detail_row = json.loads(
        (assets / "detail_128_rows.jsonl").read_text(encoding="utf-8")
    )
    concept_row = json.loads(
        (assets / "concept_512_rows.jsonl").read_text(encoding="utf-8")
    )
    concept_row["candidate_key"] = detail_row["candidate_key"]
    _rewrite_collection(assets, config, "concept_512", [concept_row])

    with pytest.raises(ValueError, match="conflicting duplicate candidate_key"):
        module.build_graph_candidate_index_assets(
            exact_asset_dir=assets,
            output_dir=tmp_path / "output",
            config=config,
        )


def test_same_input_produces_identical_graph_asset_hashes(tmp_path):
    module = _load_module()
    assets = tmp_path / "assets"
    assets.mkdir()
    config = _make_exact_assets(assets)

    first = module.build_graph_candidate_index_assets(
        exact_asset_dir=assets,
        output_dir=tmp_path / "first",
        config=config,
    )
    second = module.build_graph_candidate_index_assets(
        exact_asset_dir=assets,
        output_dir=tmp_path / "second",
        config=config,
    )

    assert first == second
    for filename in (
        "graph_candidate_index_v0_1.json",
        "graph_candidate_index_audit_v0_1.json",
        "manifest.json",
    ):
        assert _sha256(tmp_path / "first" / filename) == _sha256(
            tmp_path / "second" / filename
        )


def test_rejects_configured_graph_index_version_mismatch(tmp_path):
    module = _load_module()
    assets = tmp_path / "assets"
    assets.mkdir()
    config = _make_exact_assets(assets)
    config["graph_index_version"] = "unsupported-graph-index"

    with pytest.raises(ValueError, match="configured graph index version mismatch"):
        module.build_graph_candidate_index_assets(
            exact_asset_dir=assets,
            output_dir=tmp_path / "output",
            config=config,
        )
