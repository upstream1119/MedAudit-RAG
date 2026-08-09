import csv
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
    / "benchmark_source_coverage.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "Benchmark source coverage builder is not implemented"
    spec = importlib.util.spec_from_file_location(
        "benchmark_source_coverage",
        MODULE_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _source(
    source_id: str,
    filename: str,
    content: bytes,
    *,
    included_in_kb: bool = True,
    status: str = "approved",
) -> dict:
    return {
        "source_id": source_id,
        "title": f"测试资料 {source_id}",
        "year": 2026,
        "publisher": "测试发布机构",
        "jurisdiction": "CN",
        "source_type": "clinical_guideline",
        "authority_level": "national",
        "filename": filename,
        "status": status,
        "included_in_kb": included_in_kb,
        "evidence_types": ["dose"],
        "can_support": ["测试支持范围"],
        "cannot_support": ["个体化处方"],
        "content_check": {"status": "spot_checked"},
        "file_size": len(content),
        "sha256": _sha256(content),
    }


def _write_manifest(path: Path, sources: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "formal_directory": str(path.parent),
                "sources": sources,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def test_selects_only_included_sources_in_stable_order(tmp_path):
    module = _load_module()
    formal_dir = tmp_path / "guidelines"
    formal_dir.mkdir()
    contents = {
        "b.pdf": b"%PDF-source-b",
        "a.pdf": b"%PDF-source-a",
        "excluded.pdf": b"%PDF-source-excluded",
    }
    for filename, content in contents.items():
        (formal_dir / filename).write_bytes(content)

    manifest_path = tmp_path / "source_manifest.json"
    _write_manifest(
        manifest_path,
        [
            _source("SRC-002", "b.pdf", contents["b.pdf"], status="indexed"),
            _source("SRC-001", "a.pdf", contents["a.pdf"]),
            _source(
                "SRC-999",
                "excluded.pdf",
                contents["excluded.pdf"],
                included_in_kb=False,
                status="registered",
            ),
        ],
    )

    rows = module.build_source_coverage(
        manifest_path,
        formal_dir,
        expected_count=2,
    )

    assert [row["source_id"] for row in rows] == ["SRC-001", "SRC-002"]
    assert all(row["file_exists"] for row in rows)
    assert all(row["hash_matches"] for row in rows)
    assert all(row["file_size_matches"] for row in rows)
    assert [row["actual_file_size"] for row in rows] == [
        len(contents["a.pdf"]),
        len(contents["b.pdf"]),
    ]
    assert all(row["coverage_status"] == "pending_candidate_extraction" for row in rows)
    assert all(row["candidate_anchor_count"] == 0 for row in rows)
    assert all(row["verified_anchor_count"] == 0 for row in rows)


def test_rejects_missing_required_source_field(tmp_path):
    module = _load_module()
    formal_dir = tmp_path / "guidelines"
    formal_dir.mkdir()
    content = b"%PDF-missing-field"
    (formal_dir / "source.pdf").write_bytes(content)
    source = _source("SRC-001", "source.pdf", content)
    source.pop("can_support")
    manifest_path = tmp_path / "source_manifest.json"
    _write_manifest(manifest_path, [source])

    with pytest.raises(ValueError, match="缺少必要字段.*can_support"):
        module.build_source_coverage(
            manifest_path,
            formal_dir,
            expected_count=1,
        )


def test_rejects_invalid_admission_status(tmp_path):
    module = _load_module()
    formal_dir = tmp_path / "guidelines"
    formal_dir.mkdir()
    content = b"%PDF-invalid-status"
    (formal_dir / "source.pdf").write_bytes(content)
    manifest_path = tmp_path / "source_manifest.json"
    _write_manifest(
        manifest_path,
        [_source("SRC-001", "source.pdf", content, status="inspected")],
    )

    with pytest.raises(ValueError, match="无效准入状态.*inspected"):
        module.build_source_coverage(
            manifest_path,
            formal_dir,
            expected_count=1,
        )


def test_rejects_missing_pdf(tmp_path):
    module = _load_module()
    formal_dir = tmp_path / "guidelines"
    formal_dir.mkdir()
    content = b"%PDF-missing"
    manifest_path = tmp_path / "source_manifest.json"
    _write_manifest(
        manifest_path,
        [_source("SRC-001", "missing.pdf", content)],
    )

    with pytest.raises(FileNotFoundError, match="PDF 不存在"):
        module.build_source_coverage(
            manifest_path,
            formal_dir,
            expected_count=1,
        )


def test_rejects_sha256_mismatch(tmp_path):
    module = _load_module()
    formal_dir = tmp_path / "guidelines"
    formal_dir.mkdir()
    expected_content = b"%PDF-expected"
    (formal_dir / "source.pdf").write_bytes(b"%PDF-actual")
    manifest_path = tmp_path / "source_manifest.json"
    _write_manifest(
        manifest_path,
        [_source("SRC-001", "source.pdf", expected_content)],
    )

    with pytest.raises(ValueError, match="SHA-256 不匹配"):
        module.build_source_coverage(
            manifest_path,
            formal_dir,
            expected_count=1,
        )


def test_rejects_duplicate_source_id(tmp_path):
    module = _load_module()
    formal_dir = tmp_path / "guidelines"
    formal_dir.mkdir()
    first_content = b"%PDF-first"
    second_content = b"%PDF-second"
    (formal_dir / "first.pdf").write_bytes(first_content)
    (formal_dir / "second.pdf").write_bytes(second_content)
    manifest_path = tmp_path / "source_manifest.json"
    _write_manifest(
        manifest_path,
        [
            _source("SRC-001", "first.pdf", first_content),
            _source("SRC-001", "second.pdf", second_content),
        ],
    )

    with pytest.raises(ValueError, match="重复 source_id"):
        module.build_source_coverage(
            manifest_path,
            formal_dir,
            expected_count=2,
        )


def test_writes_deterministic_jsonl_csv_and_markdown_outputs(tmp_path):
    module = _load_module()
    formal_dir = tmp_path / "guidelines"
    formal_dir.mkdir()
    contents = {
        "a.pdf": b"%PDF-source-a",
        "b.pdf": b"%PDF-source-b",
    }
    for filename, content in contents.items():
        (formal_dir / filename).write_bytes(content)
    manifest_path = tmp_path / "source_manifest.json"
    _write_manifest(
        manifest_path,
        [
            _source("SRC-002", "b.pdf", contents["b.pdf"], status="indexed"),
            _source("SRC-001", "a.pdf", contents["a.pdf"]),
        ],
    )
    output_dir = tmp_path / "outputs"

    summary = module.run_source_coverage_audit(
        manifest_path,
        formal_dir,
        output_dir,
        expected_count=2,
    )
    first_bytes = {
        name: Path(path).read_bytes()
        for name, path in summary["output_files"].items()
    }
    second_summary = module.run_source_coverage_audit(
        manifest_path,
        formal_dir,
        output_dir,
        expected_count=2,
    )
    second_bytes = {
        name: Path(path).read_bytes()
        for name, path in second_summary["output_files"].items()
    }

    assert first_bytes == second_bytes
    assert summary["source_count"] == 2
    assert summary["all_hashes_match"] is True
    assert summary["manifest_sha256"] == _sha256(manifest_path.read_bytes())

    jsonl_rows = [
        json.loads(line)
        for line in Path(summary["output_files"]["jsonl"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["source_id"] for row in jsonl_rows] == ["SRC-001", "SRC-002"]

    with Path(summary["output_files"]["csv"]).open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file_obj:
        csv_rows = list(csv.DictReader(file_obj))
    assert json.loads(csv_rows[0]["can_support"]) == ["测试支持范围"]
    assert csv_rows[0]["coverage_status"] == "pending_candidate_extraction"

    report = Path(summary["output_files"]["report"]).read_text(encoding="utf-8")
    assert "正式来源数量：2/2" in report
    assert "SRC-001" in report
    assert "SRC-002" in report
    assert "外部 API 调用：0" in report
