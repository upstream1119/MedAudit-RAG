import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "benchmark_anchor_candidates.py"
)
PROHIBITED_CANDIDATE_FIELDS = {
    "expected_decision",
    "required_claims",
    "allowed_claims",
    "forbidden_claims",
    "gold_evidence_status",
}


def _load_module():
    assert MODULE_PATH.exists(), "Benchmark anchor candidate builder is not implemented"
    spec = importlib.util.spec_from_file_location(
        "benchmark_anchor_candidates",
        MODULE_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _coverage(
    source_id: str,
    *,
    evidence_types: list[str] | None = None,
) -> dict:
    return {
        "source_id": source_id,
        "title": f"测试资料 {source_id}",
        "filename": f"{source_id}.pdf",
        "actual_sha256": f"sha256-{source_id}",
        "hash_matches": True,
        "file_size_matches": True,
        "file_exists": True,
        "included_in_kb": True,
        "status": "approved",
        "coverage_status": "pending_candidate_extraction",
        "evidence_types": evidence_types or ["pediatric_dosing"],
        "can_support": ["测试范围"],
        "cannot_support": ["个体化处方"],
    }


def _chunk(
    text: str,
    *,
    page: int = 1,
    granularity: int = 128,
    block_type: str = "text",
):
    return SimpleNamespace(
        content=text,
        granularity=granularity,
        metadata=SimpleNamespace(
            page_number=page,
            block_type=block_type,
        ),
    )


def _config() -> dict:
    return {
        "config_version": "benchmark-anchor-search-test-v0.1",
        "parser_version": "dual-track-medical-parser-test",
        "chunker_version": "semantic-chunker-test",
        "granularities": [128, 512, 1024],
        "min_text_chars": 20,
        "max_text_chars": 2000,
        "title_only_max_chars": 120,
        "scope_limited_evidence_types": [
            "medicine_identity",
            "dosage_form",
            "essential_medicine_status",
        ],
        "topic_terms": {
            "dose": ["剂量", "mg/kg"],
            "frequency": ["每日一次", "qd", "bid"],
            "route": ["静脉滴注", "口服"],
            "monitoring": ["监测", "再评估"],
        },
    }


def test_builds_only_unverified_candidates_from_verified_sources():
    module = _load_module()
    coverage_rows = [_coverage("SRC-002"), _coverage("SRC-001")]
    chunks_by_source = {
        "SRC-001": {
            128: [
                _chunk(
                    "阿奇霉素静脉滴注剂量为10 mg/kg，每日一次，并应监测疗效。",
                    page=14,
                )
            ]
        },
        "SRC-002": {
            128: [
                _chunk(
                    "治疗后症状无改善时应再次评估，并结合临床情况进行监测。",
                    page=26,
                )
            ]
        },
    }

    candidates, source_results = module.build_anchor_candidates(
        coverage_rows,
        _config(),
        lambda source: chunks_by_source[source["source_id"]],
        expected_count=2,
    )

    assert [candidate["source_id"] for candidate in candidates] == [
        "SRC-001",
        "SRC-002",
    ]
    assert all(
        candidate["review_status"] == "candidate_unverified"
        for candidate in candidates
    )
    assert all(candidate["page_number"] > 0 for candidate in candidates)
    assert all(candidate["raw_text"].strip() for candidate in candidates)
    assert all(
        not (PROHIBITED_CANDIDATE_FIELDS & set(candidate))
        for candidate in candidates
    )
    assert [result["source_id"] for result in source_results] == [
        "SRC-001",
        "SRC-002",
    ]
    assert all(result["processing_status"] == "candidate_ready" for result in source_results)


@pytest.mark.parametrize(
    ("text", "page"),
    [
        ("儿童肺炎支原体肺炎诊疗指南（2023年版）", 14),
        ("## 参考文献\n[1] 某某. 儿科剂量研究。", 14),
        (
            "## ［关键词］超说明书用药;儿童剂量;循证药学;指南共识;推荐意见"
            "［中图分类号］R95［文献标志码］A DOI 10.1000/example",
            1,
        ),
        ("**==> picture [31 x 14] intentionally omitted <==** 剂量表", 14),
        ("目录\n第一章........1\n第二章........2\n第三章........3", 14),
        ("剂量", 14),
        ("阿奇霉素剂量为10 mg/kg，每日一次。", 0),
    ],
)
def test_filters_noise_title_low_information_and_invalid_page(text, page):
    module = _load_module()
    candidates, source_results = module.build_anchor_candidates(
        [_coverage("SRC-001")],
        _config(),
        lambda _source: {128: [_chunk(text, page=page)]},
        expected_count=1,
    )

    assert candidates == []
    assert source_results[0]["processing_status"] == "no_candidate_after_filtering"


def test_deduplicates_cross_granularity_and_keeps_stable_ids():
    module = _load_module()
    duplicate_text = "儿童给药剂量应结合体重计算，推荐剂量为10 mg/kg，每日一次。"
    chunks = {
        1024: [_chunk(duplicate_text, page=8, granularity=1024)],
        128: [_chunk(duplicate_text, page=8, granularity=128)],
        512: [_chunk(duplicate_text, page=8, granularity=512)],
    }

    first, _ = module.build_anchor_candidates(
        [_coverage("SRC-001")],
        _config(),
        lambda _source: chunks,
        expected_count=1,
    )
    second, _ = module.build_anchor_candidates(
        [_coverage("SRC-001")],
        _config(),
        lambda _source: dict(reversed(list(chunks.items()))),
        expected_count=1,
    )

    assert len(first) == 1
    assert first == second
    assert first[0]["granularity"] == 128
    assert first[0]["candidate_id"].startswith("CAND-")


def test_records_scope_limited_and_parse_failed_sources():
    module = _load_module()
    coverage_rows = [
        _coverage(
            "SRC-001",
            evidence_types=[
                "medicine_identity",
                "dosage_form",
                "essential_medicine_status",
            ],
        ),
        _coverage("SRC-002"),
    ]

    def chunk_loader(source):
        if source["source_id"] == "SRC-002":
            raise RuntimeError("parser failed")
        return {128: [_chunk("药品目录仅列出剂型信息，不提供临床剂量。")]}

    candidates, source_results = module.build_anchor_candidates(
        coverage_rows,
        _config(),
        chunk_loader,
        expected_count=2,
    )

    assert candidates == []
    assert source_results == [
        {
            "source_id": "SRC-001",
            "candidate_count": 0,
            "parsed_chunk_count": 1,
            "processing_status": "scope_limited",
            "failure_reason": "",
        },
        {
            "source_id": "SRC-002",
            "candidate_count": 0,
            "parsed_chunk_count": 0,
            "processing_status": "parse_failed",
            "failure_reason": "RuntimeError: parser failed",
        },
    ]


def test_rejects_unverified_or_duplicate_source_coverage():
    module = _load_module()
    invalid = _coverage("SRC-001")
    invalid["hash_matches"] = False

    with pytest.raises(ValueError, match="未通过 B1.0"):
        module.build_anchor_candidates(
            [invalid],
            _config(),
            lambda _source: {},
            expected_count=1,
        )

    with pytest.raises(ValueError, match="重复 source_id"):
        module.build_anchor_candidates(
            [_coverage("SRC-001"), _coverage("SRC-001")],
            _config(),
            lambda _source: {},
            expected_count=2,
        )


def test_writes_deterministic_jsonl_and_summary(tmp_path):
    module = _load_module()
    config = _config()
    candidates, source_results = module.build_anchor_candidates(
        [_coverage("SRC-001")],
        config,
        lambda _source: {
            128: [
                _chunk(
                    "阿奇霉素静脉滴注剂量为10 mg/kg，每日一次，并应监测疗效。",
                    page=14,
                )
            ]
        },
        expected_count=1,
    )

    first = module.write_candidate_outputs(
        candidates,
        source_results,
        tmp_path,
        config=config,
        coverage_sha256="coverage-sha256",
        expected_count=1,
    )
    first_bytes = {
        name: Path(path).read_bytes()
        for name, path in first.items()
    }
    second = module.write_candidate_outputs(
        candidates,
        source_results,
        tmp_path,
        config=config,
        coverage_sha256="coverage-sha256",
        expected_count=1,
    )
    second_bytes = {
        name: Path(path).read_bytes()
        for name, path in second.items()
    }

    assert first_bytes == second_bytes
    jsonl_rows = [
        json.loads(line)
        for line in Path(first["jsonl"]).read_text(encoding="utf-8").splitlines()
    ]
    assert jsonl_rows == candidates
    report = Path(first["report"]).read_text(encoding="utf-8")
    assert "candidate_unverified" in report
    assert "SRC-001" in report
    assert "外部 API 调用：0" in report
