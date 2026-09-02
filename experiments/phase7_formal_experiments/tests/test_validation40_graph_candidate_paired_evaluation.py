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
    / "validation40_graph_candidate_paired_evaluation.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "Paired evaluation module is missing"
    spec = importlib.util.spec_from_file_location("paired_evaluation", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(source: str, page: int) -> dict:
    return {
        "candidate_key": f"detail_128::{source}-{page}",
        "source_file": source,
        "page_number": page,
        "content": "evidence",
    }


def _method(
    candidates: list[dict],
    evidence: list[dict],
    *,
    candidate_field: str = "candidates_top20",
) -> dict:
    return {candidate_field: candidates, "evidence_top4": evidence}


def test_rank_metrics_separate_strict_source_and_adjacent_hits():
    module = _load_module()
    gold = {"source_filename": "A.pdf", "page_number": 10}
    candidates = [_candidate("A.pdf", 9), _candidate("B.pdf", 10), _candidate("A.pdf", 10)]

    metrics = module.rank_metrics(candidates, gold, cutoff=3)

    assert metrics["strict_hit"] is True
    assert metrics["strict_rank"] == 3
    assert metrics["strict_mrr"] == pytest.approx(1 / 3)
    assert metrics["source_hit"] is True
    assert metrics["source_rank"] == 1
    assert metrics["adjacent_hit"] is True
    assert metrics["adjacent_rank"] == 1


def test_paired_summary_counts_added_lost_both_and_neither():
    module = _load_module()
    sample_rows = [
        {"f_candidate_strict_hit": False, "g1_candidate_strict_hit": True,
         "f_final_strict_hit": False, "g1_final_strict_hit": True},
        {"f_candidate_strict_hit": True, "g1_candidate_strict_hit": False,
         "f_final_strict_hit": True, "g1_final_strict_hit": False},
        {"f_candidate_strict_hit": True, "g1_candidate_strict_hit": True,
         "f_final_strict_hit": True, "g1_final_strict_hit": True},
        {"f_candidate_strict_hit": False, "g1_candidate_strict_hit": False,
         "f_final_strict_hit": False, "g1_final_strict_hit": False},
    ]

    summary = module.summarize_paired_results(sample_rows)

    assert summary["candidate_pair_counts"] == {
        "added": 1,
        "lost": 1,
        "both": 1,
        "neither": 1,
    }
    assert summary["final_pair_counts"] == {
        "added": 1,
        "lost": 1,
        "both": 1,
        "neither": 1,
    }


def test_freeze_recommendation_requires_candidate_gain_and_no_final_regression():
    module = _load_module()

    accepted = module.freeze_recommendation(
        f_candidate_strict_recall=0.50,
        g1_candidate_strict_recall=0.55,
        f_final_strict_recall=0.50,
        g1_final_strict_recall=0.50,
    )
    no_candidate_gain = module.freeze_recommendation(
        f_candidate_strict_recall=0.50,
        g1_candidate_strict_recall=0.50,
        f_final_strict_recall=0.50,
        g1_final_strict_recall=0.55,
    )
    final_regression = module.freeze_recommendation(
        f_candidate_strict_recall=0.50,
        g1_candidate_strict_recall=0.55,
        f_final_strict_recall=0.50,
        g1_final_strict_recall=0.45,
    )

    assert accepted["decision"] == "freeze_g1_candidate_expansion"
    assert no_candidate_gain["decision"] == "do_not_freeze_g1"
    assert final_regression["decision"] == "do_not_freeze_g1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_evaluation_rejects_manifest_hash_drift_and_sample_mismatch(tmp_path):
    module = _load_module()
    results = tmp_path / "results.jsonl"
    manifest = tmp_path / "manifest.json"
    gold = tmp_path / "gold.jsonl"
    result_row = {
        "sample_id": "S1",
        "question": "Q1",
        "methods": {
            "f_exact_hybrid_reranker_dedup": _method([_candidate("A.pdf", 1)], [_candidate("A.pdf", 1)]),
            "g1_exact_graph_expand_reranker_dedup": _method([_candidate("A.pdf", 1)], [_candidate("A.pdf", 1)]),
        },
    }
    _write_jsonl(results, [result_row])
    _write_json(manifest, {"files": {"results": {"sha256": "0" * 64}}})
    _write_jsonl(gold, [{"candidate_id": "S1", "question": "Q1", "source_filename": "A.pdf", "page_number": 1}])

    config = {"expected_count": 1, "expected_gold_sha256": _sha(gold)}
    with pytest.raises(ValueError, match="retrieval results SHA-256 mismatch"):
        module.evaluate_paired_retrieval(
            results_path=results,
            retrieval_manifest_path=manifest,
            gold_path=gold,
            output_dir=tmp_path / "out-hash",
            config=config,
        )

    _write_json(manifest, {"files": {"results": {"sha256": _sha(results)}}})
    _write_jsonl(gold, [{"candidate_id": "OTHER", "question": "Q1", "source_filename": "A.pdf", "page_number": 1}])
    config["expected_gold_sha256"] = _sha(gold)
    with pytest.raises(ValueError, match="sample_id/question mismatch"):
        module.evaluate_paired_retrieval(
            results_path=results,
            retrieval_manifest_path=manifest,
            gold_path=gold,
            output_dir=tmp_path / "out-mismatch",
            config=config,
        )


def test_evaluation_writes_candidate_and_final_metrics_separately(tmp_path):
    module = _load_module()
    results = tmp_path / "results.jsonl"
    manifest = tmp_path / "manifest.json"
    gold = tmp_path / "gold.jsonl"
    result_row = {
        "sample_id": "S1",
        "question": "Q1",
        "methods": {
            "f_exact_hybrid_reranker_dedup": _method([_candidate("A.pdf", 1)], [_candidate("B.pdf", 1)]),
            "g1_exact_graph_expand_reranker_dedup": _method([_candidate("A.pdf", 1)], [_candidate("A.pdf", 1)]),
        },
    }
    _write_jsonl(results, [result_row])
    _write_json(manifest, {"files": {"results": {"sha256": _sha(results)}}})
    _write_jsonl(gold, [{"candidate_id": "S1", "question": "Q1", "source_filename": "A.pdf", "page_number": 1}])
    config = {
        "expected_count": 1,
        "expected_gold_sha256": _sha(gold),
        "sample_metrics_filename": "sample_metrics.jsonl",
        "summary_filename": "summary.json",
        "report_filename": "report.md",
        "audit_filename": "audit.json",
        "manifest_filename": "manifest.json",
    }

    output = module.evaluate_paired_retrieval(
        results_path=results,
        retrieval_manifest_path=manifest,
        gold_path=gold,
        output_dir=tmp_path / "evaluation",
        config=config,
    )

    assert output["summary"]["methods"]["f_exact_hybrid_reranker_dedup"]["candidate_strict_recall_at_20"] == 1.0
    assert output["summary"]["methods"]["f_exact_hybrid_reranker_dedup"]["final_strict_recall_at_4"] == 0.0
    assert output["summary"]["methods"]["g1_exact_graph_expand_reranker_dedup"]["final_strict_recall_at_4"] == 1.0
    assert (tmp_path / "evaluation" / "report.md").exists()


def test_evaluation_uses_configured_candidate_budget_in_fields_metrics_and_report(
    tmp_path,
):
    module = _load_module()
    results = tmp_path / "results-top3.jsonl"
    manifest = tmp_path / "manifest-top3.json"
    gold = tmp_path / "gold-top3.jsonl"
    f_method = "f3"
    g1_method = "g1_3"
    candidate_field = "candidates_top3"
    result_row = {
        "sample_id": "S1",
        "question": "Q1",
        "methods": {
            f_method: _method(
                [_candidate("A.pdf", 1)],
                [_candidate("B.pdf", 1)],
                candidate_field=candidate_field,
            ),
            g1_method: _method(
                [_candidate("A.pdf", 1)],
                [_candidate("A.pdf", 1)],
                candidate_field=candidate_field,
            ),
        },
    }
    _write_jsonl(results, [result_row])
    _write_json(manifest, {"files": {"results": {"sha256": _sha(results)}}})
    _write_jsonl(
        gold,
        [
            {
                "candidate_id": "S1",
                "question": "Q1",
                "source_filename": "A.pdf",
                "page_number": 1,
            }
        ],
    )
    config = {
        "expected_count": 1,
        "expected_gold_sha256": _sha(gold),
        "f_method": f_method,
        "g1_method": g1_method,
        "candidate_budget": 3,
    }

    output = module.evaluate_paired_retrieval(
        results_path=results,
        retrieval_manifest_path=manifest,
        gold_path=gold,
        output_dir=tmp_path / "evaluation-top3",
        config=config,
    )

    f_summary = output["summary"]["methods"][f_method]
    assert f_summary["candidate_strict_recall_at_3"] == 1.0
    assert "candidate_strict_recall_at_20" not in f_summary
    assert output["summary"]["candidate_budget"] == 3
    assert output["summary"]["candidate_output_field"] == candidate_field
    report = (tmp_path / "evaluation-top3" / "report.md").read_text(encoding="utf-8")
    assert "Candidate strict recall@3" in report
    assert f"| {f_method} |" in report
    assert f"| {g1_method} |" in report


def test_evaluation_accepts_configured_method_ids_and_versions(tmp_path):
    module = _load_module()
    results = tmp_path / "results.jsonl"
    manifest = tmp_path / "manifest.json"
    gold = tmp_path / "gold.jsonl"
    f_method = "f_exact_hybrid_reranker_dedup"
    g1_method = "g1_v0_2_source_routed_graph_expand_reranker_dedup"
    result_row = {
        "sample_id": "S1",
        "question": "Q1",
        "methods": {
            f_method: _method([_candidate("A.pdf", 1)], [_candidate("A.pdf", 1)]),
            g1_method: _method([_candidate("A.pdf", 1)], [_candidate("A.pdf", 1)]),
        },
    }
    _write_jsonl(results, [result_row])
    _write_json(manifest, {"files": {"results": {"sha256": _sha(results)}}})
    _write_jsonl(
        gold,
        [
            {
                "candidate_id": "S1",
                "question": "Q1",
                "source_filename": "A.pdf",
                "page_number": 1,
            }
        ],
    )
    config = {
        "expected_count": 1,
        "expected_gold_sha256": _sha(gold),
        "f_method": f_method,
        "g1_method": g1_method,
        "summary_version": "paired-evaluation-v0.2",
        "audit_version": "paired-evaluation-audit-v0.2",
        "manifest_version": "paired-evaluation-manifest-v0.2",
    }

    output = module.evaluate_paired_retrieval(
        results_path=results,
        retrieval_manifest_path=manifest,
        gold_path=gold,
        output_dir=tmp_path / "evaluation-v0.2",
        config=config,
    )

    assert set(output["summary"]["methods"]) == {f_method, g1_method}
    assert output["summary"]["summary_version"] == "paired-evaluation-v0.2"
    assert output["audit"]["audit_version"] == "paired-evaluation-audit-v0.2"
    assert output["manifest"]["manifest_version"] == "paired-evaluation-manifest-v0.2"


def test_evaluation_rejects_configured_manifest_and_results_hash_drift(tmp_path):
    module = _load_module()
    results = tmp_path / "results.jsonl"
    manifest = tmp_path / "manifest.json"
    gold = tmp_path / "gold.jsonl"
    result_row = {
        "sample_id": "S1",
        "question": "Q1",
        "methods": {
            "f_exact_hybrid_reranker_dedup": _method(
                [_candidate("A.pdf", 1)], [_candidate("A.pdf", 1)]
            ),
            "g1_exact_graph_expand_reranker_dedup": _method(
                [_candidate("A.pdf", 1)], [_candidate("A.pdf", 1)]
            ),
        },
    }
    _write_jsonl(results, [result_row])
    _write_json(manifest, {"files": {"results": {"sha256": _sha(results)}}})
    _write_jsonl(
        gold,
        [
            {
                "candidate_id": "S1",
                "question": "Q1",
                "source_filename": "A.pdf",
                "page_number": 1,
            }
        ],
    )
    config = {
        "expected_count": 1,
        "expected_gold_sha256": _sha(gold),
        "expected_retrieval_manifest_sha256": "0" * 64,
        "expected_retrieval_results_sha256": _sha(results),
    }

    with pytest.raises(ValueError, match="retrieval manifest SHA-256 mismatch"):
        module.evaluate_paired_retrieval(
            results_path=results,
            retrieval_manifest_path=manifest,
            gold_path=gold,
            output_dir=tmp_path / "out-manifest-drift",
            config=config,
        )

    config["expected_retrieval_manifest_sha256"] = _sha(manifest)
    config["expected_retrieval_results_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="configured retrieval results SHA-256 mismatch"):
        module.evaluate_paired_retrieval(
            results_path=results,
            retrieval_manifest_path=manifest,
            gold_path=gold,
            output_dir=tmp_path / "out-results-drift",
            config=config,
        )


def test_evaluation_outputs_are_byte_identical_for_same_inputs(tmp_path):
    module = _load_module()
    results = tmp_path / "results.jsonl"
    manifest = tmp_path / "manifest.json"
    gold = tmp_path / "gold.jsonl"
    result_row = {
        "sample_id": "S1",
        "question": "Q1",
        "methods": {
            "f_exact_hybrid_reranker_dedup": _method(
                [_candidate("A.pdf", 1)], [_candidate("B.pdf", 1)]
            ),
            "g1_exact_graph_expand_reranker_dedup": _method(
                [_candidate("A.pdf", 1)], [_candidate("A.pdf", 1)]
            ),
        },
    }
    _write_jsonl(results, [result_row])
    _write_json(manifest, {"files": {"results": {"sha256": _sha(results)}}})
    _write_jsonl(
        gold,
        [
            {
                "candidate_id": "S1",
                "question": "Q1",
                "source_filename": "A.pdf",
                "page_number": 1,
            }
        ],
    )
    config = {
        "expected_count": 1,
        "expected_gold_sha256": _sha(gold),
        "sample_metrics_filename": "sample_metrics.jsonl",
        "summary_filename": "summary.json",
        "report_filename": "report.md",
        "audit_filename": "audit.json",
        "manifest_filename": "manifest.json",
    }
    output_a = tmp_path / "evaluation-a"
    output_b = tmp_path / "evaluation-b"

    for output_dir in (output_a, output_b):
        module.evaluate_paired_retrieval(
            results_path=results,
            retrieval_manifest_path=manifest,
            gold_path=gold,
            output_dir=output_dir,
            config=config,
        )

    for filename in (
        "sample_metrics.jsonl",
        "summary.json",
        "report.md",
        "audit.json",
        "manifest.json",
    ):
        assert (output_a / filename).read_bytes() == (output_b / filename).read_bytes()
