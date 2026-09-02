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
    / "validation40_graph_candidate_loss_audit.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "Graph candidate loss audit module is missing"
    spec = importlib.util.spec_from_file_location("graph_candidate_loss_audit", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(key: str, source: str, page: int, *, rank: int, score: float) -> dict:
    return {
        "candidate_key": key,
        "source_file": source,
        "page_number": page,
        "content": f"evidence-{key}",
        "post_rerank_rank": rank,
        "reranker_score": score,
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def test_classifies_gold_removed_before_reranking_as_budget_replacement():
    module = _load_module()
    gold = {"source_filename": "A.pdf", "page_number": 10}
    removed_gold = _candidate("gold", "A.pdf", 10, rank=4, score=0.8)

    result = module.classify_loss_mechanism(
        gold=gold,
        f_candidates=[removed_gold],
        g1_candidates=[_candidate("graph", "B.pdf", 2, rank=1, score=0.2)],
        replaced_candidate_keys=["gold"],
        candidate_budget=20,
    )

    assert result["primary_mechanism"] == "pre_rerank_budget_replacement"
    assert result["removed_gold_candidate_keys"] == ["gold"]
    assert result["removed_gold_best_cached_rerank_rank"] == 4


def test_shared_candidate_score_audit_separates_float_noise_from_content_drift():
    module = _load_module()
    f_candidates = [
        _candidate("same", "A.pdf", 1, rank=1, score=0.5000000),
        _candidate("changed", "B.pdf", 2, rank=2, score=0.2),
    ]
    g1_candidates = [
        _candidate("same", "A.pdf", 1, rank=1, score=0.5000002),
        {**_candidate("changed", "B.pdf", 2, rank=2, score=0.4), "content": "drift"},
    ]

    audit = module.audit_shared_candidate_score_parity(
        f_candidates,
        g1_candidates,
        tolerance=1e-5,
    )

    assert audit["shared_candidate_count"] == 2
    assert audit["nonzero_score_difference_count"] == 2
    assert audit["score_difference_above_tolerance_count"] == 1
    assert audit["content_mismatch_count"] == 1


def test_late_union_restores_removed_gold_before_fixed_budget_pruning():
    module = _load_module()
    f_candidates = [
        _candidate("keep", "B.pdf", 1, rank=1, score=0.9),
        _candidate("gold", "A.pdf", 10, rank=2, score=0.8),
    ]
    g1_candidates = [
        _candidate("keep", "B.pdf", 1, rank=1, score=0.9),
        _candidate("graph", "C.pdf", 3, rank=2, score=0.2),
    ]

    result = module.simulate_late_union_from_cached_scores(
        f_candidates=f_candidates,
        g1_candidates=g1_candidates,
        candidate_budget=2,
        final_evidence_k=2,
        dedup_ngram_size=3,
        dedup_overlap_threshold=0.75,
    )

    assert result["reranker_input_count"] == 3
    assert [row["candidate_key"] for row in result["candidates"]] == ["keep", "gold"]
    assert [row["candidate_key"] for row in result["evidence"]] == ["keep", "gold"]
    assert result["diagnostic_only"] is True
    assert result["matched_non_graph_compute_control_required"] is True


def test_run_loss_audit_writes_guarded_diagnostic_outputs(tmp_path):
    module = _load_module()
    f_method = "f"
    g1_method = "g1"
    question = "Q"
    keep = _candidate("keep", "B.pdf", 1, rank=1, score=0.9)
    gold = _candidate("gold", "A.pdf", 10, rank=2, score=0.8)
    graph = _candidate("graph", "C.pdf", 3, rank=2, score=0.2)
    results_path = tmp_path / "results.jsonl"
    metrics_path = tmp_path / "metrics.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_jsonl(
        results_path,
        [
            {
                "sample_id": "S1",
                "question": question,
                "methods": {
                    f_method: {
                        "candidates_top20": [keep, gold],
                        "evidence_top4": [keep, gold],
                    },
                    g1_method: {
                        "candidates_top20": [keep, graph],
                        "evidence_top4": [keep, graph],
                    },
                },
                "graph_expansion_audit": {
                    "added_candidate_keys": ["graph"],
                    "replaced_candidate_keys": ["gold"],
                },
            }
        ],
    )
    _write_jsonl(
        metrics_path,
        [
            {
                "sample_id": "S1",
                "question": question,
                "gold_source_filename": "A.pdf",
                "gold_page_number": 10,
                "f_candidate_strict_hit": True,
                "g1_candidate_strict_hit": False,
                "f_final_strict_hit": True,
                "g1_final_strict_hit": False,
            }
        ],
    )
    _write_json(summary_path, {"sample_count": 1})
    config = {
        "config_version": "loss-audit-test-v0.1",
        "expected_count": 1,
        "expected_results_sha256": _sha(results_path),
        "expected_sample_metrics_sha256": _sha(metrics_path),
        "expected_paired_summary_sha256": _sha(summary_path),
        "f_method": f_method,
        "g1_method": g1_method,
        "candidate_budget": 2,
        "final_evidence_k": 2,
        "dedup_ngram_size": 3,
        "dedup_overlap_threshold": 0.75,
        "score_tolerance": 1e-5,
        "samples_filename": "samples.jsonl",
        "summary_filename": "summary.json",
        "report_filename": "report.md",
        "audit_filename": "audit.json",
        "manifest_filename": "manifest.json",
    }

    output = module.run_loss_audit(
        results_path=results_path,
        sample_metrics_path=metrics_path,
        paired_summary_path=summary_path,
        output_dir=tmp_path / "audit-output",
        config=config,
    )

    assert output["summary"]["candidate_loss_count"] == 1
    assert output["summary"]["final_loss_count"] == 1
    assert output["summary"]["late_union_diagnostic"]["candidate_strict_hits"] == 1
    assert output["audit"]["pilot_test_accessed"] is False
    assert output["audit"]["external_model_calls"] == 0
    assert output["audit"]["counterfactual_diagnostic_only"] is True
    assert set(output["manifest"]["files"]) == {
        "samples",
        "summary",
        "report",
        "audit",
    }

    second_output_dir = tmp_path / "audit-output-second"
    module.run_loss_audit(
        results_path=results_path,
        sample_metrics_path=metrics_path,
        paired_summary_path=summary_path,
        output_dir=second_output_dir,
        config=config,
    )
    for filename in (
        "samples.jsonl",
        "summary.json",
        "report.md",
        "audit.json",
        "manifest.json",
    ):
        assert (tmp_path / "audit-output" / filename).read_bytes() == (
            second_output_dir / filename
        ).read_bytes()


def test_run_loss_audit_rejects_locked_input_hash_drift(tmp_path):
    module = _load_module()
    results_path = tmp_path / "results.jsonl"
    metrics_path = tmp_path / "metrics.jsonl"
    summary_path = tmp_path / "summary.json"
    results_path.write_text("", encoding="utf-8")
    metrics_path.write_text("", encoding="utf-8")
    _write_json(summary_path, {"sample_count": 0})

    with pytest.raises(ValueError, match="results SHA-256 mismatch"):
        module.run_loss_audit(
            results_path=results_path,
            sample_metrics_path=metrics_path,
            paired_summary_path=summary_path,
            output_dir=tmp_path / "must-not-exist",
            config={
                "expected_count": 0,
                "expected_results_sha256": "0" * 64,
                "expected_sample_metrics_sha256": _sha(metrics_path),
                "expected_paired_summary_sha256": _sha(summary_path),
                "f_method": "f",
                "g1_method": "g1",
            },
        )

    assert not (tmp_path / "must-not-exist").exists()
