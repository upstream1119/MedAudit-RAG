from __future__ import annotations

import importlib.util
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "validation40_graph_path_reranking_evaluation.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "G1/G2 Gold-only evaluation module is missing"
    spec = importlib.util.spec_from_file_location(
        "validation40_graph_path_reranking_evaluation", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_freeze_rule_requires_strict_gain_and_source_non_degradation():
    module = _load_module()

    accepted = module.g2_freeze_recommendation(
        g1_final_strict_recall=0.50,
        g2_final_strict_recall=0.55,
        g1_final_source_recall=0.75,
        g2_final_source_recall=0.75,
        g1_final_strict_mrr=0.40,
        g2_final_strict_mrr=0.42,
    )
    diagnostic = module.g2_freeze_recommendation(
        g1_final_strict_recall=0.50,
        g2_final_strict_recall=0.50,
        g1_final_source_recall=0.75,
        g2_final_source_recall=0.75,
        g1_final_strict_mrr=0.40,
        g2_final_strict_mrr=0.42,
    )
    source_regression = module.g2_freeze_recommendation(
        g1_final_strict_recall=0.50,
        g2_final_strict_recall=0.55,
        g1_final_source_recall=0.75,
        g2_final_source_recall=0.70,
        g1_final_strict_mrr=0.40,
        g2_final_strict_mrr=0.42,
    )

    assert accepted["decision"] == "freeze_g2"
    assert diagnostic["decision"] == "diagnostic_only_g2"
    assert source_regression["decision"] == "do_not_freeze_g2"
    assert accepted["scope"] == "Validation40 development decision only"
    assert accepted["statistical_significance_claimed"] is False
    assert accepted["clinical_significance_claimed"] is False


def test_candidate_pool_identity_allows_only_reordering_and_audit_fields():
    module = _load_module()
    g1_candidates = [
        {
            "candidate_key": "detail_128::a",
            "collection": "detail_128",
            "document_id": "a",
            "content": "证据 A",
            "source_file": "A.pdf",
            "page_number": 1,
            "chapter_title": "治疗",
        },
        {
            "candidate_key": "detail_128::b",
            "collection": "detail_128",
            "document_id": "b",
            "content": "证据 B",
            "source_file": "B.pdf",
            "page_number": 2,
            "chapter_title": "复评",
        },
    ]
    g2_candidates = [deepcopy(g1_candidates[1]), deepcopy(g1_candidates[0])]
    for rank, candidate in enumerate(g2_candidates, start=1):
        candidate["graph_rerank_rank"] = rank
        candidate["graph_path_score"] = 0.5

    audit = module.assert_same_candidate_pool(g1_candidates, g2_candidates)

    assert audit == {"candidate_count": 2, "candidate_identity_set_equal": True}

    drifted = deepcopy(g2_candidates)
    drifted[0]["page_number"] = 99
    with pytest.raises(ValueError, match="candidate identity/content drift"):
        module.assert_same_candidate_pool(g1_candidates, drifted)


def test_predeclared_freeze_rule_cannot_be_weakened():
    module = _load_module()
    expected = {
        "final_strict_recall_at_4_must_improve": True,
        "final_source_recall_at_4_must_not_degrade": True,
        "equal_strict_mrr_gain_is_diagnostic_only": True,
    }

    module.validate_freeze_rule(expected)
    weakened = {**expected, "final_strict_recall_at_4_must_improve": False}
    with pytest.raises(ValueError, match="Predeclared G2 freeze rule mismatch"):
        module.validate_freeze_rule(weakened)


def _sha256(path: Path) -> str:
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


def _candidate(key: str, source: str, page: int) -> dict:
    collection, document_id = key.split("::", maxsplit=1)
    return {
        "candidate_key": key,
        "collection": collection,
        "document_id": document_id,
        "content": f"{source} 第 {page} 页证据",
        "source_file": source,
        "page_number": page,
        "chapter_title": "治疗",
    }


def test_gold_only_evaluation_compares_final_evidence_without_decoding_pilot(
    tmp_path,
):
    module = _load_module()
    g1_method = "g1"
    g2_method = "g2"
    candidate_field = "candidates_top2"
    gold_candidate = _candidate("detail_128::gold", "A.pdf", 10)
    distractor = _candidate("detail_128::other", "B.pdf", 3)
    g2_gold = {**deepcopy(gold_candidate), "graph_rerank_rank": 1}
    g2_other = {**deepcopy(distractor), "graph_rerank_rank": 2}
    results_path = tmp_path / "results.jsonl"
    _write_jsonl(
        results_path,
        [
            {
                "sample_id": "S1",
                "question": "治疗后是否需要复评？",
                "methods": {
                    g1_method: {
                        candidate_field: [distractor, gold_candidate],
                        "evidence_top4": [distractor],
                    },
                    g2_method: {
                        candidate_field: [g2_gold, g2_other],
                        "evidence_top4": [g2_gold],
                    },
                },
            }
        ],
    )
    retrieval_manifest_path = tmp_path / "retrieval_manifest.json"
    _write_json(
        retrieval_manifest_path,
        {
            "ready": True,
            "files": {
                "results": {
                    "path": results_path.name,
                    "sha256": _sha256(results_path),
                }
            },
        },
    )
    gold_path = tmp_path / "gold.jsonl"
    _write_jsonl(
        gold_path,
        [
            {
                "candidate_id": "S1",
                "question": "治疗后是否需要复评？",
                "source_filename": "A.pdf",
                "page_number": 10,
            }
        ],
    )
    pilot_path = tmp_path / "pilot.bin"
    pilot_path.write_bytes(b"\xff\xfePILOT_HASH_ONLY")
    output_dir = tmp_path / "evaluation"
    config = {
        "config_version": "fixture-v0.1",
        "summary_version": "fixture-summary-v0.1",
        "audit_version": "fixture-audit-v0.1",
        "manifest_version": "fixture-manifest-v0.1",
        "phase": "Phase 7-C1c-4e-3b-2",
        "dataset_version": "fixture-dataset",
        "kb_version": "fixture-kb",
        "expected_count": 1,
        "candidate_budget": 2,
        "candidate_output_field": candidate_field,
        "final_evidence_k": 4,
        "g1_method": g1_method,
        "g2_method": g2_method,
        "expected_results_sha256": _sha256(results_path),
        "expected_results_manifest_sha256": _sha256(retrieval_manifest_path),
        "expected_gold_sha256": _sha256(gold_path),
        "expected_pilot_test_sha256": _sha256(pilot_path),
        "sample_metrics_filename": "sample_metrics.jsonl",
        "summary_filename": "summary.json",
        "report_filename": "report.md",
        "audit_filename": "audit.json",
        "manifest_filename": "manifest.json",
        "freeze_rule": {
            "final_strict_recall_at_4_must_improve": True,
            "final_source_recall_at_4_must_not_degrade": True,
            "equal_strict_mrr_gain_is_diagnostic_only": True,
        },
        "execution_guards": {
            "validation40_gold_only": True,
            "pilot_test_content_access": False,
            "external_model_calls": False,
            "clinical_validation_claimed": False,
        },
    }

    result = module.evaluate_g1_g2(
        results_path=results_path,
        results_manifest_path=retrieval_manifest_path,
        gold_path=gold_path,
        pilot_test_path=pilot_path,
        output_dir=output_dir,
        config=config,
    )

    summary = result["summary"]
    assert summary["candidate_identity_equal_count"] == 1
    assert summary["methods"][g1_method]["candidate_strict_recall_at_2"] == 1.0
    assert summary["methods"][g2_method]["candidate_strict_recall_at_2"] == 1.0
    assert summary["methods"][g1_method]["final_strict_recall_at_4"] == 0.0
    assert summary["methods"][g2_method]["final_strict_recall_at_4"] == 1.0
    assert summary["final_strict_pair_counts"]["added"] == 1
    assert summary["freeze_recommendation"]["decision"] == "freeze_g2"
    assert result["audit"]["gold_accessed"] is True
    assert result["audit"]["pilot_test_accessed"] is False
    assert result["audit"]["external_model_calls"] == 0
    assert (output_dir / "manifest.json").exists()


def test_gold_only_evaluation_rejects_hash_drift(tmp_path):
    module = _load_module()
    results_path = tmp_path / "results.jsonl"
    results_path.write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, {"ready": True, "files": {}})
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text("{}\n", encoding="utf-8")
    pilot_path = tmp_path / "pilot.bin"
    pilot_path.write_bytes(b"sealed")
    config = {
        "g1_method": "g1",
        "g2_method": "g2",
        "candidate_budget": 2,
        "candidate_output_field": "candidates_top2",
        "final_evidence_k": 4,
        "freeze_rule": {
            "final_strict_recall_at_4_must_improve": True,
            "final_source_recall_at_4_must_not_degrade": True,
            "equal_strict_mrr_gain_is_diagnostic_only": True,
        },
        "expected_results_sha256": "0" * 64,
        "execution_guards": {
            "validation40_gold_only": True,
            "pilot_test_content_access": False,
            "external_model_calls": False,
            "clinical_validation_claimed": False,
        },
    }

    with pytest.raises(ValueError, match="G1/G2 results SHA-256 mismatch"):
        module.evaluate_g1_g2(
            results_path=results_path,
            results_manifest_path=manifest_path,
            gold_path=gold_path,
            pilot_test_path=pilot_path,
            output_dir=tmp_path / "output",
            config=config,
        )


def test_gold_only_evaluation_rejects_nonempty_output_before_reading_inputs(tmp_path):
    module = _load_module()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Output directory is not empty"):
        module.evaluate_g1_g2(
            results_path=tmp_path / "missing-results.jsonl",
            results_manifest_path=tmp_path / "missing-manifest.json",
            gold_path=tmp_path / "missing-gold.jsonl",
            pilot_test_path=tmp_path / "missing-pilot.jsonl",
            output_dir=output_dir,
            config={},
        )
