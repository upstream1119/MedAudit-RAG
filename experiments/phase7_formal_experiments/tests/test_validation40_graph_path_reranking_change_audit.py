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
    / "validation40_graph_path_reranking_change_audit.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "G2 change audit module is missing"
    spec = importlib.util.spec_from_file_location("g2_change_audit", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(
    key: str,
    *,
    source: str,
    page: int,
    origin: str = "baseline",
    graph_rank: int | None = None,
    path_score: float = 0.0,
) -> dict:
    candidate = {
        "candidate_key": key,
        "source_file": source,
        "page_number": page,
        "content": f"evidence-{key}",
        "candidate_origin": origin,
        "post_rerank_rank": graph_rank or 1,
        "reranker_score": 0.5,
    }
    if graph_rank is not None:
        candidate.update(
            {
                "g2_path_eligible": True,
                "graph_rerank_rank": graph_rank,
                "reranker_rank_before_graph": graph_rank + 1,
                "graph_path_score": path_score,
                "graph_path_score_components": {
                    "source_specificity": path_score,
                    "matched_type_coverage": 0.5,
                    "content_type_coverage": 0.5,
                    "route_reciprocal": 0.25,
                },
                "g2_graph_path_trace": {"trace_version": "trace-v0.1"},
            }
        )
    return candidate


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


def test_classify_final_change_separates_membership_from_order_only():
    module = _load_module()
    a = _candidate("a", source="A.pdf", page=1)
    b = _candidate("b", source="B.pdf", page=2)
    c = _candidate("c", source="C.pdf", page=3)

    membership = module.classify_final_change([a, b], [a, c])
    order_only = module.classify_final_change([a, b], [b, a])

    assert membership["change_type"] == "membership_changed"
    assert membership["added_candidate_keys"] == ["c"]
    assert membership["removed_candidate_keys"] == ["b"]
    assert order_only["change_type"] == "order_only"
    assert order_only["added_candidate_keys"] == []
    assert order_only["removed_candidate_keys"] == []


@pytest.mark.parametrize(
    ("g1_hit", "g2_hit", "g1_rank", "g2_rank", "expected"),
    [
        (False, True, None, 4, "strict_hit_added"),
        (True, False, 4, None, "strict_hit_lost"),
        (True, True, 3, 2, "strict_rank_improved"),
        (True, True, 1, 2, "strict_rank_worsened"),
        (True, True, 2, 2, "strict_metric_unchanged"),
        (False, False, None, None, "strict_metric_unchanged"),
    ],
)
def test_classify_gold_effect_is_directional(
    g1_hit, g2_hit, g1_rank, g2_rank, expected
):
    module = _load_module()
    result = module.classify_gold_effect(
        {
            "g1_final_strict_hit": g1_hit,
            "g2_final_strict_hit": g2_hit,
            "g1_final_strict_rank": g1_rank,
            "g2_final_strict_rank": g2_rank,
            "g1_final_strict_mrr": 0.0 if g1_rank is None else 1 / g1_rank,
            "g2_final_strict_mrr": 0.0 if g2_rank is None else 1 / g2_rank,
        },
        level="final",
    )

    assert result["effect"] == expected


def test_audit_sample_records_observed_path_features_without_causal_claim():
    module = _load_module()
    old = _candidate("old", source="A.pdf", page=1)
    promoted = _candidate(
        "gold",
        source="Gold.pdf",
        page=10,
        origin="graph_expansion",
        graph_rank=4,
        path_score=0.75,
    )
    result_row = {
        "sample_id": "S1",
        "question": "Q",
        "graph_rerank_audit": {"evidence_order_changed": True},
        "methods": {
            "g1": {
                "candidates_top24": [old, promoted],
                "evidence_top4": [old],
            },
            "g2": {
                "candidates_top24": [promoted, old],
                "evidence_top4": [promoted],
            },
        },
    }
    metric_row = {
        "sample_id": "S1",
        "question": "Q",
        "gold_source_filename": "Gold.pdf",
        "gold_page_number": 10,
        "g1_candidate_strict_mrr": 0.5,
        "g2_candidate_strict_mrr": 1.0,
        "g1_final_strict_hit": False,
        "g2_final_strict_hit": True,
        "g1_final_strict_rank": None,
        "g2_final_strict_rank": 1,
        "g1_final_strict_mrr": 0.0,
        "g2_final_strict_mrr": 1.0,
    }

    audit = module.audit_changed_sample(
        result_row,
        metric_row,
        g1_method="g1",
        g2_method="g2",
    )

    assert audit["final_change"]["change_type"] == "membership_changed"
    assert audit["final_gold_effect"]["effect"] == "strict_hit_added"
    assert audit["added_evidence"][0]["graph_path_score"] == 0.75
    assert audit["added_evidence"][0]["candidate_origin"] == "graph_expansion"
    assert audit["interpretation_scope"] == "observational_structural_audit_only"
    assert "caused_by" not in audit


def test_run_change_audit_writes_deterministic_guarded_outputs(tmp_path):
    module = _load_module()
    old = _candidate("old", source="A.pdf", page=1)
    gold = _candidate(
        "gold",
        source="Gold.pdf",
        page=10,
        origin="graph_expansion",
        graph_rank=4,
        path_score=0.75,
    )
    results_path = tmp_path / "results.jsonl"
    results_manifest_path = tmp_path / "results_manifest.json"
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_manifest_path = tmp_path / "metrics_manifest.json"
    pilot_path = tmp_path / "pilot.jsonl"
    _write_jsonl(
        results_path,
        [
            {
                "sample_id": "S1",
                "question": "Q",
                "graph_rerank_audit": {
                    "candidate_order_changed": True,
                    "evidence_order_changed": True,
                },
                "methods": {
                    "g1": {
                        "candidates_top24": [old, gold],
                        "evidence_top4": [old],
                    },
                    "g2": {
                        "candidates_top24": [gold, old],
                        "evidence_top4": [gold],
                    },
                },
            }
        ],
    )
    _write_json(results_manifest_path, {"result": "locked"})
    _write_jsonl(
        metrics_path,
        [
            {
                "sample_id": "S1",
                "question": "Q",
                "gold_source_filename": "Gold.pdf",
                "gold_page_number": 10,
                "g1_candidate_strict_hit": True,
                "g2_candidate_strict_hit": True,
                "g1_candidate_strict_rank": 2,
                "g2_candidate_strict_rank": 1,
                "g1_candidate_strict_mrr": 0.5,
                "g2_candidate_strict_mrr": 1.0,
                "g1_final_strict_hit": False,
                "g2_final_strict_hit": True,
                "g1_final_strict_rank": None,
                "g2_final_strict_rank": 1,
                "g1_final_strict_mrr": 0.0,
                "g2_final_strict_mrr": 1.0,
            }
        ],
    )
    _write_json(metrics_manifest_path, {"metrics": "locked"})
    pilot_path.write_text('{"must_not_be_parsed": true}\n', encoding="utf-8")
    config = {
        "config_version": "change-audit-test-v0.1",
        "phase": "Phase 7-C1c-4e-3b-3",
        "expected_count": 1,
        "expected_changed_evidence_count": 1,
        "expected_audited_sample_count": 1,
        "expected_results_sha256": _sha(results_path),
        "expected_results_manifest_sha256": _sha(results_manifest_path),
        "expected_sample_metrics_sha256": _sha(metrics_path),
        "expected_metrics_manifest_sha256": _sha(metrics_manifest_path),
        "expected_pilot_test_sha256": _sha(pilot_path),
        "g1_method": "g1",
        "g2_method": "g2",
        "samples_filename": "samples.jsonl",
        "summary_filename": "summary.json",
        "report_filename": "report.md",
        "audit_filename": "audit.json",
        "manifest_filename": "manifest.json",
        "execution_guards": {
            "validation40_gold_only": True,
            "pilot_test_content_access": False,
            "external_model_calls": False,
            "causal_claims": False,
        },
    }

    first = module.run_change_audit(
        results_path=results_path,
        results_manifest_path=results_manifest_path,
        sample_metrics_path=metrics_path,
        metrics_manifest_path=metrics_manifest_path,
        pilot_test_path=pilot_path,
        output_dir=tmp_path / "audit-a",
        config=config,
    )
    second = module.run_change_audit(
        results_path=results_path,
        results_manifest_path=results_manifest_path,
        sample_metrics_path=metrics_path,
        metrics_manifest_path=metrics_manifest_path,
        pilot_test_path=pilot_path,
        output_dir=tmp_path / "audit-b",
        config=config,
    )

    assert first["summary"]["final_evidence_changed_count"] == 1
    assert first["summary"]["audited_sample_count"] == 1
    assert first["summary"]["strict_hit_added_count"] == 1
    assert first["audit"]["pilot_test_accessed"] is False
    assert first["audit"]["external_model_calls"] == 0
    assert first["audit"]["causal_claims_made"] is False
    assert set(first["manifest"]["files"]) == {
        "samples",
        "summary",
        "report",
        "audit",
    }
    for filename in (
        "samples.jsonl",
        "summary.json",
        "report.md",
        "audit.json",
        "manifest.json",
    ):
        assert (tmp_path / "audit-a" / filename).read_bytes() == (
            tmp_path / "audit-b" / filename
        ).read_bytes()


def test_run_change_audit_rejects_hash_drift_before_writing(tmp_path):
    module = _load_module()
    paths = {
        name: tmp_path / f"{name}.json"
        for name in ("results", "results_manifest", "metrics", "metrics_manifest")
    }
    for path in paths.values():
        path.write_text("{}\n", encoding="utf-8")
    pilot_path = tmp_path / "pilot.jsonl"
    pilot_path.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="results SHA-256 mismatch"):
        module.run_change_audit(
            results_path=paths["results"],
            results_manifest_path=paths["results_manifest"],
            sample_metrics_path=paths["metrics"],
            metrics_manifest_path=paths["metrics_manifest"],
            pilot_test_path=pilot_path,
            output_dir=tmp_path / "must-not-exist",
            config={
                "expected_results_sha256": "0" * 64,
                "expected_results_manifest_sha256": _sha(paths["results_manifest"]),
                "expected_sample_metrics_sha256": _sha(paths["metrics"]),
                "expected_metrics_manifest_sha256": _sha(paths["metrics_manifest"]),
                "expected_pilot_test_sha256": _sha(pilot_path),
                "execution_guards": {
                    "validation40_gold_only": True,
                    "pilot_test_content_access": False,
                    "external_model_calls": False,
                    "causal_claims": False,
                },
            },
        )

    assert not (tmp_path / "must-not-exist").exists()


def test_run_change_audit_rejects_unsafe_execution_guards(tmp_path):
    module = _load_module()

    with pytest.raises(ValueError, match="pilot_test_content_access"):
        module.validate_execution_guards(
            {
                "validation40_gold_only": True,
                "pilot_test_content_access": True,
                "external_model_calls": False,
                "causal_claims": False,
            }
        )
