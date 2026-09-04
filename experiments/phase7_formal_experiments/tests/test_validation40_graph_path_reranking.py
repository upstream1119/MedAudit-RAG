from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "validation40_graph_path_reranking.py"
)
CONFIG_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "configs"
    / "validation40_graph_path_reranking_v0_1.json"
)


def _load_module():
    assert MODULE_PATH.exists(), "Validation40 graph path reranking module is missing"
    spec = importlib.util.spec_from_file_location(
        "validation40_graph_path_reranking", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate() -> dict:
    return {
        "candidate_key": "detail_128::baseline",
        "collection": "detail_128",
        "document_id": "baseline",
        "content": "MPP 在限定情况下可考虑糖皮质激素。",
        "source_file": "MPP诊疗指南.pdf",
        "page_number": 1,
        "candidate_origin": "baseline",
        "post_rerank_rank": 1,
        "pre_rerank_rank": 1,
        "reranker_score": 3.0,
    }


def test_gold_free_runner_adds_g2_without_changing_the_g1_candidate_set(tmp_path):
    module = _load_module()
    candidate = _candidate()
    f_method = "f24"
    g1_method = "g1"
    g2_method = "g2"
    results_path = tmp_path / "input" / "results.jsonl"
    _write_jsonl(
        results_path,
        [
            {
                "sample_id": "S1",
                "question": "MPP 糖皮质激素治疗是否有依据？",
                "methods": {
                    f_method: {
                        "candidates_top24": [candidate],
                        "evidence_top4": [candidate],
                        "dedup_audit": {"output_count": 1},
                    },
                    g1_method: {
                        "candidates_top24": [candidate],
                        "evidence_top4": [candidate],
                        "dedup_audit": {"output_count": 1},
                    },
                },
            }
        ],
    )
    input_manifest_path = results_path.parent / "manifest.json"
    _write_json(
        input_manifest_path,
        {
            "ready": True,
            "files": {"results": {"path": results_path.name, "sha256": _sha256(results_path)}},
        },
    )
    graph_index_path = tmp_path / "graph" / "graph.json"
    graph_candidate = {
        key: value
        for key, value in candidate.items()
        if key not in {
            "candidate_origin",
            "post_rerank_rank",
            "pre_rerank_rank",
            "reranker_score",
        }
    }
    _write_json(graph_index_path, {"candidates": {candidate["candidate_key"]: graph_candidate}})
    graph_manifest_path = graph_index_path.parent / "manifest.json"
    _write_json(
        graph_manifest_path,
        {
            "ready": True,
            "files": {"graph_index": {"path": graph_index_path.name, "sha256": _sha256(graph_index_path)}},
        },
    )
    lexicon_path = tmp_path / "lexicon.json"
    _write_json(
        lexicon_path,
        {
            "lexicon_version": "fixture-v0.2",
            "entries": [
                {
                    "constraint_type": "clinical_condition",
                    "normalized_value": "mycoplasma_pneumoniae_pneumonia",
                    "aliases": ["MPP"],
                    "strong_anchor": True,
                },
                {
                    "constraint_type": "medication_class",
                    "normalized_value": "corticosteroid",
                    "aliases": ["糖皮质激素"],
                    "strong_anchor": True,
                },
            ],
        },
    )
    routing_audit_path = tmp_path / "routing" / "audit.json"
    _write_json(
        routing_audit_path,
        {
            "audit_version": "fixture-routing-audit-v0.1",
            "question_details": [
                {
                    "sample_id": "S1",
                    "query_constraint_count": 2,
                    "selected_paths": [
                        {
                            "candidate_key": candidate["candidate_key"],
                            "source_file": candidate["source_file"],
                            "page_number": candidate["page_number"],
                            "graph_route_rank": 1,
                            "graph_source_condition_tier": 0,
                            "graph_source_condition_tier_label": "source_condition_exact",
                            "graph_path_constraint_types": [
                                "clinical_condition",
                                "medication_class",
                            ],
                        }
                    ],
                }
            ],
        },
    )
    routing_manifest_path = routing_audit_path.parent / "manifest.json"
    _write_json(
        routing_manifest_path,
        {"router_version": "phase7-runtime-graph-path-router-v0.1"},
    )
    pilot_path = tmp_path / "pilot.bin"
    pilot_path.write_bytes(b"\xff\xfePILOT_HASH_ONLY")
    output_dir = tmp_path / "output"
    config = {
        "config_version": "fixture-v0.1",
        "audit_version": "fixture-audit-v0.1",
        "manifest_version": "fixture-manifest-v0.1",
        "phase": "Phase 7-C1c-4e-3b-1",
        "dataset_version": "fixture-dataset",
        "kb_version": "fixture-kb",
        "expected_count": 1,
        "candidate_budget": 24,
        "candidate_output_field": "candidates_top24",
        "final_evidence_k": 4,
        "f_method": f_method,
        "g1_method": g1_method,
        "g2_method": g2_method,
        "max_rank_shift": 2.0,
        "allow_specific_condition_class_path": True,
        "tier_weights": {
            "source_condition_exact": 1.0,
            "content_condition_generic_source": 0.75,
            "query_without_condition": 0.5,
            "content_condition_other_source": 0.25,
            "context_only_condition": 0.0,
        },
        "dedup_ngram_size": 3,
        "dedup_overlap_threshold": 0.75,
        "expected_input_results_sha256": _sha256(results_path),
        "expected_input_manifest_sha256": _sha256(input_manifest_path),
        "expected_graph_manifest_sha256": _sha256(graph_manifest_path),
        "expected_lexicon_sha256": _sha256(lexicon_path),
        "expected_routing_audit_sha256": _sha256(routing_audit_path),
        "expected_routing_manifest_sha256": _sha256(routing_manifest_path),
        "expected_pilot_test_sha256": _sha256(pilot_path),
        "expected_path_eligible_count": 1,
        "expected_existing_trace_count": 0,
        "expected_path_eligible_origin_counts": {"baseline": 1},
        "results_filename": "results.jsonl",
        "audit_filename": "audit.json",
        "manifest_filename": "manifest.json",
        "execution_guards": {
            "validation40_only": True,
            "gold_access": False,
            "pilot_test_content_access": False,
            "external_model_calls": False,
            "clinical_validation_claimed": False,
        },
    }

    result = module.run_graph_path_reranking(
        input_results_path=results_path,
        input_manifest_path=input_manifest_path,
        graph_manifest_path=graph_manifest_path,
        lexicon_path=lexicon_path,
        routing_audit_path=routing_audit_path,
        routing_manifest_path=routing_manifest_path,
        pilot_test_path=pilot_path,
        output_dir=output_dir,
        config=config,
    )

    output_row = json.loads((output_dir / "results.jsonl").read_text(encoding="utf-8"))
    g1_keys = [row["candidate_key"] for row in output_row["methods"][g1_method]["candidates_top24"]]
    g2_candidates = output_row["methods"][g2_method]["candidates_top24"]
    assert [row["candidate_key"] for row in g2_candidates] == g1_keys
    assert g2_candidates[0]["g2_path_eligible"] is True
    assert g2_candidates[0]["g2_graph_path_trace"]
    assert result["path_eligible_count"] == 1
    assert result["zero_shift_identity_count"] == 1
    assert result["pilot_test_accessed"] is False
    assert result["gold_accessed"] is False


def test_repository_config_freezes_the_gold_free_g2_contract():
    assert CONFIG_PATH.exists(), "G2 configuration is missing"
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["phase"] == "Phase 7-C1c-4e-3b-1"
    assert config["candidate_budget"] == 24
    assert config["max_rank_shift"] == 2.0
    assert config["expected_path_eligible_count"] == 85
    assert config["expected_existing_trace_count"] == 69
    assert config["expected_path_eligible_origin_counts"] == {
        "baseline": 16,
        "graph_expansion": 69,
    }
    assert config["execution_guards"]["gold_access"] is False
    assert config["execution_guards"]["pilot_test_content_access"] is False
    assert config["execution_guards"]["external_model_calls"] is False
