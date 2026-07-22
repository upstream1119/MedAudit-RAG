import copy
import json
from importlib import import_module
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SELECTION_PATH = REPO_ROOT / "revision/phase6/phase6a_sample_selection_v0_1.jsonl"
DEV50_PATH = REPO_ROOT / "revision/benchmark/dev50/dev50_v1_0_frozen.jsonl"


def _load_contract_module():
    try:
        return import_module("experiments.phase6_evidence_graph.graph_contract")
    except ModuleNotFoundError:
        pytest.fail("graph_contract module has not been implemented")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _runtime_only_graph() -> dict:
    return {
        "graph_id": "inference_graph::PMSQA_DEV_001",
        "graph_type": "inference_graph",
        "sample_id": "PMSQA_DEV_001",
        "versions": {
            "schema_version": "phase6a-pedimedkg-schema-v0.1",
            "dataset_version": "dev50-v1.0",
            "kb_version": "KB-medium-v1",
        },
        "build_status": "success",
        "failure_reason": None,
        "nodes": [
            {
                "id": "question::PMSQA_DEV_001",
                "type": "Question",
                "properties": {"text": "测试问题"},
                "provenance": "runtime_question",
            }
        ],
        "edges": [],
    }


def _valid_evidence_graph(contract) -> dict:
    graph = _runtime_only_graph()
    evidence_id = contract.evidence_node_id("SRC-002", 26, "48-72小时无改善应再次评估")
    graph["nodes"].extend(
        [
            {
                "id": "source::SRC-002",
                "type": "SourceDocument",
                "properties": {
                    "source_id": "SRC-002",
                    "source_title": "儿童社区获得性肺炎诊疗规范（2019年版）",
                },
                "provenance": "source_registry",
            },
            {
                "id": evidence_id,
                "type": "EvidenceSpan",
                "properties": {
                    "source_id": "SRC-002",
                    "page_number": 26,
                    "content": "48-72小时无改善应再次评估",
                },
                "provenance": "runtime_retrieval",
            },
        ]
    )
    graph["edges"].append(
        {
            "source": evidence_id,
            "target": "source::SRC-002",
            "type": "EVIDENCE_FROM_SOURCE",
        }
    )
    return graph


def test_equivalent_text_produces_same_evidence_id():
    contract = _load_contract_module()

    first = contract.evidence_node_id("SRC-001", 14, "10 mg/kg   qd")
    second = contract.evidence_node_id("SRC-001", 14, "  10 MG/KG qd  ")

    assert first == second
    assert first.startswith("evidence::SRC-001::p14::")


def test_medical_numeric_or_frequency_change_produces_different_id():
    contract = _load_contract_module()

    baseline = contract.evidence_node_id("SRC-001", 14, "10 mg/kg qd")
    dose_changed = contract.evidence_node_id("SRC-001", 14, "5 mg/kg qd")
    frequency_changed = contract.evidence_node_id("SRC-001", 14, "10 mg/kg bid")

    assert baseline != dose_changed
    assert baseline != frequency_changed


def test_selection_manifest_accepts_matching_frozen_snapshot():
    contract = _load_contract_module()

    contract.validate_selection_manifest(
        _read_jsonl(SELECTION_PATH),
        _read_jsonl(DEV50_PATH),
    )


def test_selection_manifest_rejects_duplicate_sample_id():
    contract = _load_contract_module()
    selection_rows = _read_jsonl(SELECTION_PATH)
    selection_rows.append(copy.deepcopy(selection_rows[0]))

    with pytest.raises(ValueError, match="duplicate sample_id"):
        contract.validate_selection_manifest(selection_rows, _read_jsonl(DEV50_PATH))


def test_selection_manifest_rejects_snapshot_mismatch():
    contract = _load_contract_module()
    selection_rows = _read_jsonl(SELECTION_PATH)
    selection_rows[0]["expected_decision"] = "boundary_refusal"

    with pytest.raises(ValueError, match="snapshot mismatch"):
        contract.validate_selection_manifest(selection_rows, _read_jsonl(DEV50_PATH))


def test_inference_graph_rejects_nested_gold_key():
    contract = _load_contract_module()
    graph = _runtime_only_graph()
    graph["nodes"][0]["properties"]["expected_decision"] = "answer"

    with pytest.raises(ValueError, match="gold-only key"):
        contract.assert_no_gold_only_content(graph)


def test_inference_graph_rejects_gold_relation():
    contract = _load_contract_module()
    graph = _runtime_only_graph()
    graph["edges"].append(
        {
            "source": "question::PMSQA_DEV_001",
            "target": "evidence::SRC-002::p26::abc",
            "type": "QUESTION_HAS_GOLD_EVIDENCE",
        }
    )

    with pytest.raises(ValueError, match="gold-only relation"):
        contract.assert_no_gold_only_content(graph)


def test_inference_graph_rejects_annotation_provenance():
    contract = _load_contract_module()
    graph = _runtime_only_graph()
    graph["nodes"][0]["provenance"] = "dev50_annotation"

    with pytest.raises(ValueError, match="gold-only provenance"):
        contract.assert_no_gold_only_content(graph)


def test_inference_graph_accepts_runtime_only_content():
    contract = _load_contract_module()

    contract.assert_no_gold_only_content(_runtime_only_graph())


def test_node_id_helpers_follow_declared_patterns():
    contract = _load_contract_module()

    assert contract.question_node_id("PMSQA_DEV_001") == "question::PMSQA_DEV_001"
    assert contract.source_node_id("SRC-002") == "source::SRC-002"
    assert contract.entity_node_id("Drug", "阿奇霉素").startswith("entity::drug::")
    assert contract.constraint_node_id(
        "PMSQA_DEV_002", "frequency", "qd"
    ).startswith("constraint::PMSQA_DEV_002::frequency::")


def test_inference_graph_schema_accepts_valid_runtime_graph():
    contract = _load_contract_module()

    contract.validate_inference_graph(_valid_evidence_graph(contract), {"SRC-002"})


def test_inference_graph_schema_rejects_unadmitted_source():
    contract = _load_contract_module()

    with pytest.raises(ValueError, match="unadmitted source"):
        contract.validate_inference_graph(_valid_evidence_graph(contract), {"SRC-001"})


def test_inference_graph_schema_rejects_dangling_edge():
    contract = _load_contract_module()
    graph = _valid_evidence_graph(contract)
    graph["edges"][0]["target"] = "source::SRC-999"

    with pytest.raises(ValueError, match="edge endpoint"):
        contract.validate_inference_graph(graph, {"SRC-002"})


def test_inference_graph_schema_rejects_duplicate_node_id():
    contract = _load_contract_module()
    graph = _valid_evidence_graph(contract)
    graph["nodes"].append(copy.deepcopy(graph["nodes"][0]))

    with pytest.raises(ValueError, match="duplicate node id"):
        contract.validate_inference_graph(graph, {"SRC-002"})


def test_empty_evidence_graph_requires_failure_reason():
    contract = _load_contract_module()
    graph = _runtime_only_graph()
    graph["build_status"] = "empty_evidence"

    with pytest.raises(ValueError, match="failure_reason"):
        contract.validate_inference_graph(graph, set())

    graph["failure_reason"] = "retrieval_returned_no_admitted_evidence"
    contract.validate_inference_graph(graph, set())
