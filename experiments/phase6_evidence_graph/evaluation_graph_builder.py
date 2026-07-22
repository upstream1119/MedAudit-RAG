"""Build physically isolated evaluation graphs from runtime inference graphs."""

from __future__ import annotations

import copy
import hashlib
import json

from experiments.phase6_evidence_graph.graph_contract import (
    claim_node_id,
    decision_node_id,
    evidence_node_id,
    normalize_id_text,
    policy_node_id,
    question_node_id,
    risk_node_id,
    source_node_id,
    validate_evaluation_graph,
)


BUILDER_VERSION = "phase6a-evaluation-builder-v0.1"


def _canonical_sha256(value: dict) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_evaluation_graph(
    *,
    inference_graph: dict,
    benchmark_record: dict,
) -> dict:
    """Create an evaluation-side copy without mutating the runtime graph."""
    if not isinstance(inference_graph, dict):
        raise ValueError("inference_graph must be a dictionary")
    if not isinstance(benchmark_record, dict):
        raise ValueError("benchmark_record must be a dictionary")
    if inference_graph.get("graph_type") != "inference_graph":
        raise ValueError("inference_graph.graph_type must be inference_graph")

    sample_id = inference_graph.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("inference_graph.sample_id must be a non-empty string")
    if benchmark_record.get("sample_id") != sample_id:
        raise ValueError("benchmark sample_id does not match inference graph")
    if benchmark_record.get("freeze_status") != "frozen":
        raise ValueError("benchmark record must be frozen")

    question_nodes = [
        node for node in inference_graph.get("nodes", []) if node.get("type") == "Question"
    ]
    if len(question_nodes) != 1:
        raise ValueError("inference graph must contain exactly one Question node")
    runtime_question = question_nodes[0].get("properties", {}).get("text")
    benchmark_question = benchmark_record.get("question")
    if normalize_id_text(runtime_question) != normalize_id_text(benchmark_question):
        raise ValueError("benchmark question does not match inference graph")

    inference_versions = inference_graph.get("versions")
    if not isinstance(inference_versions, dict):
        raise ValueError("inference_graph.versions must be a dictionary")
    for field in ("dataset_version", "kb_version"):
        if benchmark_record.get(field) != inference_versions.get(field):
            raise ValueError(f"benchmark {field} does not match inference graph")

    evaluation_graph = copy.deepcopy(inference_graph)
    evaluation_graph["graph_id"] = f"evaluation_graph::{sample_id}"
    evaluation_graph["graph_type"] = "evaluation_graph"
    evaluation_graph["inference_graph_id"] = inference_graph["graph_id"]
    evaluation_graph["inference_graph_sha256"] = _canonical_sha256(inference_graph)
    evaluation_graph["method_output_status"] = "not_attached"
    evaluation_graph["evaluation_status"] = "awaiting_method_output"
    evaluation_graph["versions"]["evaluation_builder_version"] = BUILDER_VERSION
    evaluation_graph["benchmark_record_sha256"] = _canonical_sha256(benchmark_record)
    evaluation_graph["gold_evidence_status"] = benchmark_record.get(
        "gold_evidence_status"
    )

    node_ids = {node["id"] for node in evaluation_graph["nodes"]}
    edge_keys = {
        (edge["source"], edge["target"], edge["type"])
        for edge in evaluation_graph["edges"]
    }

    def add_node(node: dict) -> None:
        if node["id"] not in node_ids:
            evaluation_graph["nodes"].append(node)
            node_ids.add(node["id"])

    def add_edge(source: str, target: str, edge_type: str) -> None:
        key = (source, target, edge_type)
        if key not in edge_keys:
            evaluation_graph["edges"].append(
                {"source": source, "target": target, "type": edge_type}
            )
            edge_keys.add(key)

    question_id = question_node_id(sample_id)
    decision_type = benchmark_record.get("expected_decision")
    decision_id = decision_node_id(sample_id, decision_type)
    add_node(
        {
            "id": decision_id,
            "type": "Decision",
            "properties": {"decision_type": decision_type},
            "provenance": "benchmark_annotation",
        }
    )
    add_edge(question_id, decision_id, "QUESTION_EXPECTS_DECISION")

    for risk_label in benchmark_record.get("risk_labels", []):
        risk_id = risk_node_id(risk_label)
        add_node(
            {
                "id": risk_id,
                "type": "RiskLabel",
                "properties": {"risk_label": risk_label},
                "provenance": "benchmark_annotation",
            }
        )
        add_edge(question_id, risk_id, "QUESTION_HAS_RISK")

    for claim_role, field_name in (
        ("required", "required_claims"),
        ("allowed", "allowed_claims"),
        ("forbidden", "forbidden_claims"),
    ):
        for claim_index, claim_text in enumerate(
            benchmark_record.get(field_name, []),
            start=1,
        ):
            claim_id = claim_node_id(sample_id, claim_role, claim_index)
            add_node(
                {
                    "id": claim_id,
                    "type": "Claim",
                    "properties": {
                        "claim_text": claim_text,
                        "claim_role": claim_role,
                    },
                    "provenance": "benchmark_annotation",
                }
            )
            add_edge(question_id, claim_id, "QUESTION_HAS_GOLD_CLAIM")

    if benchmark_record.get("gold_evidence_status") == "page_span_located":
        for gold_evidence in benchmark_record.get("gold_evidence", []):
            source_id = gold_evidence.get("source_id")
            page_number = gold_evidence.get("page")
            text_span = gold_evidence.get("text_span")
            source_id_for_node = source_node_id(source_id)
            add_node(
                {
                    "id": source_id_for_node,
                    "type": "SourceDocument",
                    "properties": {
                        "source_id": source_id,
                        "source_title": gold_evidence.get("source_title"),
                        "source_type": gold_evidence.get("source_type"),
                        "publication_date": gold_evidence.get("publication_date"),
                    },
                    "provenance": "gold_evidence",
                }
            )

            gold_evidence_id = evidence_node_id(
                source_id,
                page_number,
                text_span,
            )
            add_node(
                {
                    "id": gold_evidence_id,
                    "type": "EvidenceSpan",
                    "properties": {
                        "source_id": source_id,
                        "page_number": page_number,
                        "content": text_span,
                        "supported_claim_types": gold_evidence.get(
                            "supported_claim_types",
                            [],
                        ),
                        "evidence_scope": gold_evidence.get("evidence_scope"),
                        "anchor_id": gold_evidence.get("anchor_id"),
                    },
                    "provenance": "gold_evidence",
                }
            )
            add_edge(
                gold_evidence_id,
                source_id_for_node,
                "EVIDENCE_FROM_SOURCE",
            )
            add_edge(
                question_id,
                gold_evidence_id,
                "QUESTION_HAS_GOLD_EVIDENCE",
            )
    elif benchmark_record.get("gold_evidence_status") == "policy_rule":
        for policy_evidence in benchmark_record.get("gold_evidence", []):
            rule_id = policy_evidence.get("source_id")
            policy_id = policy_node_id(rule_id)
            add_node(
                {
                    "id": policy_id,
                    "type": "PolicyRule",
                    "properties": {
                        "rule_id": rule_id,
                        "rule_type": policy_evidence.get("source_type"),
                        "rule_text": policy_evidence.get("text_span"),
                        "version": policy_evidence.get("publication_date"),
                        "source_title": policy_evidence.get("source_title"),
                    },
                    "provenance": "gold_evidence",
                }
            )
            add_edge(policy_id, decision_id, "POLICY_SUPPORTS_DECISION")

    validate_evaluation_graph(evaluation_graph)
    return evaluation_graph
