"""Deterministic contracts shared by Phase 6 evidence-graph builders."""

from __future__ import annotations

import hashlib
import re
import unicodedata


SELECTION_SNAPSHOT_FIELDS = (
    "scenario_type",
    "expected_decision",
    "fact_cluster_id",
    "gold_evidence_status",
)

GOLD_ONLY_KEYS = frozenset(
    {
        "expected_decision",
        "risk_labels",
        "gold_evidence",
        "gold_evidence_status",
        "gold_evidence_hint",
        "required_claims",
        "allowed_claims",
        "forbidden_claims",
        "drug_entities",
        "disease_entities",
        "scenario_type",
    }
)
GOLD_ONLY_RELATIONS = frozenset(
    {
        "QUESTION_EXPECTS_DECISION",
        "QUESTION_HAS_GOLD_EVIDENCE",
        "QUESTION_HAS_GOLD_CLAIM",
        "QUESTION_HAS_RISK",
        "CLAIM_MATCHES_REQUIRED",
        "CLAIM_MATCHES_ALLOWED",
        "CLAIM_MATCHES_FORBIDDEN",
    }
)
GOLD_ONLY_PROVENANCE = frozenset(
    {"dev50_annotation", "benchmark_annotation", "gold", "gold_evidence"}
)
GOLD_ONLY_CLAIM_ROLES = frozenset({"required", "allowed", "forbidden", "gold"})


def normalize_id_text(text: str) -> str:
    """Normalize formatting variance while preserving medical values and units."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("ID text must be a non-empty string")
    normalized = unicodedata.normalize("NFKC", text).strip().lower()
    return re.sub(r"\s+", " ", normalized)


def _stable_text_hash(text: str, length: int = 16) -> str:
    return hashlib.sha256(normalize_id_text(text).encode("utf-8")).hexdigest()[:length]


def _require_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _identifier_slug(value: str, field_name: str) -> str:
    normalized = normalize_id_text(_require_identifier(value, field_name))
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    if not slug:
        raise ValueError(f"{field_name} must contain an ASCII identifier")
    return slug


def question_node_id(sample_id: str) -> str:
    return f"question::{_require_identifier(sample_id, 'sample_id')}"


def entity_node_id(entity_type: str, normalized_name: str) -> str:
    entity_slug = _identifier_slug(entity_type, "entity_type")
    return f"entity::{entity_slug}::{_stable_text_hash(normalized_name)}"


def source_node_id(source_id: str) -> str:
    return f"source::{_require_identifier(source_id, 'source_id')}"


def constraint_node_id(sample_id: str, constraint_type: str, value: str) -> str:
    sample = _require_identifier(sample_id, "sample_id")
    constraint_slug = _identifier_slug(constraint_type, "constraint_type")
    return f"constraint::{sample}::{constraint_slug}::{_stable_text_hash(value)}"


def evidence_node_id(source_id: str, page_number: int, text_span: str) -> str:
    source = _require_identifier(source_id, "source_id")
    if not isinstance(page_number, int) or page_number < 1:
        raise ValueError("page_number must be a positive integer")
    return f"evidence::{source}::p{page_number}::{_stable_text_hash(text_span)}"


def claim_node_id(sample_id: str, claim_role: str, claim_index: int) -> str:
    sample = _require_identifier(sample_id, "sample_id")
    role = _identifier_slug(claim_role, "claim_role")
    if role not in GOLD_ONLY_CLAIM_ROLES:
        raise ValueError(f"unsupported gold claim role: {claim_role}")
    if not isinstance(claim_index, int) or claim_index < 1:
        raise ValueError("claim_index must be a positive integer")
    return f"claim::{sample}::gold::{role}::{claim_index}"


def risk_node_id(risk_label: str) -> str:
    return f"risk::{_identifier_slug(risk_label, 'risk_label')}"


def decision_node_id(sample_id: str, decision_type: str) -> str:
    sample = _require_identifier(sample_id, "sample_id")
    decision = _identifier_slug(decision_type, "decision_type")
    return f"decision::{sample}::expected::{decision}"


def policy_node_id(rule_id: str) -> str:
    return f"policy::{_require_identifier(rule_id, 'rule_id')}"


def validate_selection_manifest(
    selection_rows: list[dict],
    dev50_records: list[dict],
) -> None:
    """Verify that the frozen selection snapshot still matches Dev50."""
    dev50_by_id: dict[str, dict] = {}
    for record in dev50_records:
        sample_id = record.get("sample_id")
        if not sample_id:
            raise ValueError("Dev50 record is missing sample_id")
        if sample_id in dev50_by_id:
            raise ValueError(f"duplicate Dev50 sample_id: {sample_id}")
        dev50_by_id[sample_id] = record

    seen_sample_ids: set[str] = set()
    seen_ranks: set[int] = set()
    for row in selection_rows:
        sample_id = row.get("sample_id")
        if sample_id in seen_sample_ids:
            raise ValueError(f"duplicate sample_id in selection manifest: {sample_id}")
        seen_sample_ids.add(sample_id)

        selection_rank = row.get("selection_rank")
        if selection_rank in seen_ranks:
            raise ValueError(f"duplicate selection_rank: {selection_rank}")
        seen_ranks.add(selection_rank)

        if sample_id not in dev50_by_id:
            raise ValueError(f"selection sample_id not found in Dev50: {sample_id}")
        record = dev50_by_id[sample_id]

        for field in SELECTION_SNAPSHOT_FIELDS:
            if row.get(field) != record.get(field):
                raise ValueError(
                    f"snapshot mismatch for {sample_id}.{field}: "
                    f"selection={row.get(field)!r}, dev50={record.get(field)!r}"
                )

        if row.get("selected_from_dataset_version") != record.get("dataset_version"):
            raise ValueError(f"snapshot mismatch for {sample_id}.dataset_version")
        if row.get("kb_version") != record.get("kb_version"):
            raise ValueError(f"snapshot mismatch for {sample_id}.kb_version")
        if row.get("selection_status") != "frozen":
            raise ValueError(f"selection row is not frozen: {sample_id}")
        if row.get("graph_schema_version") != "phase6a-pedimedkg-schema-v0.1":
            raise ValueError(f"unexpected graph schema version: {sample_id}")


def assert_no_gold_only_content(graph: dict) -> None:
    """Reject evaluation-only annotations anywhere in an inference graph."""

    def visit(value: object, path: str) -> None:
        if isinstance(value, dict):
            for raw_key, item in value.items():
                key = str(raw_key).strip().lower()
                item_path = f"{path}.{raw_key}"
                if key in GOLD_ONLY_KEYS:
                    raise ValueError(f"gold-only key found at {item_path}")

                if key in {"type", "relation_type"} and isinstance(item, str):
                    if item.strip().upper() in GOLD_ONLY_RELATIONS:
                        raise ValueError(f"gold-only relation found at {item_path}")

                if key == "provenance" and isinstance(item, str):
                    if item.strip().lower() in GOLD_ONLY_PROVENANCE:
                        raise ValueError(f"gold-only provenance found at {item_path}")

                if key in {"claim_role", "role"} and isinstance(item, str):
                    if item.strip().lower() in GOLD_ONLY_CLAIM_ROLES:
                        raise ValueError(f"gold-only claim role found at {item_path}")

                visit(item, item_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(graph, "graph")


def validate_inference_graph(graph: dict, admitted_source_ids: set[str]) -> None:
    """Validate the minimal runtime-only inference graph contract."""
    if not isinstance(graph, dict):
        raise ValueError("inference graph must be a dictionary")
    assert_no_gold_only_content(graph)

    if graph.get("graph_type") != "inference_graph":
        raise ValueError("graph_type must be inference_graph")
    sample_id = _require_identifier(graph.get("sample_id"), "sample_id")
    if graph.get("graph_id") != f"inference_graph::{sample_id}":
        raise ValueError("graph_id does not match sample_id")

    versions = graph.get("versions")
    if not isinstance(versions, dict):
        raise ValueError("versions must be a dictionary")
    for field in ("schema_version", "dataset_version", "kb_version"):
        _require_identifier(versions.get(field), f"versions.{field}")
    if versions["schema_version"] != "phase6a-pedimedkg-schema-v0.1":
        raise ValueError("unsupported schema_version")

    build_status = graph.get("build_status")
    if build_status not in {"success", "empty_evidence", "failed"}:
        raise ValueError("unsupported build_status")

    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("nodes and edges must be lists")

    nodes_by_id: dict[str, dict] = {}
    question_nodes: list[dict] = []
    source_nodes_by_source_id: dict[str, dict] = {}
    evidence_nodes: list[dict] = []

    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("every node must be a dictionary")
        node_id = _require_identifier(node.get("id"), "node.id")
        if node_id in nodes_by_id:
            raise ValueError(f"duplicate node id: {node_id}")
        nodes_by_id[node_id] = node

        node_type = _require_identifier(node.get("type"), f"{node_id}.type")
        properties = node.get("properties")
        if not isinstance(properties, dict):
            raise ValueError(f"node properties must be a dictionary: {node_id}")

        if node_type == "Question":
            question_nodes.append(node)
        elif node_type == "SourceDocument":
            source_id = _require_identifier(properties.get("source_id"), "source_id")
            if source_id not in admitted_source_ids:
                raise ValueError(f"unadmitted source: {source_id}")
            if node_id != source_node_id(source_id):
                raise ValueError(f"source node id mismatch: {node_id}")
            source_nodes_by_source_id[source_id] = node
        elif node_type == "EvidenceSpan":
            evidence_nodes.append(node)

    if len(question_nodes) != 1:
        raise ValueError("inference graph must contain exactly one Question node")
    if question_nodes[0]["id"] != question_node_id(sample_id):
        raise ValueError("Question node id does not match sample_id")

    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError("every edge must be a dictionary")
        source = _require_identifier(edge.get("source"), "edge.source")
        target = _require_identifier(edge.get("target"), "edge.target")
        _require_identifier(edge.get("type"), "edge.type")
        if source not in nodes_by_id or target not in nodes_by_id:
            raise ValueError(f"edge endpoint does not exist: {source} -> {target}")

    for evidence_node in evidence_nodes:
        properties = evidence_node["properties"]
        source_id = _require_identifier(properties.get("source_id"), "source_id")
        if source_id not in admitted_source_ids:
            raise ValueError(f"unadmitted source: {source_id}")
        if source_id not in source_nodes_by_source_id:
            raise ValueError(f"EvidenceSpan has no SourceDocument: {source_id}")

        page_number = properties.get("page_number")
        content = properties.get("content")
        expected_id = evidence_node_id(source_id, page_number, content)
        if evidence_node["id"] != expected_id:
            raise ValueError(f"evidence node id mismatch: {evidence_node['id']}")

        expected_source_node_id = source_node_id(source_id)
        has_source_edge = any(
            edge.get("source") == evidence_node["id"]
            and edge.get("target") == expected_source_node_id
            and edge.get("type") == "EVIDENCE_FROM_SOURCE"
            for edge in edges
        )
        if not has_source_edge:
            raise ValueError(f"EvidenceSpan is missing EVIDENCE_FROM_SOURCE edge: {evidence_node['id']}")

    if not evidence_nodes:
        if build_status != "empty_evidence":
            raise ValueError("graph without evidence must use empty_evidence build_status")
        if not isinstance(graph.get("failure_reason"), str) or not graph["failure_reason"].strip():
            raise ValueError("empty evidence graph requires failure_reason")
    elif build_status != "success":
        raise ValueError("graph with evidence must use success build_status")


def validate_evaluation_graph(graph: dict) -> None:
    """Validate an evaluation-side graph without treating gold data as runtime input."""
    if not isinstance(graph, dict):
        raise ValueError("evaluation graph must be a dictionary")
    if graph.get("graph_type") != "evaluation_graph":
        raise ValueError("graph_type must be evaluation_graph")

    sample_id = _require_identifier(graph.get("sample_id"), "sample_id")
    if graph.get("graph_id") != f"evaluation_graph::{sample_id}":
        raise ValueError("evaluation graph_id does not match sample_id")
    if graph.get("inference_graph_id") != f"inference_graph::{sample_id}":
        raise ValueError("inference_graph_id does not match sample_id")

    parent_sha256 = graph.get("inference_graph_sha256")
    if not isinstance(parent_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", parent_sha256):
        raise ValueError("inference_graph_sha256 must be a lowercase SHA-256 digest")

    versions = graph.get("versions")
    if not isinstance(versions, dict):
        raise ValueError("versions must be a dictionary")
    for field in (
        "schema_version",
        "dataset_version",
        "kb_version",
        "evaluation_builder_version",
    ):
        _require_identifier(versions.get(field), f"versions.{field}")
    if versions["schema_version"] != "phase6a-pedimedkg-schema-v0.1":
        raise ValueError("unsupported schema_version")

    if graph.get("method_output_status") != "not_attached":
        raise ValueError("unsupported method_output_status")
    if graph.get("evaluation_status") != "awaiting_method_output":
        raise ValueError("unsupported evaluation_status")

    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("nodes and edges must be lists")

    nodes_by_id: dict[str, dict] = {}
    nodes_by_type: dict[str, list[dict]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("every node must be a dictionary")
        node_id = _require_identifier(node.get("id"), "node.id")
        if node_id in nodes_by_id:
            raise ValueError(f"duplicate node id: {node_id}")
        node_type = _require_identifier(node.get("type"), f"{node_id}.type")
        if not isinstance(node.get("properties"), dict):
            raise ValueError(f"node properties must be a dictionary: {node_id}")
        nodes_by_id[node_id] = node
        nodes_by_type.setdefault(node_type, []).append(node)

    question_nodes = nodes_by_type.get("Question", [])
    if len(question_nodes) != 1 or question_nodes[0]["id"] != question_node_id(sample_id):
        raise ValueError("evaluation graph must contain its single Question node")

    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError("every edge must be a dictionary")
        source = _require_identifier(edge.get("source"), "edge.source")
        target = _require_identifier(edge.get("target"), "edge.target")
        _require_identifier(edge.get("type"), "edge.type")
        if source not in nodes_by_id or target not in nodes_by_id:
            raise ValueError(f"edge endpoint does not exist: {source} -> {target}")

    decision_nodes = nodes_by_type.get("Decision", [])
    if len(decision_nodes) != 1:
        raise ValueError("evaluation graph must contain exactly one Decision node")
    decision_node = decision_nodes[0]
    decision_type = _require_identifier(
        decision_node["properties"].get("decision_type"),
        "Decision.decision_type",
    )
    if decision_node["id"] != decision_node_id(sample_id, decision_type):
        raise ValueError("Decision node id mismatch")
    if not any(
        edge.get("source") == question_node_id(sample_id)
        and edge.get("target") == decision_node["id"]
        and edge.get("type") == "QUESTION_EXPECTS_DECISION"
        for edge in edges
    ):
        raise ValueError("Decision node is missing QUESTION_EXPECTS_DECISION edge")

    for claim_node in nodes_by_type.get("Claim", []):
        claim_role = _require_identifier(
            claim_node["properties"].get("claim_role"),
            "Claim.claim_role",
        )
        if claim_role not in GOLD_ONLY_CLAIM_ROLES:
            raise ValueError(f"unsupported gold claim role: {claim_role}")
        if not any(
            edge.get("source") == question_node_id(sample_id)
            and edge.get("target") == claim_node["id"]
            and edge.get("type") == "QUESTION_HAS_GOLD_CLAIM"
            for edge in edges
        ):
            raise ValueError(f"Claim node is missing gold edge: {claim_node['id']}")

    for risk_node in nodes_by_type.get("RiskLabel", []):
        risk_label = _require_identifier(
            risk_node["properties"].get("risk_label"),
            "RiskLabel.risk_label",
        )
        if risk_node["id"] != risk_node_id(risk_label):
            raise ValueError("RiskLabel node id mismatch")
        if not any(
            edge.get("source") == question_node_id(sample_id)
            and edge.get("target") == risk_node["id"]
            and edge.get("type") == "QUESTION_HAS_RISK"
            for edge in edges
        ):
            raise ValueError(f"RiskLabel node is missing gold edge: {risk_node['id']}")

    gold_evidence_edges = [
        edge for edge in edges if edge.get("type") == "QUESTION_HAS_GOLD_EVIDENCE"
    ]
    gold_evidence_status = graph.get("gold_evidence_status")
    if gold_evidence_status == "page_span_located":
        if not gold_evidence_edges:
            raise ValueError("page_span_located evaluation graph requires gold evidence")
        for edge in gold_evidence_edges:
            evidence_node = nodes_by_id[edge["target"]]
            if evidence_node.get("type") != "EvidenceSpan":
                raise ValueError("gold evidence edge must target an EvidenceSpan")
            properties = evidence_node["properties"]
            source_id = _require_identifier(properties.get("source_id"), "source_id")
            expected_id = evidence_node_id(
                source_id,
                properties.get("page_number"),
                properties.get("content"),
            )
            if evidence_node["id"] != expected_id:
                raise ValueError("gold EvidenceSpan node id mismatch")
            if not any(
                item.get("source") == evidence_node["id"]
                and item.get("target") == source_node_id(source_id)
                and item.get("type") == "EVIDENCE_FROM_SOURCE"
                for item in edges
            ):
                raise ValueError("gold EvidenceSpan is missing EVIDENCE_FROM_SOURCE edge")
    elif gold_evidence_status == "missing_source":
        if gold_evidence_edges:
            raise ValueError("missing_source evaluation graph cannot contain gold evidence")
    elif gold_evidence_status == "policy_rule":
        if gold_evidence_edges:
            raise ValueError("policy_rule evaluation graph cannot contain PDF gold evidence")
        policy_nodes = nodes_by_type.get("PolicyRule", [])
        if not policy_nodes:
            raise ValueError("policy_rule evaluation graph requires a PolicyRule node")
        for policy_node in policy_nodes:
            rule_id = _require_identifier(
                policy_node["properties"].get("rule_id"),
                "PolicyRule.rule_id",
            )
            if policy_node["id"] != policy_node_id(rule_id):
                raise ValueError("PolicyRule node id mismatch")
            _require_identifier(
                policy_node["properties"].get("rule_text"),
                "PolicyRule.rule_text",
            )
            if not any(
                item.get("source") == policy_node["id"]
                and item.get("target") == decision_node["id"]
                and item.get("type") == "POLICY_SUPPORTS_DECISION"
                for item in edges
            ):
                raise ValueError("PolicyRule is missing POLICY_SUPPORTS_DECISION edge")
    else:
        raise ValueError("unsupported gold_evidence_status")
