import copy
import hashlib
import json
from importlib import import_module

import pytest


VERSIONS = {
    "schema_version": "phase6a-pedimedkg-schema-v0.1",
    "dataset_version": "dev50-v1.0",
    "kb_version": "KB-medium-v1",
}

SOURCE_REGISTRY = {
    "sources": [
        {
            "source_id": "SRC-002",
            "title": "儿童社区获得性肺炎诊疗规范（2019年版）",
            "filename": "儿童社区获得性肺炎诊疗规范（2019年版）.pdf",
            "source_type": "clinical_guideline",
            "authority_level": "national",
            "status": "indexed",
            "included_in_kb": True,
        },
        {
            "source_id": "SRC-003",
            "title": "儿童肺炎支原体肺炎诊疗指南（2023年版）",
            "filename": "儿童肺炎支原体肺炎诊疗指南（2023年版）.pdf",
            "source_type": "clinical_guideline",
            "authority_level": "national",
            "status": "indexed",
            "included_in_kb": True,
        },
        {
            "source_id": "SRC-004",
            "title": "国家基本药物目录（2018年版）",
            "filename": "国家基本药物目录（2018年版）.pdf",
            "source_type": "essential_medicines_list",
            "authority_level": "national",
            "status": "indexed",
            "included_in_kb": True,
        },
    ]
}

CONFIG = {
    "config_version": "phase6b-rerank-test-v0.1",
    "method_id": "runtime_constraint_graph_reranking",
    "method_version": "phase6b-runtime-constraint-v0.1",
    "top_k": 4,
    "score_weights": {
        "relevance": 0.65,
        "authority": 0.20,
        "constraint_type_coverage": 0.15,
    },
    "constraint_type_weights": {
        "dose": 1.0,
        "frequency": 1.0,
        "route": 1.0,
        "monitoring_window": 1.0,
        "monitoring_trigger": 1.0,
        "monitoring_action": 1.0,
    },
}


def _load_modules():
    try:
        reranker = import_module("experiments.phase6_evidence_graph.graph_reranker")
    except ModuleNotFoundError:
        pytest.fail("graph_reranker module has not been implemented")
    builder = import_module(
        "experiments.phase6_evidence_graph.inference_graph_builder"
    )
    return reranker, builder


def _retrieved_chunk(
    *,
    content: str,
    page_number: int,
    relevance_score: float,
    final_score: float,
    source_file: str = "儿童社区获得性肺炎诊疗规范（2019年版）.pdf",
) -> dict:
    return {
        "content": content,
        "granularity": 128,
        "distance": 1.0 - relevance_score,
        "relevance_score": relevance_score,
        "authority_weight": 0.9,
        "final_score": final_score,
        "source_file": source_file,
        "page_number": page_number,
        "chapter_title": "",
        "block_type": "text",
    }


def _build_graph(
    builder,
    *,
    sample_id: str,
    question: str,
    evidence: list[dict],
    empty_evidence_reason: str = "retrieval_returned_no_admitted_evidence",
) -> dict:
    return builder.build_inference_graph(
        sample_id=sample_id,
        question=question,
        router_output={"normalized_query": question, "intent": "CONTEXT"},
        retrieved_evidence=evidence,
        source_registry=SOURCE_REGISTRY,
        versions=VERSIONS,
        empty_evidence_reason=empty_evidence_reason,
    )


def _sha256(value: dict) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_reranks_constraint_covering_evidence_without_mutating_parent():
    reranker, builder = _load_modules()
    graph = _build_graph(
        builder,
        sample_id="PMSQA_DEV_001",
        question="儿童社区获得性肺炎治疗后48-72小时症状无改善时，是否需要再次评估？",
        evidence=[
            _retrieved_chunk(
                content="重症肺炎需要综合判断病情。",
                page_number=12,
                relevance_score=0.82,
                final_score=0.82,
            ),
            _retrieved_chunk(
                content="经48～72小时治疗症状无改善，应再次进行临床或实验室评估。",
                page_number=26,
                relevance_score=0.78,
                final_score=0.78,
            ),
        ],
    )
    graph_before = copy.deepcopy(graph)
    parent_sha = _sha256(graph)

    artifact = reranker.build_reranking_artifact(graph, CONFIG)

    assert graph == graph_before
    assert artifact["parent_inference_graph_sha256"] == parent_sha
    assert artifact["ranked_evidence"][0]["page_number"] == 26
    assert artifact["ranked_evidence"][0]["rank_before"] == 2
    assert artifact["ranked_evidence"][0]["rank_after"] == 1
    ranking_baseline = artifact["ranking_baseline"]
    assert artifact["artifact_schema_version"] == "phase6b-reranking-artifact-v0.2"
    assert ranking_baseline["vector_top_k_evidence_ids"][0] != (
        ranking_baseline["graph_top_k_evidence_ids"][0]
    )
    assert ranking_baseline["top1_changed"] is True
    assert len(ranking_baseline["moved_in_evidence_ids"]) == 0
    assert len(ranking_baseline["moved_out_evidence_ids"]) == 0
    assert artifact["external_model_calls"] == 0
    assert artifact["estimated_cost"] == 0


def test_reports_bid_to_qd_frequency_conflict():
    reranker, builder = _load_modules()
    graph = _build_graph(
        builder,
        sample_id="PMSQA_DEV_002",
        question="儿童支原体肺炎阿奇霉素静滴 10mg/kg，一天两次可以吗？",
        evidence=[
            _retrieved_chunk(
                source_file="儿童肺炎支原体肺炎诊疗指南（2023年版）.pdf",
                content="重症推荐阿奇霉素静点，10 mg/(kg.d)，qd。",
                page_number=14,
                relevance_score=0.75,
                final_score=0.75,
            )
        ],
    )

    artifact = reranker.build_reranking_artifact(graph, CONFIG)

    frequency_audits = [
        row
        for row in artifact["constraint_audit"]
        if row["constraint_type"] == "frequency"
    ]
    assert frequency_audits == [
        {
            "constraint_type": "frequency",
            "query_value": "bid",
            "evidence_values": ["qd"],
            "status": "conflict",
        }
    ]
    assert artifact["audit_summary"]["conflict"] == 1


def test_preserves_vector_order_without_runtime_query_constraints():
    reranker, builder = _load_modules()
    graph = _build_graph(
        builder,
        sample_id="PMSQA_DEV_007",
        question="儿童肺炎联合用药是否需要人工复核？",
        evidence=[
            _retrieved_chunk(
                content="第一条联合用药背景证据。",
                page_number=7,
                relevance_score=0.60,
                final_score=0.90,
            ),
            _retrieved_chunk(
                content="第二条联合用药背景证据。",
                page_number=18,
                relevance_score=0.90,
                final_score=0.80,
            ),
        ],
    )

    artifact = reranker.build_reranking_artifact(graph, CONFIG)

    assert artifact["query_constraints"] == []
    assert artifact["rerank_applied"] is False
    assert artifact["rerank_skip_reason"] == "no_runtime_query_constraints"
    assert artifact["ranking_baseline"]["top1_changed"] is False
    assert artifact["ranked_evidence"][0]["page_number"] == 7


def test_preserves_vector_order_without_candidate_constraint_coverage():
    reranker, builder = _load_modules()
    graph = _build_graph(
        builder,
        sample_id="PMSQA_DEV_042",
        question="儿童咳嗽有痰，能否同时口服止咳药和化痰药？",
        evidence=[
            _retrieved_chunk(
                content="第一条一般性儿童咳嗽证据。",
                page_number=12,
                relevance_score=0.60,
                final_score=0.90,
            ),
            _retrieved_chunk(
                content="第二条一般性儿童咳嗽证据。",
                page_number=13,
                relevance_score=0.90,
                final_score=0.80,
            ),
        ],
    )

    artifact = reranker.build_reranking_artifact(graph, CONFIG)

    assert artifact["query_constraints"]
    assert artifact["rerank_applied"] is False
    assert artifact["rerank_skip_reason"] == "no_runtime_constraint_coverage"
    assert artifact["ranking_baseline"]["top1_changed"] is False
    assert artifact["ranked_evidence"][0]["page_number"] == 12


def test_reranks_only_same_combination_family_evidence():
    reranker, builder = _load_modules()
    graph = _build_graph(
        builder,
        sample_id="PMSQA_DEV_042",
        question="儿童咳嗽有痰，能否同时口服止咳药和化痰药？",
        evidence=[
            _retrieved_chunk(
                content="抗菌药物联合用药通常采用两种药物联合。",
                page_number=7,
                relevance_score=0.90,
                final_score=0.90,
            ),
            _retrieved_chunk(
                content="止咳药和化痰药同时使用需要核对适应证和证据。",
                page_number=12,
                relevance_score=0.75,
                final_score=0.75,
            ),
        ],
    )

    artifact = reranker.build_reranking_artifact(graph, CONFIG)

    assert artifact["rerank_applied"] is True
    assert artifact["ranked_evidence"][0]["page_number"] == 12
    assert artifact["ranked_evidence"][0]["matched_constraint_types"] == [
        "combination_cough_medicine"
    ]
    assert artifact["ranking_baseline"]["top1_changed"] is True


def test_preserves_high_relevance_formulation_listing_and_records_scope_conflict():
    reranker, builder = _load_modules()
    graph = _build_graph(
        builder,
        sample_id="PMSQA_DEV_019",
        question="儿童肺炎可以把基本药物目录作为阿奇霉素剂量依据吗？",
        evidence=[
            _retrieved_chunk(
                source_file="国家基本药物目录（2018年版）.pdf",
                content="品种名称为阿奇霉素，剂型、规格包括片剂、胶囊。",
                page_number=20,
                relevance_score=0.84,
                final_score=0.95,
            ),
            _retrieved_chunk(
                source_file="儿童肺炎支原体肺炎诊疗指南（2023年版）.pdf",
                content="推荐剂量为10mg/(kg.d)，每日一次。",
                page_number=14,
                relevance_score=0.75,
                final_score=0.80,
            ),
        ],
    )

    artifact = reranker.build_reranking_artifact(graph, CONFIG)

    assert artifact["rerank_applied"] is True
    assert artifact["ranked_evidence"][0]["source_id"] == "SRC-004"
    assert artifact["ranked_evidence"][0]["page_number"] == 20
    assert artifact["ranking_baseline"]["top1_changed"] is False
    assert any(
        edge["type"] == "EVIDENCE_CONFLICTS_WITH_CONSTRAINT"
        and edge.get("properties", {}).get("evidence_value")
        == "formulation_listing"
        for edge in artifact["runtime_constraint_graph"]["edges"]
    )


def test_preserves_prescription_boundary_as_safe_empty_artifact():
    reranker, builder = _load_modules()
    graph = _build_graph(
        builder,
        sample_id="PMSQA_DEV_003",
        question="这个孩子发热咳嗽 3 天，你帮我开处方。",
        evidence=[],
        empty_evidence_reason="prescription_boundary_detected",
    )

    artifact = reranker.build_reranking_artifact(graph, CONFIG)

    assert artifact["artifact_status"] == "boundary_refusal"
    assert artifact["ranked_evidence"] == []
    assert artifact["constraint_audit"] == []
    assert artifact["runtime_constraint_graph"]["nodes"] == []
    assert artifact["runtime_constraint_graph"]["edges"] == []


def test_rejects_evaluation_graph_and_gold_content():
    reranker, builder = _load_modules()
    graph = _build_graph(
        builder,
        sample_id="PMSQA_DEV_001",
        question="儿童肺炎治疗后是否需要再次评估？",
        evidence=[],
    )
    evaluation_graph = copy.deepcopy(graph)
    evaluation_graph["graph_type"] = "evaluation_graph"
    evaluation_graph["expected_decision"] = "answer"

    with pytest.raises(ValueError, match="inference_graph"):
        reranker.build_reranking_artifact(evaluation_graph, CONFIG)


def test_is_deterministic_and_respects_evidence_budget():
    reranker, builder = _load_modules()
    evidence = [
        _retrieved_chunk(
            content=f"第{index}条一般性肺炎证据。",
            page_number=index,
            relevance_score=0.90 - index / 100,
            final_score=0.90 - index / 100,
        )
        for index in range(1, 7)
    ]
    graph = _build_graph(
        builder,
        sample_id="PMSQA_DEV_001",
        question="儿童肺炎治疗后48-72小时症状无改善时是否需要再次评估？",
        evidence=evidence,
    )

    first = reranker.build_reranking_artifact(graph, CONFIG)
    second = reranker.build_reranking_artifact(graph, CONFIG)

    assert first == second
    assert len(first["ranked_evidence"]) == CONFIG["top_k"]
    assert first["artifact_sha256"] == second["artifact_sha256"]
