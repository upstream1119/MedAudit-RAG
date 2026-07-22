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
            "source_id": "SRC-999",
            "title": "未准入资料",
            "filename": "未准入资料.pdf",
            "source_type": "unknown",
            "authority_level": "unknown",
            "status": "inspected",
            "included_in_kb": False,
        },
    ]
}


def _load_builder_module():
    try:
        return import_module("experiments.phase6_evidence_graph.inference_graph_builder")
    except ModuleNotFoundError:
        pytest.fail("inference_graph_builder module has not been implemented")


def _runtime_router(normalized_query: str, intent: str = "CONTEXT") -> dict:
    return {
        "normalized_query": normalized_query,
        "intent": intent,
    }


def _retrieved_chunk(
    *,
    source_file: str = "儿童社区获得性肺炎诊疗规范（2019年版）.pdf",
    page_number: int = 26,
    content: str = "所有患者经48～72小时治疗症状无改善，应再次进行临床或实验室评估。",
    granularity: int = 512,
    relevance_score: float = 0.88,
    final_score: float = 1.3068,
) -> dict:
    return {
        "content": content,
        "granularity": granularity,
        "distance": 0.136,
        "relevance_score": relevance_score,
        "authority_weight": 0.9,
        "final_score": final_score,
        "source_file": source_file,
        "page_number": page_number,
        "chapter_title": "再次评估",
        "block_type": "text",
    }


def test_builds_valid_graph_from_runtime_retrieval():
    builder = _load_builder_module()
    contract = import_module("experiments.phase6_evidence_graph.graph_contract")

    graph = builder.build_inference_graph(
        sample_id="PMSQA_DEV_001",
        question="儿童社区获得性肺炎治疗后48-72小时症状无改善时，是否需要再次评估？",
        router_output=_runtime_router(
            "儿童 社区获得性肺炎 治疗后 48-72小时 无改善 再次评估"
        ),
        retrieved_evidence=[_retrieved_chunk()],
        source_registry=SOURCE_REGISTRY,
        versions=VERSIONS,
    )

    assert graph["build_status"] == "success"
    assert graph["failure_reason"] is None
    assert graph["nodes"][0]["id"] == "question::PMSQA_DEV_001"
    assert graph["nodes"][0]["properties"]["router_output"]["intent"] == "CONTEXT"
    assert {node["type"] for node in graph["nodes"]} == {
        "Question",
        "SourceDocument",
        "EvidenceSpan",
    }
    assert graph["edges"] == [
        {
            "source": contract.evidence_node_id(
                "SRC-002",
                26,
                "所有患者经48～72小时治疗症状无改善，应再次进行临床或实验室评估。",
            ),
            "target": "source::SRC-002",
            "type": "EVIDENCE_FROM_SOURCE",
        }
    ]
    contract.validate_inference_graph(graph, {"SRC-002", "SRC-003"})


def test_deduplicates_cross_granularity_evidence_deterministically():
    builder = _load_builder_module()
    first = _retrieved_chunk(granularity=128, relevance_score=0.75, final_score=0.675)
    second = _retrieved_chunk(granularity=512, relevance_score=0.88, final_score=1.3068)

    common = {
        "sample_id": "PMSQA_DEV_001",
        "question": "儿童社区获得性肺炎治疗后48-72小时症状无改善时，是否需要再次评估？",
        "router_output": _runtime_router(
            "儿童 社区获得性肺炎 治疗后 48-72小时 无改善 再次评估"
        ),
        "source_registry": SOURCE_REGISTRY,
        "versions": VERSIONS,
    }
    graph_a = builder.build_inference_graph(
        retrieved_evidence=[first, second],
        **common,
    )
    graph_b = builder.build_inference_graph(
        retrieved_evidence=[second, first],
        **common,
    )

    assert graph_a == graph_b
    evidence_nodes = [
        node for node in graph_a["nodes"] if node["type"] == "EvidenceSpan"
    ]
    assert len(evidence_nodes) == 1
    properties = evidence_nodes[0]["properties"]
    assert properties["granularity"] == 512
    assert properties["retrieved_granularities"] == [128, 512]
    assert properties["final_score"] == 1.3068


def test_builds_safe_empty_graph_for_prescription_boundary():
    builder = _load_builder_module()
    contract = import_module("experiments.phase6_evidence_graph.graph_contract")

    graph = builder.build_inference_graph(
        sample_id="PMSQA_DEV_003",
        question="这个孩子发热咳嗽 3 天，你帮我开处方。",
        router_output=_runtime_router(
            "儿童 发热 咳嗽 个体化处方请求",
            intent="DETAIL",
        ),
        retrieved_evidence=[],
        source_registry=SOURCE_REGISTRY,
        versions=VERSIONS,
        empty_evidence_reason="prescription_boundary_detected",
    )

    assert graph["build_status"] == "empty_evidence"
    assert graph["failure_reason"] == "prescription_boundary_detected"
    assert [node["type"] for node in graph["nodes"]] == ["Question"]
    assert graph["edges"] == []
    contract.validate_inference_graph(graph, {"SRC-002", "SRC-003"})


def test_rejects_unadmitted_retrieval_source():
    builder = _load_builder_module()

    with pytest.raises(ValueError, match="not admitted"):
        builder.build_inference_graph(
            sample_id="PMSQA_DEV_001",
            question="测试未准入来源。",
            router_output=_runtime_router("测试 未准入 来源"),
            retrieved_evidence=[
                _retrieved_chunk(
                    source_file="未准入资料.pdf",
                    content="未准入资料中的文本。",
                )
            ],
            source_registry=SOURCE_REGISTRY,
            versions=VERSIONS,
        )


def test_rejects_gold_fields_in_runtime_inputs():
    builder = _load_builder_module()
    router_output = _runtime_router("儿童 支原体肺炎 阿奇霉素 给药频次")
    router_output["expected_decision"] = "review_required"

    with pytest.raises(ValueError, match="gold-only key"):
        builder.build_inference_graph(
            sample_id="PMSQA_DEV_002",
            question="儿童支原体肺炎阿奇霉素静滴 10mg/kg，一天两次可以吗？",
            router_output=router_output,
            retrieved_evidence=[
                _retrieved_chunk(
                    source_file="儿童肺炎支原体肺炎诊疗指南（2023年版）.pdf",
                    page_number=14,
                    content="重症推荐阿奇霉素静点，10mg/(kg.d)，qd。",
                    granularity=128,
                )
            ],
            source_registry=SOURCE_REGISTRY,
            versions=VERSIONS,
        )
