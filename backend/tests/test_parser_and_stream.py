import sys
from pathlib import Path
from types import SimpleNamespace

from app.agents.nodes import auditor as auditor_module
from app.agents.nodes.auditor import FaithfulnessScore
from app.agents.nodes import router as router_module
from app.agents.graph import route_after_router
from app.agents.nodes import retriever_node as retriever_node_module
from app.api import routes
from app.config import Settings
from app.knowledge import indexer as indexer_module
from app.knowledge import parser as parser_module
from app.knowledge import retriever as knowledge_retriever_module
from app.knowledge.indexer import ZhipuEmbeddingFunction
from app.knowledge.parser import DualTrackMedicalParser
from app.knowledge.retriever import MultiGranularityRetriever, RetrievedChunk
from app.models.schemas import IntentType, TrustLevel
from rebuild_index import _build_index_status


def test_track_a_text_uses_page_number_metadata(monkeypatch):
    def fake_to_markdown(*args, **kwargs):
        return [
            {
                "metadata": {"page_number": 4},
                "text": "CAP：年龄3个月以上：10mg/kg 静脉滴注，qd，至少2天。",
            }
        ]

    monkeypatch.setattr(parser_module.pymupdf4llm, "to_markdown", fake_to_markdown)

    blocks = DualTrackMedicalParser(min_text_length=1)._track_a_text(
        Path("dummy.pdf"),
        "sha256",
    )

    assert len(blocks) == 1
    assert blocks[0].metadata.page_number == 4


def test_track_a_text_skips_reference_pages(monkeypatch):
    def fake_to_markdown(*args, **kwargs):
        return [
            {
                "metadata": {"page_number": 12},
                "text": "## ［参 考 文 献］\n［40］ Recommendations on off-label use of intravenous azithromycin in children［J］.",
            },
            {
                "metadata": {"page_number": 4},
                "text": "标准与讨论\n**==> picture [31 x 14] intentionally omitted <==**\nCAP：年龄3个月以上：10mg/kg 静脉滴注，qd，至少2天。",
            },
        ]

    monkeypatch.setattr(parser_module.pymupdf4llm, "to_markdown", fake_to_markdown)

    blocks = DualTrackMedicalParser(min_text_length=1)._track_a_text(
        Path("dummy.pdf"),
        "sha256",
    )

    assert len(blocks) == 1
    assert blocks[0].metadata.page_number == 4
    assert "参考文献" not in blocks[0].content
    assert "picture" not in blocks[0].content


def test_inspect_source_marks_scan_heavy_pdf(monkeypatch, tmp_path):
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    class FakePage:
        def get_text(self):
            return ""

        def get_images(self, full=True):
            return [object()]

    class FakeDoc:
        page_count = 3

        def load_page(self, index):
            return FakePage()

        def close(self):
            pass

    monkeypatch.setattr(parser_module.fitz, "open", lambda _: FakeDoc())

    inspection = DualTrackMedicalParser().inspect_source(pdf_path, sample_pages=3)

    assert inspection.page_count == 3
    assert inspection.sampled_pages == 3
    assert inspection.text_pages == 0
    assert inspection.image_pages == 3
    assert inspection.scan_heavy is True


def test_serialize_evidence_chunks_uses_page_number():
    chunks = [
        SimpleNamespace(
            content="CAP：年龄3个月以上：10mg/kg 静脉滴注。",
            source_file="中国儿科超药品说明书用药专家共识.pdf",
            page_number=4,
        )
    ]

    payload = routes._serialize_evidence_chunks(chunks)

    assert payload == [
        {
            "content": "CAP：年龄3个月以上：10mg/kg 静脉滴注。",
            "source": "中国儿科超药品说明书用药专家共识.pdf",
            "page": 4,
        }
    ]


def test_resolve_answer_from_state_prefers_draft_answer():
    state = {"draft_answer": "依据不足，请人工复核。", "answer": "旧字段"}
    assert routes._resolve_answer_from_state(state) == "依据不足，请人工复核。"


def test_resolve_answer_from_state_falls_back_to_answer():
    state = {"answer": "旧字段回答"}
    assert routes._resolve_answer_from_state(state) == "旧字段回答"


def test_direct_prescription_request_is_blocked():
    assert routes._is_direct_prescription_request("这个孩子发热咳嗽 3 天，你帮我开处方。")
    assert routes._is_direct_prescription_request("患儿咳嗽发热，帮我开药。")
    assert not routes._is_direct_prescription_request("儿童重症肺炎支原体肺炎，是否可以静脉滴注阿奇霉素？")


def test_blocked_prescription_uses_rejected_trust_score():
    score = routes._rejected_trust_score()

    assert score.trust_level == TrustLevel.REJECTED
    assert score.trust_score == 0.0


def test_retriever_filters_reference_and_picture_noise():
    retriever = MultiGranularityRetriever.__new__(MultiGranularityRetriever)
    retriever._settings = SimpleNamespace(
        AUTHORITY_WEIGHTS={"expert_consensus": 0.7, "default": 0.5}
    )

    response = {
        "documents": [[
            "## ［参 考 文 献］\n［40］ Recommendations on off-label use of intravenous azithromycin in children［J］.",
            "标准与讨论 **==> picture [31 x 14] intentionally omitted <==** CAP：年龄3个月以上：10mg/kg 静脉滴注。",
            "CAP：年龄3个月以上：10mg/kg 静脉滴注，qd，至少2天，然后5mg/kg口服。",
        ]],
        "metadatas": [[
            {"source_file": "中国儿科超药品说明书用药专家共识.pdf", "page_number": 12, "block_type": "text"},
            {"source_file": "中国儿科超药品说明书用药专家共识.pdf", "page_number": 4, "block_type": "text"},
            {"source_file": "中国儿科超药品说明书用药专家共识.pdf", "page_number": 4, "block_type": "text"},
        ]],
        "distances": [[0.1, 0.2, 0.3]],
    }

    chunks = retriever._parse_chroma_response(response, 128)

    assert len(chunks) == 1
    assert chunks[0].page_number == 4
    assert "CAP：年龄3个月以上" in chunks[0].content


def test_retriever_filters_title_only_guideline_chunks():
    retriever = MultiGranularityRetriever.__new__(MultiGranularityRetriever)
    retriever._settings = SimpleNamespace(
        AUTHORITY_WEIGHTS={"clinical_guideline": 0.9, "default": 0.5}
    )

    response = {
        "documents": [[
            "儿童社区获得性肺炎诊疗规范（2019 年版）",
            "## 附录 儿童社区获得性肺炎诊疗规范（2019 年版）",
            "1.初次评估。重症患者初始治疗后1～2小时应作病情和疗效评估。",
        ]],
        "metadatas": [[
            {"source_file": "儿童社区获得性肺炎诊疗规范（2019年版）.pdf", "page_number": 27, "block_type": "text"},
            {"source_file": "儿童社区获得性肺炎诊疗规范（2019年版）.pdf", "page_number": 27, "block_type": "text"},
            {"source_file": "儿童社区获得性肺炎诊疗规范（2019年版）.pdf", "page_number": 26, "block_type": "text"},
        ]],
        "distances": [[0.1, 0.2, 0.3]],
    }

    chunks = retriever._parse_chroma_response(response, 512)

    assert len(chunks) == 1
    assert "初次评估" in chunks[0].content


def test_retriever_deduplicates_multi_granularity_results():
    chunks = [
        RetrievedChunk(
            "初次评估。重症患者初始治疗后1～2小时应作病情和疗效评估。",
            128, 0.1, 0.9, 0.9, 0.81,
            "儿童社区获得性肺炎诊疗规范（2019年版）.pdf", 26, "", "text",
        ),
        RetrievedChunk(
            "初次评估。重症患者初始治疗后1～2小时应作病情和疗效评估。",
            512, 0.2, 0.8, 0.9, 0.72,
            "儿童社区获得性肺炎诊疗规范（2019年版）.pdf", 26, "", "text",
        ),
        RetrievedChunk(
            "再次评估。48～72小时症状无改善时再次评估。",
            512, 0.3, 0.7, 0.9, 0.63,
            "儿童社区获得性肺炎诊疗规范（2019年版）.pdf", 26, "", "text",
        ),
    ]

    deduped = MultiGranularityRetriever._deduplicate_results(chunks)

    assert deduped == [chunks[0], chunks[2]]


def test_retriever_maps_new_guideline_sources_to_authority_weights():
    retriever = MultiGranularityRetriever.__new__(MultiGranularityRetriever)
    retriever._settings = SimpleNamespace(
        AUTHORITY_WEIGHTS={
            "national_pharmacopoeia": 1.0,
            "clinical_guideline": 0.9,
            "expert_consensus": 0.7,
            "default": 0.5,
        }
    )

    assert retriever._get_authority_weight("儿童肺炎支原体肺炎诊疗指南（2023年版）.pdf") == 0.9
    assert retriever._get_authority_weight("儿童社区获得性肺炎诊疗规范（2019年版）.pdf") == 0.9
    assert retriever._get_authority_weight("NICE_NG143_Fever_in_under_5s.pdf") == 0.9
    assert retriever._get_authority_weight("抗菌药物临床应用指导原则（2015年版）.pdf") == 0.9
    assert retriever._get_authority_weight("国家基本药物目录（2018年版）.pdf") == 1.0


def test_embedding_function_uses_dashscope_safe_batch_size():
    embed_fn = ZhipuEmbeddingFunction.__new__(ZhipuEmbeddingFunction)
    embed_fn._provider = "dashscope"

    assert embed_fn._batch_size() == 10


def test_context_intent_uses_multi_granularity_retrieval(monkeypatch):
    calls = []

    class FakeRetriever:
        def retrieve(self, query, granularity=None):
            calls.append({"query": query, "granularity": granularity})
            return []

    monkeypatch.setattr(retriever_node_module, "_retriever", FakeRetriever())

    state = {
        "original_query": "儿童重症肺炎支原体肺炎，是否可以静脉滴注阿奇霉素？",
        "normalized_query": "儿童 重症 肺炎支原体肺炎 静脉注射 阿奇霉素",
        "intent": IntentType.CONTEXT,
    }

    retriever_node_module.retriever_node(state)

    assert calls == [
        {
            "query": "儿童 重症 肺炎支原体肺炎 静脉注射 阿奇霉素",
            "granularity": None,
        }
    ]


def test_retriever_returns_empty_when_index_status_is_not_ready():
    retriever = MultiGranularityRetriever.__new__(MultiGranularityRetriever)
    retriever._index_status = {
        "ready": False,
        "missing_sources": ["临床诊疗指南：小儿内科分册.pdf"],
    }

    assert retriever.retrieve("儿童阿奇霉素静脉滴注") == []


def test_retriever_filters_results_when_required_drug_term_is_absent():
    chunks = [
        SimpleNamespace(content="儿童支气管肺炎可根据需要进行退热、祛痰等对症治疗。"),
        SimpleNamespace(content="不推荐常规使用糖皮质激素。"),
    ]

    filtered = MultiGranularityRetriever._apply_required_term_filter(
        "儿童支气管肺炎 氨溴索 超说明书 静脉给药",
        chunks,
    )

    assert filtered == []


def test_retriever_keeps_results_when_required_drug_alias_is_present():
    chunks = [
        SimpleNamespace(content="氨溴索可静脉给药。"),
        SimpleNamespace(content="不相关片段。"),
    ]

    filtered = MultiGranularityRetriever._apply_required_term_filter(
        "儿童支气管肺炎 沐舒坦 静脉给药",
        chunks,
    )

    assert filtered == chunks[:1]


def test_retriever_requires_route_term_when_query_names_intravenous_use():
    chunks = [
        SimpleNamespace(content="阿奇霉素可用于肺炎支原体肺炎治疗。"),
        SimpleNamespace(content="重症推荐阿奇霉素静点，10mg/(kg.d)，qd。"),
    ]

    filtered = MultiGranularityRetriever._apply_required_term_filter(
        "儿童重症肺炎支原体肺炎 静脉滴注 阿奇霉素",
        chunks,
    )

    assert filtered == chunks[1:]


def test_required_term_filter_relaxes_for_combination_review_query():
    chunks = [
        SimpleNamespace(content="联合用药通常采用2种药物联合，3种及3种以上药物联合仅适用于个别情况。"),
        SimpleNamespace(content="联合用药后药物不良反应亦可能增多。"),
    ]

    filtered = MultiGranularityRetriever._apply_required_term_filter(
        "儿童肺炎能否同时使用阿奇霉素、头孢、激素和雾化？",
        chunks,
    )

    assert filtered == chunks


def test_retriever_expands_antipyretic_query_with_english_aliases():
    expanded = MultiGranularityRetriever._expand_query(
        "儿童发热可以同时吃布洛芬和对乙酰氨基酚吗？"
    )

    assert "ibuprofen" in expanded
    assert "paracetamol" in expanded
    assert "simultaneously" in expanded


def test_retriever_downranks_catalog_for_treatment_safety_query():
    chunk = RetrievedChunk(
        content="布洛芬列入基本药物目录。",
        granularity=512,
        distance=0.1,
        relevance_score=0.9,
        authority_weight=1.0,
        final_score=0.9,
        source_file="国家基本药物目录（2018年版）.pdf",
        page_number=102,
        chapter_title="",
        block_type="text",
    )

    score = MultiGranularityRetriever._adjust_score_for_query(
        "儿童发热可以同时吃布洛芬和对乙酰氨基酚吗？",
        chunk,
        0.9,
    )

    assert score < 0.9


def test_retriever_boosts_fever_guideline_for_antipyretic_query():
    chunk = RetrievedChunk(
        content="When using paracetamol or ibuprofen in children with fever: do not give both agents simultaneously.",
        granularity=512,
        distance=0.1,
        relevance_score=0.9,
        authority_weight=0.9,
        final_score=0.81,
        source_file="NICE_NG143_Fever_in_under_5s.pdf",
        page_number=28,
        chapter_title="",
        block_type="text",
    )

    score = MultiGranularityRetriever._adjust_score_for_query(
        "儿童发热可以同时吃布洛芬和对乙酰氨基酚吗？",
        chunk,
        0.81,
    )

    assert score > 0.81


def test_retriever_collects_source_specific_lexical_fallback(monkeypatch):
    class FakeCollection:
        def get(self, where=None, include=None):
            assert where == {"source_file": {"$eq": "NICE_NG143_Fever_in_under_5s.pdf"}}
            assert include == ["documents", "metadatas"]
            return {
                "documents": [
                    "When using paracetamol or ibuprofen in children with fever, do not give both agents simultaneously.",
                    "This unrelated paragraph describes general assessment.",
                ],
                "metadatas": [
                    {
                        "source_file": "NICE_NG143_Fever_in_under_5s.pdf",
                        "page_number": 28,
                        "chapter_title": "Antipyretic interventions",
                        "block_type": "text",
                    },
                    {
                        "source_file": "NICE_NG143_Fever_in_under_5s.pdf",
                        "page_number": 2,
                        "chapter_title": "Introduction",
                        "block_type": "text",
                    },
                ],
            }

    class FakeChroma:
        def get_collection(self, name):
            return FakeCollection()

    retriever = MultiGranularityRetriever.__new__(MultiGranularityRetriever)
    retriever._settings = SimpleNamespace(
        AUTHORITY_WEIGHTS={"clinical_guideline": 0.9, "default": 0.5},
        RETRIEVAL_TOP_K=3,
    )
    retriever._chroma = FakeChroma()

    chunks = retriever._collect_lexical_fallback_candidates(
        "儿童发热可以同时吃布洛芬和对乙酰氨基酚吗？",
        [512],
    )

    assert len(chunks) == 1
    assert chunks[0].source_file == "NICE_NG143_Fever_in_under_5s.pdf"
    assert chunks[0].page_number == 28
    assert chunks[0].relevance_score > 0.5


def test_required_term_filter_keeps_principle_chunks_for_broad_review_query():
    drug_specific = SimpleNamespace(content="阿奇霉素可用于肺炎支原体肺炎治疗。")
    principle = SimpleNamespace(
        content="联合用药通常采用2种药物联合，3种及3种以上药物联合仅适用于个别情况；联合用药后药物不良反应亦可能增多。"
    )

    filtered = MultiGranularityRetriever._apply_required_term_filter(
        "儿童肺炎能否同时使用阿奇霉素、头孢、激素和雾化？",
        [drug_specific, principle],
    )

    assert drug_specific in filtered
    assert principle in filtered


def test_retriever_collects_cap_nonresponse_fallback_for_48_hour_query():
    class FakeCollection:
        def get(self, where=None, include=None):
            assert where == {"source_file": {"$eq": "儿童社区获得性肺炎诊疗规范（2019年版）.pdf"}}
            return {
                "documents": [
                    "所有患者经48～72 小时治疗症状无改善，应再次进行临床或/和实验室评估，并考虑抗生素覆盖、剂量、耐药等问题。",
                    "肺炎支原体肺炎可选用大环内酯类抗菌药物。",
                ],
                "metadatas": [
                    {
                        "source_file": "儿童社区获得性肺炎诊疗规范（2019年版）.pdf",
                        "page_number": 26,
                        "chapter_title": "",
                        "block_type": "text",
                    },
                    {
                        "source_file": "儿童社区获得性肺炎诊疗规范（2019年版）.pdf",
                        "page_number": 15,
                        "chapter_title": "",
                        "block_type": "text",
                    },
                ],
            }

    class FakeChroma:
        def get_collection(self, name):
            return FakeCollection()

    retriever = MultiGranularityRetriever.__new__(MultiGranularityRetriever)
    retriever._settings = SimpleNamespace(
        AUTHORITY_WEIGHTS={"clinical_guideline": 0.9, "default": 0.5},
        RETRIEVAL_TOP_K=3,
    )
    retriever._chroma = FakeChroma()

    chunks = retriever._collect_lexical_fallback_candidates(
        "儿童肺炎使用抗生素 48 小时没有好转，是不是一定要换药？",
        [512],
    )

    assert len(chunks) == 1
    assert chunks[0].page_number == 26


def test_retriever_collects_bronchiolitis_fallback_for_wheeze_nebulization_query():
    class FakeCollection:
        def get(self, where=None, include=None):
            assert where == {"source_file": {"$eq": "NICE_NG9_Bronchiolitis_in_children.pdf"}}
            return {
                "documents": [
                    "Do not use antibiotics, salbutamol, ipratropium bromide, systemic or inhaled corticosteroids for bronchiolitis.",
                ],
                "metadatas": [
                    {
                        "source_file": "NICE_NG9_Bronchiolitis_in_children.pdf",
                        "page_number": 11,
                        "chapter_title": "",
                        "block_type": "text",
                    },
                ],
            }

    class FakeChroma:
        def get_collection(self, name):
            return FakeCollection()

    retriever = MultiGranularityRetriever.__new__(MultiGranularityRetriever)
    retriever._settings = SimpleNamespace(
        AUTHORITY_WEIGHTS={"clinical_guideline": 0.9, "default": 0.5},
        RETRIEVAL_TOP_K=3,
    )
    retriever._chroma = FakeChroma()

    chunks = retriever._collect_lexical_fallback_candidates(
        "儿童肺炎合并喘息时，雾化药物能否和抗生素同时用？",
        [512],
    )

    assert len(chunks) == 1
    assert chunks[0].source_file == "NICE_NG9_Bronchiolitis_in_children.pdf"


def test_retriever_boosts_specific_combination_principle_over_general_adverse_effects():
    principle = RetrievedChunk(
        content="联合用药通常采用2种药物联合，3种及3种以上药物联合仅适用于个别情况；联合用药后药物不良反应亦可能增多。",
        granularity=512,
        distance=0.4,
        relevance_score=0.7,
        authority_weight=0.9,
        final_score=0.63,
        source_file="抗菌药物临床应用指导原则（2015年版）.pdf",
        page_number=7,
        chapter_title="",
        block_type="text",
    )
    general_adverse_effect = RetrievedChunk(
        content="本类药物可能出现凝血功能障碍和不良反应，治疗期间应注意观察。",
        granularity=512,
        distance=0.4,
        relevance_score=0.7,
        authority_weight=0.9,
        final_score=0.63,
        source_file="抗菌药物临床应用指导原则（2015年版）.pdf",
        page_number=31,
        chapter_title="",
        block_type="text",
    )

    query = "儿童肺炎能否同时使用阿奇霉素、头孢、激素和雾化？"

    principle_score = MultiGranularityRetriever._adjust_score_for_query(query, principle, 0.63)
    general_score = MultiGranularityRetriever._adjust_score_for_query(query, general_adverse_effect, 0.63)

    assert principle_score > general_score


def test_lexical_fallback_prefers_combination_principle_page():
    class FakeCollection:
        def get(self, where=None, include=None):
            assert where == {"source_file": {"$eq": "抗菌药物临床应用指导原则（2015年版）.pdf"}}
            return {
                "documents": [
                    "抗菌药物联合应用时应注意2种、3种药物相关不良反应和单一药物治疗情况。",
                    "联合用药通常采用2种药物联合，3种及3种以上药物联合仅适用于个别情况；联合用药后药物不良反应亦可能增多。",
                ],
                "metadatas": [
                    {
                        "source_file": "抗菌药物临床应用指导原则（2015年版）.pdf",
                        "page_number": 31,
                        "chapter_title": "",
                        "block_type": "text",
                    },
                    {
                        "source_file": "抗菌药物临床应用指导原则（2015年版）.pdf",
                        "page_number": 7,
                        "chapter_title": "",
                        "block_type": "text",
                    },
                ],
            }

    class FakeChroma:
        def get_collection(self, name):
            return FakeCollection()

    retriever = MultiGranularityRetriever.__new__(MultiGranularityRetriever)
    retriever._settings = SimpleNamespace(
        AUTHORITY_WEIGHTS={"clinical_guideline": 0.9, "default": 0.5},
        RETRIEVAL_TOP_K=3,
    )
    retriever._chroma = FakeChroma()

    chunks = retriever._collect_lexical_fallback_candidates(
        "儿童肺炎能否同时使用阿奇霉素、头孢、激素和雾化？",
        [512],
    )

    assert chunks[0].page_number == 7


def test_faithfulness_score_accepts_reasoning_alias():
    score = FaithfulnessScore.model_validate(
        {
            "score": 8,
            "reasoning": "The answer is supported by the evidence.",
        }
    )

    assert score.reason == "The answer is supported by the evidence."


def test_router_failure_returns_rejected_state(monkeypatch):
    monkeypatch.setattr(
        router_module,
        "generate_structured_output",
        lambda **_: (_ for _ in ()).throw(ValueError("invalid router payload")),
    )

    state = {"original_query": "儿童阿奇霉素静滴剂量是否合规？"}

    result = router_module.router_node(state)

    assert result["current_node"] == "router"
    assert result["intent"] == IntentType.DETAIL
    assert result["evidence"] == []
    assert result["trust_score"].trust_level == TrustLevel.REJECTED
    assert result["draft_answer"]
    assert "Router 解析失败" in result["error_message"]


def test_graph_routes_router_failure_to_end():
    state = {
        "original_query": "儿童阿奇霉素静滴剂量是否合规？",
        "error_message": "Router 解析失败: invalid router payload",
    }

    assert route_after_router(state) == "end"


def test_auditor_demotes_twice_daily_when_evidence_recommends_qd(monkeypatch):
    monkeypatch.setattr(
        auditor_module,
        "generate_structured_output",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("frequency conflict should skip Judge LLM")
        ),
    )

    state = {
        "original_query": "儿童支原体肺炎阿奇霉素静滴 10mg/kg，一天两次可以吗？",
        "normalized_query": "儿童 支原体肺炎 阿奇霉素 静脉滴注 10mg/kg 每日两次",
        "draft_answer": "指南推荐阿奇霉素静滴 10mg/kg。",
        "evidence": [
            SimpleNamespace(
                content="重症推荐阿奇霉素静点，10mg/(kg.d)，qd，连用7d左右。",
                relevance_score=0.68,
                authority_weight=0.9,
            )
        ],
    }

    result = auditor_module.auditor_node(state)

    assert result["trust_score"].trust_level == TrustLevel.REJECTED
    assert result["trust_score"].s_faith == 2.0
    assert "频次" in result["draft_answer"]


def test_auditor_frequency_conflict_uses_clean_rejection_message(monkeypatch):
    monkeypatch.setattr(
        auditor_module,
        "generate_structured_output",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("frequency conflict should skip Judge LLM")
        ),
    )

    state = {
        "original_query": "儿童支原体肺炎阿奇霉素静滴 10mg/kg，一天两次可以吗？",
        "normalized_query": "儿童 支原体肺炎 阿奇霉素 静脉滴注 10mg/kg 给药频率 一天两次",
        "draft_answer": '{"normalized_query": "儿童 支原体肺炎 阿奇霉素 静脉滴注", "intent": "DETAIL"}',
        "evidence": [
            SimpleNamespace(
                content="重症推荐阿奇霉素静点，10mg/(kg.d)，qd，连用7d左右。",
                relevance_score=0.69,
                authority_weight=0.9,
            )
        ],
    }

    result = auditor_module.auditor_node(state)

    assert result["trust_score"].trust_level == TrustLevel.REJECTED
    assert "存在冲突" in result["draft_answer"]
    assert "人工复核" in result["draft_answer"]
    assert "normalized_query" not in result["draft_answer"]
    assert "intent" not in result["draft_answer"]


def test_auditor_missing_draft_answer_returns_safe_rejection_message():
    state = {
        "original_query": "儿童支原体肺炎阿奇霉素静滴 10mg/kg，一天两次可以吗？",
        "normalized_query": "儿童 支原体肺炎 阿奇霉素 静脉滴注 10mg/kg 给药频率 一天两次",
        "evidence": [
            SimpleNamespace(
                content="重症推荐阿奇霉素静点，10mg/(kg.d)，qd，连用7d左右。",
                relevance_score=0.69,
                authority_weight=0.9,
            )
        ],
    }

    result = auditor_module.auditor_node(state)

    assert result["trust_score"].trust_level == TrustLevel.REJECTED
    assert "系统不应强行给出治疗方案" in result["draft_answer"]
    assert "人工复核" in result["draft_answer"]


def test_retriever_treats_missing_index_status_as_not_ready(tmp_path):
    retriever = MultiGranularityRetriever.__new__(MultiGranularityRetriever)
    retriever._persist_dir = str(tmp_path)

    status = retriever._load_index_status()

    assert status["ready"] is False
    assert "index_status.json" in status["reason"]


def test_index_status_treats_scan_heavy_sources_as_incomplete():
    pdfs = [Path("text.pdf"), Path("scan.pdf")]
    per_doc_summary = {
        "text.pdf": {"blocks_total": 10},
        "scan.pdf": {"blocks_total": 1},
    }
    source_inspections = {
        "text.pdf": {"scan_heavy": False},
        "scan.pdf": {"scan_heavy": True},
    }

    status = _build_index_status(pdfs, per_doc_summary, source_inspections)

    assert status["ready"] is False
    assert status["indexed_sources"] == ["text.pdf", "scan.pdf"]
    assert status["incomplete_sources"] == ["scan.pdf"]


def test_settings_accepts_release_debug_values():
    settings = Settings.model_validate(
        {
            "DEBUG": "release",
            "LLM_PROVIDER": "zhipu",
            "EMBEDDING_PROVIDER": "zhipu",
        }
    )
    assert settings.DEBUG is False


def test_settings_accepts_local_embedding_provider():
    settings = Settings.model_validate(
        {
            "DEBUG": "release",
            "LLM_PROVIDER": "zhipu",
            "EMBEDDING_PROVIDER": "local",
            "EMBEDDING_MODEL": "BAAI/bge-small-zh-v1.5",
        }
    )

    assert settings.EMBEDDING_PROVIDER == "local"
    assert settings.EMBEDDING_MODEL == "BAAI/bge-small-zh-v1.5"


def test_settings_env_file_points_to_backend_env():
    env_file = Path(Settings.model_config["env_file"])
    assert env_file.name == ".env"
    assert env_file.parent.name == "backend"


def test_settings_default_chroma_dir_points_to_backend_data():
    settings = Settings.model_validate(
        {
            "DEBUG": "release",
            "LLM_PROVIDER": "zhipu",
            "EMBEDDING_PROVIDER": "zhipu",
        }
    )
    chroma_dir = Path(settings.CHROMA_PERSIST_DIR)

    assert chroma_dir.parts[-3:] == ("backend", "data", "chroma_db")


def test_create_embedding_function_uses_local_sentence_transformer(monkeypatch):
    class FakeSentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name

        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            assert texts == ["儿童用药"]
            assert normalize_embeddings is True
            assert show_progress_bar is False
            return [[0.1, 0.2, 0.3]]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    monkeypatch.setattr(
        indexer_module,
        "get_settings",
        lambda: SimpleNamespace(
            EMBEDDING_PROVIDER="local",
            EMBEDDING_MODEL="fake-local-model",
        ),
    )

    embed_fn = indexer_module.create_embedding_function()

    assert embed_fn(["儿童用药"]) == [[0.1, 0.2, 0.3]]


def test_index_status_records_embedding_metadata():
    status = _build_index_status(
        pdfs=[Path("source.pdf")],
        per_doc_summary={"source.pdf": {"blocks_total": 1}},
        source_inspections={"source.pdf": {"scan_heavy": False}},
        embedding_provider="local",
        embedding_model="BAAI/bge-small-zh-v1.5",
    )

    assert status["ready"] is True
    assert status["embedding_provider"] == "local"
    assert status["embedding_model"] == "BAAI/bge-small-zh-v1.5"


def test_retriever_rejects_embedding_config_mismatch(monkeypatch, tmp_path):
    status_path = tmp_path / "index_status.json"
    status_path.write_text(
        (
            '{"ready": true, "embedding_provider": "dashscope", '
            '"embedding_model": "text-embedding-v4"}'
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        knowledge_retriever_module,
        "get_settings",
        lambda: SimpleNamespace(
            CHROMA_PERSIST_DIR=str(tmp_path),
            RETRIEVAL_TOP_K=3,
            EMBEDDING_PROVIDER="local",
            EMBEDDING_MODEL="BAAI/bge-small-zh-v1.5",
        ),
    )

    retriever = knowledge_retriever_module.MultiGranularityRetriever(
        persist_dir=str(tmp_path)
    )

    assert retriever.retrieve("儿童用药") == []
    assert retriever._index_status["ready"] is False
    assert "向量空间错配" in retriever._index_status["reason"]
