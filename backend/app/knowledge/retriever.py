"""
Medaudit-RAG 多粒度检索服务层 (Multi-Granularity Retriever)
=============================================================
职责:
  1. 封装三粒度 ChromaDB 查询, 暴露统一的 retrieve() 接口
  2. 对检索结果进行权威度加权 (W_authority), 优先返回权威来源切片
  3. 支持单粒度查询和三粒度融合查询两种模式

权威度权重规则 (来自 config.AUTHORITY_WEIGHTS):
  国家药典 (1.0) > 临床诊疗指南 (0.9) > 专家共识 (0.7)
    > 教材 (0.6) > 未标注 (0.5) > 个案报道 (0.3)

文件名 → 权威等级映射 (基于我们已知的三本书目):
  《国家基本药物处方集》  → national_pharmacopoeia (1.0)
  《临床诊疗指南·小儿内科分册》 → clinical_guideline (0.9)
  《儿科超说明书用药专家共识》  → expert_consensus (0.7)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import chromadb

from app.config import get_settings
from app.knowledge.indexer import create_embedding_function, _COLLECTION_NAMES

logger = logging.getLogger(__name__)

_REFERENCE_HEADING_RE = re.compile(r"^\s*##\s*[［\[]?\s*参\s*考\s*文\s*献", re.MULTILINE)
_PICTURE_PLACEHOLDER_RE = re.compile(
    r"\*\*==>\s*picture\s*\[[^\]]+\]\s*intentionally omitted\s*<==\*\*",
    re.IGNORECASE,
)
_PICTURE_TEXT_MARKER_RE = re.compile(
    r"\*\*-----\s*(Start|End)\s+of picture text\s*-----\*\*",
    re.IGNORECASE,
)
_TITLE_NORMALIZE_RE = re.compile(r"[\s#　，。、《》（）()：:;；·.\-]+")
_TITLE_ONLY_TERMS = (
    "儿童社区获得性肺炎诊疗规范2019年版",
    "儿童肺炎支原体肺炎诊疗指南2023年版",
    "中国儿科超药品说明书用药专家共识",
    "国家基本药物目录2018年版",
)
_MEDICAL_SIGNAL_TERMS = (
    "治疗", "评估", "推荐", "剂量", "疗程", "用药", "抗生素",
    "症状", "检查", "预防", "接种", "免疫", "住院", "重症",
    "病原体", "疗效", "给药", "静脉", "口服", "qd", "bid", "mg",
)

# 文件名关键词 → 权威等级
_AUTHORITY_KEYWORD_MAP = {
    "处方集": "national_pharmacopoeia",
    "基本药物目录": "national_pharmacopoeia",
    "basic_drug": "national_pharmacopoeia",
    "诊疗指南": "clinical_guideline",
    "诊疗规范": "clinical_guideline",
    "肺炎支原体肺炎": "clinical_guideline",
    "社区获得性肺炎": "clinical_guideline",
    "内科分册": "clinical_guideline",
    "指导原则": "clinical_guideline",
    "NICE": "clinical_guideline",
    "guideline": "clinical_guideline",
    "超说明书": "expert_consensus",
    "专家共识": "expert_consensus",
    "consensus": "expert_consensus",
}

_REQUIRED_TERM_GROUPS = [
    ("阿奇霉素",),
    ("氨溴索", "沐舒坦"),
    ("红霉素",),
    ("罗红霉素",),
    ("克拉霉素",),
    ("阿莫西林",),
    ("美罗培南",),
    ("头孢",),
    ("静脉", "静点", "静注", "静滴", "静脉注射", "静脉滴注"),
]

_COMBINATION_QUERY_TERMS = (
    "同时", "联合", "联用", "合并", "多种", "两种", "三种", "交替", "一起",
)
_AUDIT_FIELD_QUERY_TERMS = (
    "过敏史", "肝肾功能", "肝功能", "肾功能", "相互作用", "审方", "用药前", "监测",
)
_SOURCE_BOUNDARY_QUERY_TERMS = (
    "基本药物目录", "目录", "作为依据", "剂量依据", "治疗依据",
)
_CATALOG_SOURCE_TERMS = (
    "基本药物目录", "Essential_Medicines", "Essential Medicines", "Model_List",
)
_CLINICAL_SOURCE_TERMS = (
    "诊疗指南", "诊疗规范", "指导原则", "NICE", "guideline", "Guideline",
)
_QUERY_EXPANSION_RULES = (
    (
        ("布洛芬", "对乙酰氨基酚", "退热", "发热"),
        "ibuprofen paracetamol acetaminophen antipyretic fever simultaneously alternating distress next dose",
    ),
    (
        ("止咳", "化痰", "祛痰", "咳嗽", "有痰"),
        "acute cough mucolytic antitussive upper respiratory tract infection acute bronchitis",
    ),
    (
        ("过敏史", "过敏", "用药前"),
        "drug allergy allergy status prescribing dispensing administering penicillin cephalosporin",
    ),
    (
        ("肝肾功能", "肝功能", "肾功能", "相互作用", "多种药物", "审方"),
        "renal function liver function drug interaction medication review pharmacist prescription order",
    ),
    (
        ("联合", "联用", "合并", "同时", "大环内酯", "头孢"),
        "antimicrobial combination antibiotic combination adverse reactions macrolide cephalosporin",
    ),
    (
        ("基本药物目录", "剂量依据", "治疗依据", "作为依据"),
        "essential medicines list formulation dose evidence source scope",
    ),
)

_LEXICAL_FALLBACK_RULES = (
    {
        "triggers": ("布洛芬", "对乙酰氨基酚", "退热", "发热"),
        "source": "NICE_NG143_Fever_in_under_5s.pdf",
        "terms": (
            "paracetamol", "ibuprofen", "antipyretic", "fever",
            "simultaneously", "alternating", "distress",
        ),
    },
    {
        "triggers": ("止咳", "化痰", "祛痰", "咳嗽", "有痰"),
        "source": "NICE_NG120_Acute_cough_antimicrobial_prescribing.pdf",
        "terms": (
            "acute cough", "upper respiratory tract infection", "acute bronchitis",
            "honey", "self-care", "mucolytic", "antitussive",
        ),
    },
    {
        "triggers": ("布地奈德", "雾化", "长期", "每天使用", "哮喘"),
        "source": "NICE_NG245_Asthma_diagnosis_monitoring_management.pdf",
        "terms": (
            "asthma", "inhaled corticosteroid", "ICS", "budesonide",
            "review", "monitoring", "dose",
        ),
    },
    {
        "triggers": ("喘息", "毛细支气管炎", "雾化药物", "抗生素同时"),
        "source": "NICE_NG9_Bronchiolitis_in_children.pdf",
        "terms": (
            "bronchiolitis", "antibiotics", "salbutamol", "ipratropium bromide",
            "systemic or inhaled corticosteroids", "do not use",
        ),
    },
    {
        "triggers": ("联合", "联用", "合并", "同时", "多种", "头孢", "抗生素"),
        "source": "抗菌药物临床应用指导原则（2015年版）.pdf",
        "terms": (
            "联合用药", "抗菌药物联合", "2种", "3种", "不良反应",
            "单一药物", "联合应用",
        ),
    },
    {
        "triggers": ("基本药物目录", "剂量依据", "治疗依据", "作为依据"),
        "source": "国家基本药物目录（2018年版）.pdf",
        "terms": (
            "国家基本药物目录", "药品名称", "剂型", "规格", "阿奇霉素",
        ),
    },
    {
        "triggers": ("48-72", "48～72", "48 至 72", "48 小时", "没有好转", "换药", "疗效不佳", "无改善", "再次评估", "治疗无效"),
        "source": "儿童社区获得性肺炎诊疗规范（2019年版）.pdf",
        "terms": (
            "48", "72", "再次评估", "症状无改善", "疗效", "抗生素覆盖",
            "剂量", "耐药", "基础疾病", "并发症",
        ),
    },
    {
        "triggers": ("肝肾功能", "肝功能", "肾功能", "相互作用", "多种药物", "审方", "药师"),
        "source": "抗菌药物临床应用指导原则（2015年版）.pdf",
        "terms": (
            "电子处方系统", "药师审方", "肝肾功能检查结果", "用药医嘱",
            "临床指南", "药物相互作用", "肾功能不全",
        ),
    },
)

_SAFETY_PRINCIPLE_TERMS = (
    "联合用药", "抗菌药物联合", "3种及3种以上", "不良反应",
    "相互作用", "肝肾功能", "用药合理性", "药师", "审核",
    "combination", "drug interaction", "renal function", "liver function",
)


# ────────────────────────────────────────────
# 返回数据结构
# ────────────────────────────────────────────
@dataclass
class RetrievedChunk:
    """检索结果单元, 带完整溯源信息和权威度评分"""
    content: str              # 切片文本
    granularity: int          # 所属粒度 (128/512/1024)
    distance: float           # 向量距离 (越小越相关, ChromaDB 默认 L2)
    relevance_score: float    # 相关性分数 = 1 / (1 + distance)
    authority_weight: float   # 权威度权重 (来自 config)
    final_score: float        # 最终综合分 = relevance_score × authority_weight
    source_file: str          # 来源文件名
    page_number: int          # 来源页码
    chapter_title: str        # 来源章节
    block_type: str           # 块类型 (text/table)


# ────────────────────────────────────────────
# 多粒度检索器
# ────────────────────────────────────────────
class MultiGranularityRetriever:
    """
    三粒度 ChromaDB 检索器

    使用方法:
        retriever = MultiGranularityRetriever()

        # 三粒度融合检索
        results = retriever.retrieve("阿莫西林儿童剂量", top_k=5)

        # 单粒度检索
        results = retriever.retrieve("阿莫西林儿童剂量", top_k=5, granularity=128)
    """

    def __init__(self, persist_dir: str | None = None):
        settings = get_settings()
        self._settings = settings
        self._persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self._index_status = self._load_index_status()

        if not self._index_status.get("ready", False):
            logger.warning(
                "[Retriever] 索引未就绪，检索将返回空证据: %s",
                self._index_status.get("reason") or self._index_status.get("missing_sources"),
            )
            self._chroma = None
            self._embed_fn = None
            return

        embedding_mismatch = self._embedding_status_error()
        if embedding_mismatch:
            logger.warning("[Retriever] %s", embedding_mismatch)
            self._index_status["ready"] = False
            self._index_status["reason"] = embedding_mismatch
            self._chroma = None
            self._embed_fn = None
            return

        # ChromaDB 只读客户端
        self._chroma = chromadb.PersistentClient(path=self._persist_dir)

        # Embedding 函数 (与 indexer 共享同一实现)
        self._embed_fn = create_embedding_function()

        logger.info(f"[Retriever] 初始化完成: {self._persist_dir}")

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        granularity: Literal[128, 512, 1024] | None = None,
        include_tables: bool = True,
    ) -> list[RetrievedChunk]:
        """
        执行多粒度语义检索

        Args:
            query: 查询文本 (儿科用药相关问题)
            top_k: 每个粒度返回的结果数, 默认读 config.RETRIEVAL_TOP_K
            granularity: 指定单粒度检索; None = 三粒度融合
            include_tables: 是否包含表格类型的切片

        Returns:
            按 final_score 降序排列的检索结果列表
        """
        if not self._index_status.get("ready", False):
            logger.warning(
                "[Retriever] 索引未通过完整性校验，拒绝返回伪完整证据: %s",
                self._index_status.get("reason") or self._index_status.get("missing_sources"),
            )
            return []

        k = top_k or self._settings.RETRIEVAL_TOP_K

        # 生成查询向量
        logger.info(f"[Retriever] 查询: '{query[:50]}...' top_k={k}")
        expanded_query = self._expand_query(query)
        query_embedding = self._embed_fn([expanded_query])[0]

        # 确定检索哪些粒度
        if granularity is not None:
            target_granularities = [granularity]
        else:
            target_granularities = [128, 512, 1024]

        all_results: list[RetrievedChunk] = []

        for g in target_granularities:
            col_name = _COLLECTION_NAMES.get(g)
            if col_name is None:
                continue
            try:
                collection = self._chroma.get_collection(name=col_name)
            except Exception:
                logger.warning(f"[Retriever] Collection {col_name} 不存在, 跳过")
                continue

            if collection.count() == 0:
                logger.warning(f"[Retriever] Collection {col_name} 为空, 跳过")
                continue

            # 构建 where 过滤条件 (排除表格)
            where = None
            if not include_tables:
                where = {"block_type": {"$eq": "text"}}

            # ChromaDB 向量查询
            response = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, collection.count()),
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            chunks = self._parse_chroma_response(response, g)
            all_results.extend(chunks)
            logger.info(f"[Retriever] 粒度 {g}: 命中 {len(chunks)} 条")

        all_results.extend(
            self._collect_lexical_fallback_candidates(
                query,
                target_granularities,
                include_tables=include_tables,
            )
        )

        # 权威度加权 + 排序
        for chunk in all_results:
            base_score = chunk.relevance_score * chunk.authority_weight
            chunk.final_score = self._adjust_score_for_query(query, chunk, base_score)

        all_results.sort(key=lambda c: c.final_score, reverse=True)
        all_results = self._apply_required_term_filter(query, all_results)
        all_results = self._deduplicate_results(all_results)

        logger.info(f"[Retriever] 检索完成, 共返回 {len(all_results)} 条结果")
        return all_results

    def get_stats(self) -> dict[str, int]:
        """获取各 Collection 的文档数量"""
        if self._chroma is None:
            return {name: 0 for name in _COLLECTION_NAMES.values()}

        stats = {}
        for g, name in _COLLECTION_NAMES.items():
            try:
                col = self._chroma.get_collection(name=name)
                stats[name] = col.count()
            except Exception:
                stats[name] = 0
        return stats

    # ── 内部方法 ──

    def _load_index_status(self) -> dict[str, object]:
        """读取知识库完整性状态；缺失或损坏时保守视为未就绪。"""
        status_path = Path(self._persist_dir) / "index_status.json"
        if not status_path.exists():
            return {
                "ready": False,
                "reason": f"{status_path.name} missing",
            }

        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "ready": False,
                "reason": f"failed to read {status_path.name}: {exc}",
            }

        if not isinstance(status, dict):
            return {
                "ready": False,
                "reason": f"{status_path.name} is not a JSON object",
            }
        status.setdefault("ready", False)
        return status

    def _embedding_status_error(self) -> str:
        """新索引必须使用与查询端一致的 embedding 配置；旧索引缺元数据时兼容放行。"""
        index_provider = self._index_status.get("embedding_provider")
        index_model = self._index_status.get("embedding_model")
        if not index_provider or not index_model:
            logger.warning(
                "[Retriever] index_status.json 缺少 embedding 元数据，按旧索引兼容处理"
            )
            return ""

        current_provider = self._settings.EMBEDDING_PROVIDER
        current_model = self._settings.EMBEDDING_MODEL
        if index_provider != current_provider or index_model != current_model:
            return (
                "索引 embedding 配置与当前查询配置不一致，拒绝检索以避免向量空间错配: "
                f"index=({index_provider}, {index_model}), "
                f"current=({current_provider}, {current_model})"
            )
        return ""

    def _parse_chroma_response(
        self, response: dict, granularity: int
    ) -> list[RetrievedChunk]:
        """解析 ChromaDB 返回结果, 注入权威度权重"""
        results = []

        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            if self._is_noise_chunk(doc):
                continue

            # 相关性分数: 从 L2 距离转换, 距离越小分越高
            relevance = 1.0 / (1.0 + dist)

            # 权威度: 根据文件名关键词确定
            authority = self._get_authority_weight(meta.get("source_file", ""))

            results.append(RetrievedChunk(
                content=doc,
                granularity=granularity,
                distance=dist,
                relevance_score=relevance,
                authority_weight=authority,
                final_score=0.0,  # 在外层统一计算
                source_file=meta.get("source_file", ""),
                page_number=meta.get("page_number", 0),
                chapter_title=meta.get("chapter_title", ""),
                block_type=meta.get("block_type", "text"),
            ))

        return results

    def _collect_lexical_fallback_candidates(
        self,
        query: str,
        granularities: list[int],
        include_tables: bool = True,
    ) -> list[RetrievedChunk]:
        """补回少量 source-specific 候选，降低跨语言或原则性证据漏召回。"""
        rules = [
            rule for rule in _LEXICAL_FALLBACK_RULES
            if any(term in query for term in rule["triggers"])
        ]
        if not rules or self._chroma is None:
            return []

        max_per_rule = max(2, min(4, getattr(self._settings, "RETRIEVAL_TOP_K", 3)))
        candidates: list[RetrievedChunk] = []

        for rule in rules:
            per_rule: list[RetrievedChunk] = []
            for granularity in granularities:
                col_name = _COLLECTION_NAMES.get(granularity)
                if col_name is None:
                    continue
                try:
                    collection = self._chroma.get_collection(name=col_name)
                    response = collection.get(
                        where={"source_file": {"$eq": rule["source"]}},
                        include=["documents", "metadatas"],
                    )
                except Exception as exc:
                    logger.debug(
                        "[Retriever] lexical fallback skipped for %s/%s: %s",
                        rule["source"],
                        col_name,
                        exc,
                    )
                    continue

                documents = response.get("documents", []) or []
                metadatas = response.get("metadatas", []) or []
                for doc, meta in zip(documents, metadatas):
                    if not include_tables and meta.get("block_type") == "table":
                        continue
                    if self._is_noise_chunk(doc):
                        continue

                    lexical_score = min(
                        1.0,
                        self._lexical_score(doc, rule["terms"])
                        + self._fallback_phrase_bonus(doc, rule),
                    )
                    if lexical_score <= 0:
                        continue

                    relevance = min(0.92, 0.52 + 0.40 * lexical_score)
                    distance = (1.0 / relevance) - 1.0
                    per_rule.append(RetrievedChunk(
                        content=doc,
                        granularity=granularity,
                        distance=distance,
                        relevance_score=relevance,
                        authority_weight=self._get_authority_weight(meta.get("source_file", "")),
                        final_score=0.0,
                        source_file=meta.get("source_file", ""),
                        page_number=meta.get("page_number", 0),
                        chapter_title=meta.get("chapter_title", ""),
                        block_type=meta.get("block_type", "text"),
                    ))

            per_rule.sort(key=lambda chunk: chunk.relevance_score, reverse=True)
            candidates.extend(per_rule[:max_per_rule])

        return candidates

    @staticmethod
    def _lexical_score(content: str, terms: tuple[str, ...]) -> float:
        text = (content or "").lower()
        unique_terms = tuple(dict.fromkeys(term.lower() for term in terms if term))
        if not unique_terms:
            return 0.0

        hits = sum(1 for term in unique_terms if term in text)
        return hits / len(unique_terms)

    @staticmethod
    def _fallback_phrase_bonus(content: str, rule: dict) -> float:
        source = rule.get("source", "")
        if "抗菌药物临床应用指导原则" not in source:
            return 0.0

        bonus = 0.0
        if "联合用药通常采用" in content:
            bonus += 0.35
        if (
            ("3种及3种以上" in content or "3 种及 3 种以上" in content)
            and "不良反应亦可能增多" in content
        ):
            bonus += 0.25
        return bonus

    @staticmethod
    def _is_noise_chunk(content: str) -> bool:
        """过滤明显不应作为临床证据展示的参考文献、标题与解析占位噪声。"""
        if not content:
            return True
        if _REFERENCE_HEADING_RE.search(content):
            return True
        if _PICTURE_PLACEHOLDER_RE.search(content):
            return True
        if _PICTURE_TEXT_MARKER_RE.search(content):
            return True
        if MultiGranularityRetriever._looks_like_title_only_chunk(content):
            return True
        return False

    @staticmethod
    def _looks_like_title_only_chunk(content: str) -> bool:
        lines = [line.strip().strip("#").strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return True

        compact = _TITLE_NORMALIZE_RE.sub("", "".join(lines))
        if not compact:
            return True

        has_signal = any(term in content for term in _MEDICAL_SIGNAL_TERMS)
        title_hit = any(term in compact for term in _TITLE_ONLY_TERMS)
        if title_hit and len(lines) <= 2 and not has_signal:
            return True
        if compact.startswith("附录") and title_hit and len(compact) <= 40:
            return True
        return len(compact) < 28 and not has_signal

    @staticmethod
    def _deduplicate_results(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        deduped = []
        seen = set()
        for chunk in chunks:
            signature = re.sub(r"\s+", "", chunk.content or "")[:120]
            key = (chunk.source_file, chunk.page_number, signature)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(chunk)
        return deduped

    @staticmethod
    def _apply_required_term_filter(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """If a concrete drug is named in the query, evidence must mention it or an alias."""
        required_groups = [
            group for group in _REQUIRED_TERM_GROUPS
            if any(term in query for term in group)
        ]
        if not required_groups:
            return chunks

        filtered = []
        for chunk in chunks:
            content = getattr(chunk, "content", "") or ""
            if all(any(term in content for term in group) for group in required_groups):
                filtered.append(chunk)
        if filtered and MultiGranularityRetriever._is_broad_review_query(query):
            principle_chunks = [
                chunk for chunk in chunks
                if MultiGranularityRetriever._has_safety_principle_signal(
                    getattr(chunk, "content", "") or ""
                )
            ]
            merged = []
            seen = set()
            for chunk in [*filtered, *principle_chunks]:
                marker = id(chunk)
                if marker in seen:
                    continue
                seen.add(marker)
                merged.append(chunk)
            return merged
        if not filtered and MultiGranularityRetriever._is_broad_review_query(query):
            return chunks
        return filtered

    @staticmethod
    def _expand_query(query: str) -> str:
        """Append compact bilingual audit terms for retrieval only."""
        additions: list[str] = []
        for triggers, expansion in _QUERY_EXPANSION_RULES:
            if any(term in query for term in triggers):
                additions.append(expansion)
        if not additions:
            return query
        return f"{query} {' '.join(dict.fromkeys(additions))}"

    @staticmethod
    def _is_broad_review_query(query: str) -> bool:
        return any(term in query for term in (
            *_COMBINATION_QUERY_TERMS,
            *_AUDIT_FIELD_QUERY_TERMS,
            *_SOURCE_BOUNDARY_QUERY_TERMS,
        ))

    @staticmethod
    def _has_safety_principle_signal(content: str) -> bool:
        text = content or ""
        text_lower = text.lower()
        return any(term in text or term.lower() in text_lower for term in _SAFETY_PRINCIPLE_TERMS)

    @staticmethod
    def _adjust_score_for_query(query: str, chunk: RetrievedChunk, base_score: float) -> float:
        """Query-aware reranking: keep catalog evidence from replacing safety guidance."""
        score = base_score
        source = chunk.source_file or ""
        content = chunk.content or ""
        source_text = f"{source} {content}"

        is_catalog = any(term in source for term in _CATALOG_SOURCE_TERMS)
        is_clinical = any(term in source for term in _CLINICAL_SOURCE_TERMS)
        is_combination_query = any(term in query for term in _COMBINATION_QUERY_TERMS)
        is_audit_query = any(term in query for term in _AUDIT_FIELD_QUERY_TERMS)
        is_source_boundary_query = any(term in query for term in _SOURCE_BOUNDARY_QUERY_TERMS)

        if is_source_boundary_query and is_catalog:
            score *= 1.45
        elif (is_combination_query or is_audit_query) and is_catalog:
            score *= 0.45

        if (is_combination_query or is_audit_query) and is_clinical:
            score *= 1.12

        if any(term in query for term in ("布洛芬", "对乙酰氨基酚", "退热", "发热")):
            if any(term in source_text for term in ("NICE_NG143", "Fever", "paracetamol", "ibuprofen")):
                score *= 1.65

        if any(term in query for term in ("止咳", "化痰", "祛痰", "咳嗽", "有痰")):
            if any(term in source_text for term in ("NICE_NG120", "acute cough", "mucolytic", "antitussive")):
                score *= 1.65

        if "过敏史" in query or "过敏" in query:
            if any(term in source_text for term in ("NICE_CG183", "Drug_allergy", "drug allergy", "过敏史")):
                score *= 1.6

        if any(term in query for term in ("肝肾功能", "肝功能", "肾功能", "相互作用", "审方")):
            if any(term in source_text for term in ("抗菌药物临床应用指导原则", "肝肾功能", "药师审方", "renal function")):
                score *= 1.45
            if any(term in content for term in ("电子处方系统", "药师审方", "肝肾功能检查结果", "用药医嘱")):
                score *= 1.55

        if any(term in query for term in ("48-72", "48～72", "48 小时", "没有好转", "无改善", "再次评估")):
            if "儿童社区获得性肺炎诊疗规范" in source and "再次评估" in content and "48" in content and "72" in content:
                score *= 1.65

        if is_combination_query:
            if (
                "联合用药通常采用" in content
                or ("联合用药" in content and ("3种" in content or "不良反应亦可能增多" in content))
            ):
                score *= 1.75
            elif any(term in content for term in ("联合用药", "抗菌药物联合", "combination")):
                score *= 1.25

        return score

    def _get_authority_weight(self, filename: str) -> float:
        """根据文件名关键词返回对应的权威度权重"""
        weights = self._settings.AUTHORITY_WEIGHTS
        for keyword, level in _AUTHORITY_KEYWORD_MAP.items():
            if keyword in filename:
                return weights.get(level, weights["default"])
        return weights["default"]
