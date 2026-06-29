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
from app.knowledge.indexer import ZhipuEmbeddingFunction, _COLLECTION_NAMES

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

        # ChromaDB 只读客户端
        self._chroma = chromadb.PersistentClient(path=self._persist_dir)

        # Embedding 函数 (与 indexer 共享同一实现)
        self._embed_fn = ZhipuEmbeddingFunction()

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

        if is_combination_query:
            if any(term in content for term in ("联合用药", "不良反应", "抗菌药物联合", "combination")):
                score *= 1.35

        return score

    def _get_authority_weight(self, filename: str) -> float:
        """根据文件名关键词返回对应的权威度权重"""
        weights = self._settings.AUTHORITY_WEIGHTS
        for keyword, level in _AUTHORITY_KEYWORD_MAP.items():
            if keyword in filename:
                return weights.get(level, weights["default"])
        return weights["default"]
