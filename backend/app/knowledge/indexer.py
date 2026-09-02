"""
Medaudit-RAG 向量索引器 (Vector Indexer)
=========================================
职责:
  1. 将 SemanticChunker 产出的 TextChunk 批量写入 ChromaDB
  2. 按粒度创建 3 个独立 Collection (detail_128 / concept_512 / context_1024)
  3. 每条记录强制携带溯源 Metadata (书名, 页码, 章节, 文件哈希)
  4. 内置 SHA-256 哈希去重, 同一份 PDF 不会重复灌入
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from openai import OpenAI

from app.config import get_settings
from app.knowledge.chunker import TextChunk

logger = logging.getLogger(__name__)

# 粒度 → Collection 名称映射
_COLLECTION_NAMES = {
    128: "detail_128",
    512: "concept_512",
    1024: "context_1024",
}


class ZhipuEmbeddingFunction:
    """
    OpenAI-compatible 远程 Embedding 封装, 实现 ChromaDB 的 EmbeddingFunction 协议

    ChromaDB 要求实现 __call__(self, input: Documents) -> Embeddings
    """

    def __init__(self):
        settings = get_settings()
        # 智谱的 embedding API 兼容 OpenAI 协议
        embedding_base_urls = {
            "zhipu": "https://open.bigmodel.cn/api/paas/v4",
            "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        }
        api_keys = {
            "zhipu": settings.ZHIPU_API_KEY,
            "dashscope": settings.DASHSCOPE_API_KEY,
        }
        provider = settings.EMBEDDING_PROVIDER
        self._provider = provider
        self._client = OpenAI(
            api_key=api_keys[provider],
            base_url=embedding_base_urls[provider],
        )
        self._model = settings.EMBEDDING_MODEL

    def _batch_size(self) -> int:
        """DashScope OpenAI 兼容 embedding 接口单批最多 10 条。"""
        if self._provider == "dashscope":
            return 10
        return 64

    def __call__(self, input: list[str]) -> list[list[float]]:
        """批量生成 Embedding 向量"""
        # OpenAI 兼容协议: 一次最多传入 2048 条
        embeddings = []
        batch_size = self._batch_size()
        for i in range(0, len(input), batch_size):
            batch = input[i:i + batch_size]
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
            )
            embeddings.extend([item.embedding for item in response.data])
        return embeddings


class LocalEmbeddingDependencyError(RuntimeError):
    """本地 embedding 可选依赖缺失时抛出明确错误。"""


class LocalSentenceTransformerEmbeddingFunction:
    """基于 sentence-transformers 的本地离线 Embedding 封装。"""

    def __init__(self):
        settings = get_settings()
        self._model_name = (
            getattr(settings, "LOCAL_EMBEDDING_MODEL_PATH", "")
            or settings.EMBEDDING_MODEL
        )
        configured_device = getattr(
            settings,
            "LOCAL_EMBEDDING_DEVICE",
            "auto",
        )
        self._device = (
            None
            if configured_device == "auto"
            else configured_device
        )
        self._batch_size = getattr(
            settings,
            "LOCAL_EMBEDDING_BATCH_SIZE",
            None,
        )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise LocalEmbeddingDependencyError(
                "EMBEDDING_PROVIDER=local 需要安装可选依赖。"
                "请先安装 backend/requirements-local-embedding.txt，"
                "并确认本地模型已下载或可访问。"
            ) from exc

        model_kwargs = {}
        if self._device is not None:
            model_kwargs["device"] = self._device
        self._model = SentenceTransformer(self._model_name, **model_kwargs)

    def __call__(self, input: list[str]) -> list[list[float]]:
        """批量生成本地 Embedding 向量。"""
        if not input:
            return []
        encode_kwargs = {
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }
        if self._batch_size is not None:
            encode_kwargs["batch_size"] = self._batch_size
        embeddings = self._model.encode(input, **encode_kwargs)
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
        return [list(row) for row in embeddings]


def create_embedding_function():
    """根据配置创建索引/检索共用的 Embedding 函数。"""
    settings = get_settings()
    if settings.EMBEDDING_PROVIDER == "local":
        return LocalSentenceTransformerEmbeddingFunction()
    return ZhipuEmbeddingFunction()


class VectorIndexer:
    """
    三粒度向量索引管理器

    使用方法:
        indexer = VectorIndexer()
        indexer.index_chunks(all_chunks)  # all_chunks = {128: [...], 512: [...], 1024: [...]}
    """

    def __init__(self, persist_dir: str | None = None):
        settings = get_settings()
        self._persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR

        # 初始化 ChromaDB 持久化客户端
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
        self._chroma = chromadb.PersistentClient(path=self._persist_dir)

        # 初始化 Embedding 函数
        self._embed_fn = create_embedding_function()

        # 记录已入库的 (文件哈希, 粒度) 组合 (防重复灌入)
        self._indexed_hashes: set[str] = set()

        logger.info(f"[Indexer] ChromaDB 初始化完成: {self._persist_dir}")

    def index_chunks(
        self,
        chunks_by_granularity: dict[int, list[TextChunk]],
        dry_run: bool = False,
    ) -> dict[int, int]:
        """
        将三粒度切片批量写入 ChromaDB

        Args:
            chunks_by_granularity: {128: [TextChunk, ...], 512: [...], 1024: [...]}
            dry_run: 仅统计, 不实际写入 (调试用)

        Returns:
            {128: 写入数, 512: 写入数, 1024: 写入数}
        """
        results = {}

        for granularity, chunks in chunks_by_granularity.items():
            if granularity not in _COLLECTION_NAMES:
                logger.warning(f"未知粒度 {granularity}, 跳过")
                continue

            # 过滤已入库的文件 (按 文件哈希+粒度 组合)
            new_chunks = self._filter_duplicates(chunks, granularity)
            if not new_chunks:
                logger.info(f"[Indexer] 粒度 {granularity}: 无新数据需要写入")
                results[granularity] = 0
                continue

            if dry_run:
                logger.info(f"[Indexer] [DRY RUN] 粒度 {granularity}: 将写入 {len(new_chunks)} 条")
                results[granularity] = len(new_chunks)
                continue

            # 写入 ChromaDB
            count = self._upsert_to_collection(granularity, new_chunks)
            results[granularity] = count

            # 记录已入库的哈希 (按 文件哈希+粒度 组合去重)
            for chunk in new_chunks:
                self._indexed_hashes.add(f"{chunk.metadata.source_hash}_g{granularity}")

        return results

    def get_collection_stats(self) -> dict[str, int]:
        """获取各 Collection 的文档数量"""
        stats = {}
        for granularity, name in _COLLECTION_NAMES.items():
            try:
                col = self._chroma.get_collection(name=name)
                stats[name] = col.count()
            except Exception:
                stats[name] = 0
        return stats

    def get_collection_dimensions(self) -> dict[str, int]:
        """从各 Collection 的真实向量读取维度，供构建后审计。"""
        dimensions = {}
        for name in _COLLECTION_NAMES.values():
            try:
                collection = self._chroma.get_collection(name=name)
                result = collection.get(limit=1, include=["embeddings"])
                embeddings = result.get("embeddings")
                dimensions[name] = (
                    len(embeddings[0])
                    if embeddings is not None and len(embeddings) > 0
                    else 0
                )
            except Exception:
                dimensions[name] = 0
        return dimensions

    # ── 内部方法 ──

    def _filter_duplicates(self, chunks: list[TextChunk], granularity: int) -> list[TextChunk]:
        """基于 (文件哈希+粒度) 组合过滤已入库的切片"""
        return [
            c for c in chunks
            if f"{c.metadata.source_hash}_g{granularity}" not in self._indexed_hashes
        ]

    def _upsert_to_collection(
        self, granularity: int, chunks: list[TextChunk]
    ) -> int:
        """将切片批量写入指定粒度的 Collection"""
        col_name = _COLLECTION_NAMES[granularity]

        # 获取或创建 Collection (不传 embedding_function, 我们手动生成)
        collection = self._chroma.get_or_create_collection(
            name=col_name,
            metadata={"granularity": granularity},
        )

        # 准备数据
        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            # 唯一 ID: 文件哈希前8位_粒度_序号
            chunk_id = (
                f"{chunk.metadata.source_hash[:8]}"
                f"_g{granularity}"
                f"_c{chunk.chunk_index}"
            )
            ids.append(chunk_id)
            documents.append(chunk.content)
            metadatas.append({
                "source_file": chunk.metadata.source_file,
                "source_hash": chunk.metadata.source_hash,
                "page_number": chunk.metadata.page_number,
                "chapter_title": chunk.metadata.chapter_title,
                "block_type": chunk.metadata.block_type,
                "granularity": granularity,
                "token_count": chunk.token_count,
            })

        # 批量生成 Embedding
        logger.info(f"[Indexer] 粒度 {granularity}: 正在生成 {len(documents)} 条 Embedding...")
        embeddings = self._embed_fn(documents)

        # 批量写入 (ChromaDB 的 upsert 支持幂等)
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            collection.upsert(
                ids=ids[i:end],
                documents=documents[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
            )

        logger.info(f"[Indexer] 粒度 {granularity}: 成功写入 {len(ids)} 条")
        return len(ids)
