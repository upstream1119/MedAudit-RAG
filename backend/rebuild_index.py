"""
重建 Medaudit-RAG 知识库索引。

执行内容：
1. 清空配置指定的 ChromaDB 目标目录
2. 读取 data/guidelines 下的 PDF
3. 解析 -> 切分 -> 建立三粒度索引
4. 输出按文档和粒度的重建摘要
"""

from __future__ import annotations

import json
import logging
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from prepare_kb_sources import compute_sha256, load_manifest


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rebuild_index")


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def report_path(root: Path, stem: str, chroma_dir: Path) -> Path:
    """为非默认索引生成独立报告文件，避免覆盖旧基线。"""
    suffix = "" if chroma_dir.name == "chroma_db" else f"_{chroma_dir.name}"
    return root / "docs" / f"{stem}{suffix}.json"


def _guideline_paths(root: Path) -> list[Path]:
    guideline_dir = root / "data" / "guidelines"
    manifest_path = guideline_dir / "source_manifest.json"
    manifest = load_manifest(manifest_path)
    pdfs: list[Path] = []

    for source in manifest["sources"]:
        if not source.get("included_in_kb"):
            continue
        source_id = source.get("source_id", "UNKNOWN")
        if source.get("status") not in {"approved", "indexed"}:
            raise ValueError(f"{source_id} 已标记入库但状态不是 approved/indexed")

        filename = source.get("filename")
        expected_sha = source.get("sha256")
        if not filename or not expected_sha:
            raise ValueError(f"{source_id} 缺少 filename 或 sha256")

        pdf_path = guideline_dir / filename
        if not pdf_path.is_file():
            raise FileNotFoundError(f"{source_id} 已批准但正式 PDF 不存在: {pdf_path}")

        actual_sha = compute_sha256(pdf_path)
        if actual_sha != expected_sha:
            raise ValueError(f"{source_id} SHA-256 不匹配，拒绝重建索引")

        pdfs.append(pdf_path)

    if not pdfs:
        raise FileNotFoundError(f"manifest 中没有已批准的正式知识库 PDF: {manifest_path}")
    return pdfs


def _reset_chroma_dir(chroma_dir: Path) -> None:
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)


def _build_index_status(
    pdfs: list[Path],
    per_doc_summary: dict[str, dict[str, object]],
    source_inspections: dict[str, dict[str, object]],
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    embedding_model_path: str | None = None,
    collection_dimensions: dict[str, int] | None = None,
    expected_embedding_dimension: int | None = None,
) -> dict[str, object]:
    expected_sources = [pdf.name for pdf in pdfs]
    indexed_sources = [
        source
        for source in expected_sources
        if per_doc_summary.get(source, {}).get("blocks_total", 0) > 0
    ]
    missing_sources = [
        source
        for source in expected_sources
        if source not in indexed_sources
    ]
    scan_heavy_sources = [
        source
        for source, inspection in source_inspections.items()
        if inspection.get("scan_heavy")
    ]
    incomplete_sources = sorted(set(missing_sources + scan_heavy_sources))
    dimensions = collection_dimensions or {}
    dimension_mismatches = {
        name: dimensions.get(name, 0)
        for name in ("detail_128", "concept_512", "context_1024")
        if expected_embedding_dimension
        and dimensions.get(name, 0) != expected_embedding_dimension
    }
    ready = not incomplete_sources and not dimension_mismatches

    return {
        "ready": ready,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_model_path": embedding_model_path,
        "expected_embedding_dimension": expected_embedding_dimension,
        "collection_dimensions": dimensions,
        "dimension_mismatches": dimension_mismatches,
        "expected_sources": expected_sources,
        "indexed_sources": indexed_sources,
        "missing_sources": missing_sources,
        "scan_heavy_sources": scan_heavy_sources,
        "incomplete_sources": incomplete_sources,
        "source_inspections": source_inspections,
        "reason": (
            ""
            if ready
            else "core guideline PDFs or embedding dimensions were incomplete"
        ),
    }


def main() -> None:
    from app.config import get_settings
    from app.knowledge.chunker import SemanticChunker
    from app.knowledge.indexer import VectorIndexer
    from app.knowledge.parser import DualTrackMedicalParser

    root = _project_root()
    settings = get_settings()
    parser = DualTrackMedicalParser()
    chunker = SemanticChunker()

    configured_chroma_dir = Path(settings.CHROMA_PERSIST_DIR)
    chroma_dir = (
        configured_chroma_dir
        if configured_chroma_dir.is_absolute()
        else root / configured_chroma_dir
    ).resolve()
    pdfs = _guideline_paths(root)

    logger.info("准备重建索引，目标目录: %s", chroma_dir)
    logger.info("共发现 %s 份 PDF", len(pdfs))
    for pdf in pdfs:
        logger.info(" - %s", pdf.name)

    _reset_chroma_dir(chroma_dir)
    indexer = VectorIndexer(persist_dir=str(chroma_dir))

    per_doc_summary: dict[str, dict[str, object]] = {}
    source_inspections: dict[str, dict[str, object]] = {}
    total_chunk_counter: Counter[int] = Counter()
    total_block_counter: Counter[str] = Counter()

    for pdf_path in pdfs:
        logger.info("开始处理: %s", pdf_path.name)
        inspection = parser.inspect_source(pdf_path)
        source_inspections[pdf_path.name] = asdict(inspection)
        if inspection.scan_heavy:
            logger.warning(
                "检测到扫描件倾向: %s | sampled=%s text_pages=%s image_pages=%s",
                pdf_path.name,
                inspection.sampled_pages,
                inspection.text_pages,
                inspection.image_pages,
            )

        blocks = parser.parse(pdf_path)
        chunks_by_granularity = chunker.chunk_all_granularities(blocks)
        write_counts = indexer.index_chunks(chunks_by_granularity)

        block_counter = Counter(block.metadata.block_type for block in blocks)
        page_numbers = [block.metadata.page_number for block in blocks if block.metadata.page_number]
        granularity_counts = {str(g): len(chunks) for g, chunks in chunks_by_granularity.items()}

        per_doc_summary[pdf_path.name] = {
            "blocks_total": len(blocks),
            "block_types": dict(block_counter),
            "page_min": min(page_numbers) if page_numbers else None,
            "page_max": max(page_numbers) if page_numbers else None,
            "granularity_chunks": granularity_counts,
            "index_written": {str(k): v for k, v in write_counts.items()},
            "source_inspection": source_inspections[pdf_path.name],
        }

        total_block_counter.update(block_counter)
        total_chunk_counter.update({g: len(chunks) for g, chunks in chunks_by_granularity.items()})
        logger.info("完成处理: %s", pdf_path.name)

    collection_dimensions = indexer.get_collection_dimensions()
    embedding_model_path = (
        settings.LOCAL_EMBEDDING_MODEL_PATH
        if settings.EMBEDDING_PROVIDER == "local"
        else None
    )
    expected_embedding_dimension = (
        settings.LOCAL_EMBEDDING_DIMENSION
        if settings.EMBEDDING_PROVIDER == "local"
        else None
    )
    index_status = _build_index_status(
        pdfs,
        per_doc_summary,
        source_inspections,
        embedding_provider=settings.EMBEDDING_PROVIDER,
        embedding_model=settings.EMBEDDING_MODEL,
        embedding_model_path=embedding_model_path,
        collection_dimensions=collection_dimensions,
        expected_embedding_dimension=expected_embedding_dimension,
    )

    summary = {
        "pdf_count": len(pdfs),
        "pdfs": [pdf.name for pdf in pdfs],
        "index_status": index_status,
        "total_blocks_by_type": dict(total_block_counter),
        "total_chunks_by_granularity": {str(k): v for k, v in total_chunk_counter.items()},
        "collection_stats": indexer.get_collection_stats(),
        "per_document": per_doc_summary,
    }

    summary_path = report_path(root, "index_rebuild_summary", chroma_dir)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    status_path = chroma_dir / "index_status.json"
    status_path.write_text(json.dumps(index_status, ensure_ascii=False, indent=2), encoding="utf-8")

    if not index_status["ready"]:
        logger.warning(
            "索引未就绪，缺失资料=%s，向量维度异常=%s",
            index_status["missing_sources"],
            index_status["dimension_mismatches"],
        )
    logger.info("索引重建完成，摘要已写入: %s", summary_path)
    logger.info("索引状态已写入: %s", status_path)


if __name__ == "__main__":
    main()
