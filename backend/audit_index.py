"""
审计当前 Chroma 索引质量，确认来源文件、页码分布与噪声过滤状态。
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import chromadb


_REFERENCE_HEADING_RE = re.compile(r"^\s*##\s*[［\[]?\s*参\s*考\s*文\s*献", re.MULTILINE)
_PICTURE_PLACEHOLDER_RE = re.compile(
    r"\*\*==>\s*picture\s*\[[^\]]+\]\s*intentionally omitted\s*<==\*\*",
    re.IGNORECASE,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def report_path(root: Path, stem: str, chroma_dir: Path) -> Path:
    """为非默认索引生成独立报告文件，避免覆盖旧基线。"""
    suffix = "" if chroma_dir.name == "chroma_db" else f"_{chroma_dir.name}"
    return root / "docs" / f"{stem}{suffix}.json"


def resolve_chroma_dir(root: Path, configured_path: Path) -> Path:
    """将配置中的相对索引目录解析为项目内绝对路径。"""
    target = configured_path if configured_path.is_absolute() else root / configured_path
    return target.resolve()


def _collection_names() -> list[str]:
    return ["detail_128", "concept_512", "context_1024"]


def _expected_sources(root: Path) -> list[str]:
    guideline_dir = root / "data" / "guidelines"
    manifest_path = guideline_dir / "source_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return sorted(
            source["filename"]
            for source in manifest.get("sources", [])
            if source.get("included_in_kb") and source.get("filename")
        )
    return sorted(pdf.name for pdf in guideline_dir.glob("*.pdf"))


def _load_index_status(chroma_dir: Path) -> dict[str, object]:
    status_path = chroma_dir / "index_status.json"
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
    return status if isinstance(status, dict) else {"ready": False, "reason": "index_status.json is not a JSON object"}


def _audit_collection(collection) -> dict[str, object]:
    data = collection.get(include=["metadatas", "documents"])
    metadatas = data.get("metadatas", [])
    documents = data.get("documents", [])
    embedding_data = collection.get(limit=1, include=["embeddings"])
    embeddings = embedding_data.get("embeddings")
    embedding_dimension = (
        len(embeddings[0])
        if embeddings is not None and len(embeddings) > 0 and embeddings[0] is not None
        else 0
    )

    source_counter: Counter[str] = Counter()
    page_buckets: dict[str, list[int]] = defaultdict(list)
    reference_like = 0
    picture_noise = 0

    for meta, doc in zip(metadatas, documents):
        source = meta.get("source_file", "")
        page = int(meta.get("page_number", 0) or 0)
        source_counter[source] += 1
        if page:
            page_buckets[source].append(page)
        if _REFERENCE_HEADING_RE.search(doc or ""):
            reference_like += 1
        if _PICTURE_PLACEHOLDER_RE.search(doc or ""):
            picture_noise += 1

    page_ranges = {
        source: {
            "min": min(pages) if pages else None,
            "max": max(pages) if pages else None,
            "count": len(pages),
        }
        for source, pages in page_buckets.items()
    }

    return {
        "count": collection.count(),
        "embedding_dimension": embedding_dimension,
        "sources": dict(source_counter),
        "page_ranges": page_ranges,
        "reference_like_documents": reference_like,
        "picture_noise_documents": picture_noise,
    }


def build_completeness(
    expected_sources: set[str],
    report: dict[str, object],
    index_status: dict[str, object],
) -> dict[str, object]:
    """综合来源、三层向量维度和构建状态，保守判断索引是否可用。"""
    sources_by_collection = {
        name: set(report.get(name, {}).get("sources", {}).keys())
        for name in _collection_names()
    }
    missing_by_collection = {
        name: sorted(expected_sources - actual_sources)
        for name, actual_sources in sources_by_collection.items()
        if actual_sources != expected_sources
    }
    dimensions = {
        name: int(report.get(name, {}).get("embedding_dimension", 0) or 0)
        for name in _collection_names()
    }
    expected_dimension = int(index_status.get("expected_embedding_dimension", 0) or 0)
    if expected_dimension:
        dimension_mismatches = {
            name: dimension
            for name, dimension in dimensions.items()
            if dimension != expected_dimension
        }
    else:
        nonzero_dimensions = {dimension for dimension in dimensions.values() if dimension > 0}
        dimension_mismatches = (
            {}
            if len(nonzero_dimensions) == 1 and all(dimensions.values())
            else dimensions
        )

    actual_sources = sources_by_collection.get("detail_128", set())
    ready = (
        bool(index_status.get("ready"))
        and not missing_by_collection
        and not dimension_mismatches
    )
    return {
        "expected_sources": sorted(expected_sources),
        "actual_sources": sorted(actual_sources),
        "missing_sources": sorted(expected_sources - actual_sources),
        "sources_by_collection": {
            name: sorted(sources) for name, sources in sources_by_collection.items()
        },
        "missing_by_collection": missing_by_collection,
        "expected_embedding_dimension": expected_dimension,
        "collection_dimensions": dimensions,
        "dimension_mismatches": dimension_mismatches,
        "index_status_ready": bool(index_status.get("ready")),
        "ready": ready,
    }


def main() -> None:
    from app.config import get_settings

    root = _project_root()
    settings = get_settings()
    chroma_dir = resolve_chroma_dir(root, Path(settings.CHROMA_PERSIST_DIR))
    client = chromadb.PersistentClient(path=str(chroma_dir))

    report = {}
    for name in _collection_names():
        try:
            collection = client.get_collection(name)
            report[name] = _audit_collection(collection)
        except Exception as exc:
            report[name] = {
                "count": 0,
                "embedding_dimension": 0,
                "sources": {},
                "page_ranges": {},
                "reference_like_documents": 0,
                "picture_noise_documents": 0,
                "error": str(exc),
            }

    expected_sources = set(_expected_sources(root))
    index_status = _load_index_status(chroma_dir)
    report["index_status"] = index_status
    report["completeness"] = build_completeness(
        expected_sources,
        report,
        index_status,
    )

    output_path = report_path(root, "index_audit_report", chroma_dir)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n索引审计报告已写入: {output_path}")


if __name__ == "__main__":
    main()
