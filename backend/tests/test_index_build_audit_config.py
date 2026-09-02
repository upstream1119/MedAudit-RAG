from pathlib import Path

import audit_index
import rebuild_index
from app.knowledge.indexer import VectorIndexer


def test_new_index_reports_do_not_overwrite_legacy_reports(tmp_path: Path):
    legacy_dir = tmp_path / "backend" / "data" / "chroma_db"
    bge_m3_dir = tmp_path / "backend" / "data" / "chroma_db_bge_m3"

    assert rebuild_index.report_path(
        tmp_path,
        "index_rebuild_summary",
        legacy_dir,
    ) == tmp_path / "docs" / "index_rebuild_summary.json"
    assert rebuild_index.report_path(
        tmp_path,
        "index_rebuild_summary",
        bge_m3_dir,
    ) == tmp_path / "docs" / "index_rebuild_summary_chroma_db_bge_m3.json"
    assert audit_index.report_path(
        tmp_path,
        "index_audit_report",
        bge_m3_dir,
    ) == tmp_path / "docs" / "index_audit_report_chroma_db_bge_m3.json"


def test_audit_resolves_configured_index_directory(tmp_path: Path):
    relative = Path("backend/data/chroma_db_bge_m3")
    absolute = tmp_path / "external_index"

    assert audit_index.resolve_chroma_dir(tmp_path, relative) == (
        tmp_path / relative
    ).resolve()
    assert audit_index.resolve_chroma_dir(tmp_path, absolute) == absolute.resolve()


def test_indexer_reports_actual_collection_embedding_dimensions():
    class FakeCollection:
        def __init__(self, dimension: int):
            self._dimension = dimension

        def get(self, limit, include):
            assert limit == 1
            assert include == ["embeddings"]
            return {"embeddings": [[0.0] * self._dimension]}

    class FakeChroma:
        def get_collection(self, name):
            dimensions = {
                "detail_128": 1024,
                "concept_512": 1024,
                "context_1024": 1024,
            }
            return FakeCollection(dimensions[name])

    indexer = VectorIndexer.__new__(VectorIndexer)
    indexer._chroma = FakeChroma()

    assert indexer.get_collection_dimensions() == {
        "detail_128": 1024,
        "concept_512": 1024,
        "context_1024": 1024,
    }


def test_index_status_fails_closed_on_embedding_dimension_mismatch():
    pdfs = [Path("guideline.pdf")]
    per_doc_summary = {"guideline.pdf": {"blocks_total": 1}}
    inspections = {"guideline.pdf": {"scan_heavy": False}}

    status = rebuild_index._build_index_status(
        pdfs,
        per_doc_summary,
        inspections,
        embedding_provider="local",
        embedding_model="BAAI/bge-m3",
        embedding_model_path="D:/models/bge-m3",
        collection_dimensions={
            "detail_128": 1024,
            "concept_512": 512,
            "context_1024": 1024,
        },
        expected_embedding_dimension=1024,
    )

    assert status["ready"] is False
    assert status["embedding_model_path"] == "D:/models/bge-m3"
    assert status["dimension_mismatches"] == {"concept_512": 512}


def test_audit_collection_reports_actual_embedding_dimension():
    class FakeCollection:
        def get(self, limit=None, include=None):
            if include == ["embeddings"]:
                assert limit == 1
                return {"embeddings": [[0.0] * 1024]}
            assert include == ["metadatas", "documents"]
            return {
                "metadatas": [{"source_file": "guideline.pdf", "page_number": 2}],
                "documents": ["正文证据"],
            }

        def count(self):
            return 1

    report = audit_index._audit_collection(FakeCollection())

    assert report["embedding_dimension"] == 1024


def test_audit_completeness_fails_closed_on_dimension_mismatch():
    report = {
        "detail_128": {"sources": {"guideline.pdf": 1}, "embedding_dimension": 1024},
        "concept_512": {"sources": {"guideline.pdf": 1}, "embedding_dimension": 512},
        "context_1024": {"sources": {"guideline.pdf": 1}, "embedding_dimension": 1024},
    }
    index_status = {"ready": True, "expected_embedding_dimension": 1024}

    completeness = audit_index.build_completeness(
        {"guideline.pdf"},
        report,
        index_status,
    )

    assert completeness["ready"] is False
    assert completeness["dimension_mismatches"] == {"concept_512": 512}
