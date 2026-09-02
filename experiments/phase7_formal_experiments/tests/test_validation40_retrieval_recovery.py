from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "phase7_formal_experiments"
    / "validation40_retrieval_recovery.py"
)


def _load_module():
    assert MODULE_PATH.exists(), "Validation40 retrieval recovery module is missing"
    spec = importlib.util.spec_from_file_location(
        "validation40_retrieval_recovery", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class _Chunk:
    content: str
    granularity: int
    source_file: str
    page_number: int
    distance: float = 0.1
    relevance_score: float = 0.9
    authority_weight: float = 0.9
    final_score: float = 0.81
    chapter_title: str = ""
    block_type: str = "text"


class _RecordingRetriever:
    def __init__(self):
        self.calls: list[dict] = []

    def retrieve(self, query: str, top_k: int, granularity: int | None = None):
        self.calls.append({"query": query, "top_k": top_k, "granularity": granularity})
        return [
            _Chunk(
                content=f"{granularity} 粒度候选 {rank}",
                granularity=granularity or 128,
                source_file=f"source-{rank % 3}.pdf",
                page_number=rank + 1,
                final_score=1.0 - rank / 100,
            )
            for rank in range(top_k)
        ]


def _config() -> dict:
    return {
        "candidate_pool_k": 20,
        "final_evidence_k": 4,
        "granularities": [128, 512, 1024],
        "rrf_constant": 60,
        "neighbor_radius": 0,
        "execute_model_calls": False,
        "query_rewrite_enabled": False,
        "cross_encoder_enabled": False,
        "seed": 20260822,
    }


def test_candidate_pool_depth_is_separate_from_final_evidence_window():
    module = _load_module()
    retriever = _RecordingRetriever()

    result = module.build_recovery_result(
        sample={"sample_id": "VAL-001", "question": "儿童肺炎是否需要复评？"},
        retriever=retriever,
        config=_config(),
    )

    assert retriever.calls == [
        {"query": "儿童肺炎是否需要复评？", "top_k": 20, "granularity": 128},
        {"query": "儿童肺炎是否需要复评？", "top_k": 20, "granularity": 512},
        {"query": "儿童肺炎是否需要复评？", "top_k": 20, "granularity": 1024},
    ]
    assert result["candidate_pool_size"] > 4
    assert len(result["evidence"]) == 4
    assert result["audit"]["external_model_calls"] == 0
    assert result["audit"]["input_tokens"] == 0
    assert result["audit"]["output_tokens"] == 0
    assert result["audit"]["estimated_cost"] == 0


class _SingleSeedRetriever:
    def retrieve(self, query: str, top_k: int, granularity: int | None = None):
        return [
            _Chunk(
                content=f"治疗建议正文 {granularity}",
                granularity=granularity or 512,
                source_file="guide-a.pdf",
                page_number=10,
            )
        ]


class _LeakyPageProvider:
    def get_page(self, source_file: str, page_number: int):
        if page_number == 9:
            return _Chunk(
                content="同一指南前一页正文",
                granularity=1024,
                source_file="guide-a.pdf",
                page_number=9,
            )
        if page_number == 11:
            return _Chunk(
                content="不应进入的其他指南正文",
                granularity=1024,
                source_file="guide-b.pdf",
                page_number=11,
            )
        return None


def test_neighbor_expansion_is_limited_to_same_source_and_radius():
    module = _load_module()
    config = _config()
    config["neighbor_radius"] = 1

    result = module.build_recovery_result(
        sample={"sample_id": "VAL-002", "question": "治疗后如何再次评估？"},
        retriever=_SingleSeedRetriever(),
        page_provider=_LeakyPageProvider(),
        config=config,
    )

    source_pages = {
        (item["source_file"], item["page_number"]) for item in result["evidence"]
    }
    assert ("guide-a.pdf", 9) in source_pages
    assert ("guide-a.pdf", 10) in source_pages
    assert ("guide-b.pdf", 11) not in source_pages
    assert result["audit"]["same_source_neighbor_count"] == 1


def test_rejects_gold_only_fields_before_any_retrieval_call():
    module = _load_module()
    retriever = _RecordingRetriever()

    with pytest.raises(ValueError, match="Gold-only field leakage"):
        module.build_recovery_result(
            sample={
                "sample_id": "VAL-003",
                "question": "是否需要复评？",
                "source_filename": "hidden-gold.pdf",
                "page_number": 26,
                "anchor_text_span": "hidden anchor",
            },
            retriever=retriever,
            config=_config(),
        )

    assert retriever.calls == []


@pytest.mark.parametrize("flag", ["query_rewrite_enabled", "cross_encoder_enabled"])
def test_rejects_unapproved_model_features_before_retrieval(flag: str):
    module = _load_module()
    retriever = _RecordingRetriever()
    config = _config()
    config[flag] = True

    with pytest.raises(ValueError, match="must remain disabled"):
        module.build_recovery_result(
            sample={"sample_id": "VAL-004", "question": "是否需要复评？"},
            retriever=retriever,
            config=config,
        )

    assert retriever.calls == []
