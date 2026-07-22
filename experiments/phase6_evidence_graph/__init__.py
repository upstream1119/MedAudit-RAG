"""Phase 6 evidence-graph contract utilities."""

from importlib import import_module

from .graph_contract import (
    assert_no_gold_only_content,
    constraint_node_id,
    entity_node_id,
    evidence_node_id,
    normalize_id_text,
    question_node_id,
    source_node_id,
    validate_inference_graph,
    validate_selection_manifest,
)
from .inference_graph_builder import build_inference_graph

_LAZY_BATCH_EXPORTS = {
    "build_runtime_source_registry",
    "run_phase6a_batch",
}


def __getattr__(name):
    if name in _LAZY_BATCH_EXPORTS:
        return getattr(import_module(f"{__name__}.batch_runner"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "assert_no_gold_only_content",
    "build_inference_graph",
    "build_runtime_source_registry",
    "constraint_node_id",
    "entity_node_id",
    "evidence_node_id",
    "normalize_id_text",
    "question_node_id",
    "source_node_id",
    "run_phase6a_batch",
    "validate_inference_graph",
    "validate_selection_manifest",
]
