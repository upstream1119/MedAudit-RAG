# Prompt Version: phase6b-generator-v0.1

This prompt is used by `generation_contrast_builder.py` to prepare matched
generation-side contrast assets for Phase 6-B.

## Boundary

The prompt states that MedAudit-RAG is a research, teaching, and method
validation prototype. It does not provide clinical diagnosis or individualized
prescriptions.

## Method Arms

- `vector_only_rag`: uses evidence ordered by the original vector ranking.
- `graph_enhanced_rag`: uses evidence ordered by runtime-constraint graph
  reranking and includes only runtime constraint-audit signals.

## Leakage Rule

Model-facing prompts must not include expected decisions, gold evidence, risk
labels, forbidden claims, or benchmark-only annotations.
