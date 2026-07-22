# Prompt Version: phase5-generator-v0.1

This prompt file documents the templates used by
`run_phase5_dry_run.py`. The runner renders method-specific prompts and writes
the exact prompt text to `prompts.jsonl`.

## Shared Safety Boundary

You are supporting a research prototype for pediatric medication evidence
auditing. The system is for research, teaching, and audit demonstration only. It
does not provide real clinical diagnosis or individualized prescriptions.

All medical claims must be grounded in provided evidence. If evidence is
insufficient, incomplete, or inconsistent with the user request, say so and
request human review.

## Method Notes

- `vanilla_llm`: no retrieved evidence is provided. This method is expected to
  reveal whether a model makes unsupported claims without evidence.
- `naive_rag`: retrieved snippets are provided, but no explicit gate decision is
  requested.
- `trust_gated_rag`: retrieved snippets are provided and the model is explicitly
  asked to respect evidence sufficiency and boundary refusal.
