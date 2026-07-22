# MedAudit-RAG

[中文说明](./README.zh-CN.md)

Rule-aware evidence auditing RAG system for safer pediatric medication question answering.

MedAudit-RAG is a research prototype for auditing whether pediatric medication answers are supported by traceable evidence from guidelines, consensus documents, drug labels, catalogs, and other admitted authoritative sources. It is not a clinical diagnosis system, prescription generator, or replacement for licensed clinicians.

The current repository contains a vector RAG + TrustScore baseline and a development-stage graph-enhanced evidence-auditing prototype. Its empirical effects have not yet been established and require formal evaluation on a frozen benchmark.

## Core Capabilities

- Route pediatric medication questions into `DETAIL`, `CONCEPT`, and `CONTEXT` audit intents.
- Retrieve evidence from multi-granularity guideline indexes.
- Generate constrained answers from retrieved evidence only.
- Audit retrieval relevance, answer faithfulness, and source authority.
- Apply a TrustScore gate to choose supported answer, review required, insufficient evidence, or boundary refusal.
- Display audit status, score breakdown, citations, source pages, and evidence snippets in the frontend.

## Medical Safety Boundary

This project is for research, teaching, and evidence-audit method validation only.

It does not provide real clinical diagnosis, individualized prescriptions, or treatment advice. All medical claims must be grounded in retrieved evidence. If evidence is insufficient, mismatched, incomplete, or outside the allowed answer boundary, the system should refuse or request human review.

## Architecture

```text
User Query
    |
    v
Router / Intent Normalizer
    |
    v
Multi-granularity Retriever
    |
    v
Constrained Generator
    |
    v
Evidence Auditor
    |
    v
TrustScore Gate
    |
    +--> answer_supported
    +--> review_required
    +--> insufficient_evidence
    +--> boundary_refusal
```

TrustScore is based on retrieval relevance, answer faithfulness, and source authority:

```text
T = alpha * S_ret + beta * S_faith
TrustScore = T * W_authority
```

## Tech Stack

- Backend: Python, FastAPI
- Workflow orchestration: LangGraph
- Vector database: ChromaDB
- Frontend: React, Ant Design, Vite
- Streaming: Server-Sent Events
- Testing: pytest

## Repository Scope

The repository includes:

- backend API routes for health checks, audit queries, and SSE streaming
- router, retriever, generator, auditor, and TrustScore gate logic
- guideline source admission and manifest tracking scripts
- vector index rebuild and audit scripts
- React frontend for audit interaction and evidence display
- unit tests for parser, retriever behavior, streaming serialization, and TrustScore behavior

The repository does not commit raw guideline PDFs, local ChromaDB indexes, API keys, or personal planning notes.

## Knowledge Base and Source Admission

Formal sources are admitted through a manifest-based pipeline before indexing. The manifest is the source-of-truth for whether a document is approved for the knowledge base.

```text
data/guidelines/source_manifest.json
backend/data/chroma_db/
backend/data/chroma_db/index_status.json
```

The source admission workflow records source type, inclusion status, parsing diagnostics, checksum, and indexing status. Generated index files are reproducible local artifacts and are not committed to Git.

## Quick Start

Install backend dependencies:

```powershell
pip install -r backend/requirements.txt
```

Run backend tests:

```powershell
$env:DEBUG='true'
$env:PYTHONPATH='backend'
python -m pytest backend/tests -q
```

Rebuild the vector index:

```powershell
$env:DEBUG='true'
$env:PYTHONPATH='backend'
python backend/rebuild_index.py
```

Optional local embedding mode is available for offline or privacy-sensitive retrieval experiments:

```powershell
pip install -r backend/requirements-local-embedding.txt
$env:EMBEDDING_PROVIDER='local'
$env:EMBEDDING_MODEL='BAAI/bge-small-zh-v1.5'
$env:CHROMA_PERSIST_DIR='backend/data/chroma_db_local'
python backend/rebuild_index.py
```

Start the backend:

```powershell
$env:PYTHONPATH='backend'
python -m uvicorn app.main:app --reload
```

Start the frontend:

```powershell
Set-Location frontend
npm install
npm run dev
```

A convenience script is also available on Windows:

```powershell
.\start-dev.ps1
```

## API Endpoints

```text
GET  /api/health
POST /api/audit/query
POST /api/audit/query/stream
```

The audit response exposes normalized query, intent, answer/refusal text, TrustScore, score breakdown, retrieved evidence snippets, citation source, page number, and final gate decision.

## Evaluation Direction

The planned evaluation is a guideline-grounded pediatric medication safety QA benchmark. Each sample should include:

- gold evidence source, page, and text span
- expected decision
- allowed answer scope
- forbidden claims
- risk labels
- dataset, prompt, model, and knowledge-base versions

Planned metrics include hallucination rate, unsupported claim rate, unsafe suggestion rate, refusal correctness, claim-evidence alignment, and evidence-source mismatch rate. Any future improvement claims must be backed by raw outputs, audit traces, confidence intervals, and statistical tests.

## Experiment Discipline

All model experiments should be cost-aware, cacheable, and reproducible:

1. Record `input_tokens`, `output_tokens`, and `estimated_cost` for every model or judge call.
2. Persist raw model outputs before post-processing.
3. Avoid rerunning the same `sample_id + method + model + prompt_version` if a valid cached output already exists.
4. Rerun failed cases only instead of full batches when possible.
5. Keep evidence context compact, usually 2-4 snippets.
6. Write `prompt_version`, `dataset_version`, and `kb_version` into every output file.

## Research Status

This repository is an active research prototype. Detailed phase status, task tracking, experiment notes, and findings are maintained under `revision/` rather than in the public README.

## Roadmap

- Improve retrieval reliability and source/page precision.
- Expand and audit the authoritative pediatric medication knowledge base.
- Build and freeze guideline-grounded benchmark splits.
- Compare vanilla LLM, naive RAG, multi-granularity RAG, TrustScore Gate, and graph-enhanced evidence auditing.
- Preserve raw outputs, failure cases, confidence intervals, and statistical tests for manuscript writing.
