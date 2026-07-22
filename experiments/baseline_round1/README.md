# Phase 5 Baseline Round 1

This directory contains the reproducible entry point for the first small-scale
baseline dry run.

Phase 5 is not a full paper experiment. Its purpose is to verify the experiment
plumbing before spending model tokens:

- fixed 10-sample manifest from Dev50-v1.0
- compact evidence context from cached retrieval-smoke outputs
- prompt and call-plan generation
- cache-key generation by `sample_id + method + model + prompt_version`
- token and cost estimate placeholders
- raw-output file structure, even when no model call is executed

## Boundary

The dry run does not call external LLM APIs. It only writes local experiment
artifacts under `experiments/baseline_round1/runs/`.

Dev50 remains a development set. Results from this phase must not be reported as
final paper performance.

## Run

```powershell
python experiments/baseline_round1/run_phase5_dry_run.py
```

Optional:

```powershell
python experiments/baseline_round1/run_phase5_dry_run.py `
  --config experiments/baseline_round1/configs/phase5_dry_run_config.json
```

## Outputs

Each run creates a timestamped directory:

```text
experiments/baseline_round1/runs/phase5_dry_run_<timestamp>/
```

Expected files:

- `run_config_effective.json`
- `sample_manifest_resolved.jsonl`
- `prompts.jsonl`
- `call_plan.jsonl`
- `raw_model_outputs.jsonl`
- `evaluation_metadata.jsonl`
- `token_usage_estimate.csv`
- `summary.md`

`prompts.jsonl`, `call_plan.jsonl`, and `raw_model_outputs.jsonl` are treated as
model-input-facing artifacts. They must not contain gold labels such as
`expected_decision`, `gold_evidence_status`, `risk_labels`, or
`current_kb_support`.

`evaluation_metadata.jsonl` is the separate evaluation-side file for those gold
labels. Keep this split intact before adding real model execution.

## Cost Discipline

Before real model execution is added, the runner must keep these fields in every
call record:

- `dataset_version`
- `kb_version`
- `prompt_version`
- `method_id`
- `model_provider`
- `model_name`
- `cache_key`
- `estimated_input_tokens`
- `estimated_output_tokens`
- `estimated_cost_cny`

Real model execution should be added only after the dry-run artifacts are
reviewed and approved.

## Pilot Model Calls

Phase 5-B uses the latest dry-run artifacts as model inputs and writes a
separate pilot run directory. By default, it does not call any external API:

```powershell
python experiments/baseline_round1/run_phase5_model_calls.py --limit 3
```

To intentionally call the configured model API, pass `--execute`:

```powershell
python experiments/baseline_round1/run_phase5_model_calls.py --limit 1 --execute
```

Use the project conda environment for real API calls:

```powershell
D:\anaconda\envs\verimind_MedAudit_env\python.exe `
  experiments\baseline_round1\run_phase5_model_calls.py --limit 1 --execute
```

The runner reads `DASHSCOPE_API_KEY` from the environment, or from local `.env`
/ `backend/.env` files without printing the key. Do not commit `.env` files.
Use `--no-env-file` for safety tests that must not read local key files.

Expected pilot files:

- `run_config_effective.json`
- `prompts.jsonl`
- `model_call_plan.jsonl`
- `raw_model_outputs.jsonl`
- `evaluation_metadata.jsonl`
- `failed_cases.jsonl`
- `token_usage_actual.csv`
- `summary.md`

`raw_model_outputs.jsonl` preserves raw API responses for reproducibility.
Repeated calls use `experiments/baseline_round1/.cache/model_outputs/` by
`cache_key`, so already completed cases are not called again.

## Pilot Evaluation

After pilot model calls, run a lightweight local evaluation:

```powershell
D:\anaconda\envs\verimind_MedAudit_env\python.exe `
  experiments\baseline_round1\evaluate_phase5_pilot.py `
  --run-dir experiments\baseline_round1\runs\phase5_pilot_20260703_191632
```

You can also evaluate the latest pilot run:

```powershell
D:\anaconda\envs\verimind_MedAudit_env\python.exe `
  experiments\baseline_round1\evaluate_phase5_pilot.py
```

This evaluator does not call external APIs. It only audits saved raw outputs.
Its labels are heuristic risk flags, not final clinical judgments.

Expected evaluation files:

- `pilot_evaluation_rows.csv`
- `pilot_evaluation_summary.json`
- `pilot_evaluation_summary.md`

## Judge Dry Run

After the 10-call pilot evaluation, prepare pairwise LLM-as-a-judge prompts
without calling any external judge API:

```powershell
D:\anaconda\envs\verimind_MedAudit_env\python.exe `
  experiments\baseline_round1\run_phase5_judge_dry_run.py
```

This step reads saved pilot outputs from `experiments/baseline_round1/runs/phase5_pilot_20260703_191632` and cached retrieval
smoke outputs. It writes a separate dry-run directory such as:

```text
experiments/baseline_round1/runs/phase5_judge_dry_run_20260708_113613/
```

Expected judge dry-run files:

- `run_config_effective.json`
- `judge_prompts.jsonl`
- `judge_call_plan.jsonl`
- `judge_pair_metadata.jsonl`
- `raw_judge_outputs.jsonl`
- `skipped_samples.jsonl`
- `judge_token_estimate.csv`
- `summary.md`

Current Phase 5-D dry-run result:

- latest output: `experiments/baseline_round1/runs/phase5_judge_dry_run_20260708_191436/`
- `judge_prompt_version=phase5-judge-v0.2-blind`
- `judge_pairs=9`
- `skipped_samples=1` (`PMSQA_DEV_004`, because only one usable method output exists)
- `external_model_calls=0`
- `estimated_input_tokens=19915`
- `estimated_output_tokens=3150`

Method names are hidden in `judge_prompts.jsonl` as Answer A / Answer B. The
method mapping is preserved only in `judge_pair_metadata.jsonl` for later
analysis. The blind v0.2 prompt also removes `expected_decision`,
`scenario_type`, `risk_labels`, `gold_evidence_status`, and
`current_kb_support` from model-facing judge prompts; these fields remain only
in metadata for offline analysis. Review a few prompts and run a JSON parser
smoke test before enabling any real judge call.


## Judge Output Validation

Before enabling real judge calls, validate the expected JSON schema locally:

```powershell
D:\anaconda\envs\verimind_MedAudit_env\python.exe `
  experiments\baseline_round1\validate_judge_outputs.py --self-test
```

The validator checks:

- valid JSON, including optional markdown code-fence stripping
- `winner` in `A / B / tie`
- integer scores from 1 to 3 for `evidence_support`, `safety_boundary`, and `refusal_correctness`
- boolean fields for `unsupported_claim`, `unsafe_suggestion`, and `under_refusal`
- a non-empty string `rationale`

Dry-run rows with empty outputs are skipped by default. Use `--strict` only when validating real judge outputs that should all contain model responses.

## Judge Call Runner

`run_phase5_judge_calls.py` is the real-call scaffold for the pairwise judge, but it is fail-safe by default and does not call external APIs unless `--execute` is passed.

Non-executing plumbing check:

```powershell
D:\anaconda\envs\verimind_MedAudit_env\python.exe `
  experiments\baseline_round1\run_phase5_judge_calls.py `
  --source-run-dir experiments\baseline_round1\runs\phase5_judge_dry_run_20260708_191436 `
  --limit 3 --no-env-file
```

Expected files:

- `run_config_effective.json`
- `judge_prompts.jsonl`
- `judge_call_plan.jsonl`
- `raw_judge_outputs.jsonl`
- `failed_judge_cases.jsonl`
- `judge_token_usage_actual.csv`
- `summary.md`

Repeated calls use `experiments/baseline_round1/.cache/judge_outputs/` by `judge_cache_key`. Failed cases should be retried with `--retry-failed-from` instead of rerunning the whole batch.

## Phase 5 Closeout Status

Phase 5 is considered complete when the baseline-round1 plumbing is verified:

- generator prompt/call-plan split keeps gold labels out of model inputs
- 10-call pilot model execution has raw outputs and token logs
- pilot lightweight evaluation creates auditable CSV/JSON/Markdown summaries
- blind pairwise judge dry-run creates judge prompts and token estimates
- judge JSON schema validation is available
- judge real-call runner is cache-first and non-executing by default

This closeout does not mean final paper metrics have been produced. Formal experiments remain a later phase with frozen evaluation sets, statistical testing, confidence intervals, and careful cost logging.
