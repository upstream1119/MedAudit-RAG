# Phase 7 Formal Experiments

本目录用于运行生成侧配对实验。当前已完成 Phase 7-A1 执行器验证、Phase 7-A2 双候选模型 3 样本真实 smoke，以及 Phase 7-A2.1 延迟记录和生成回答 claim-level 证据审计；12/12 次 smoke 调用成功，但仍未形成 Graph-enhanced 效果结论。

## 当前输入

- 开发集：`Dev50-v1.0`，只用于 smoke 和失败分析。
- 知识库：`KB-medium-v1`，22 份准入资料。
- 方法：`vector_only_rag` 与 `graph_enhanced_rag`。
- 生成模型候选：`qwen3.7-plus` 与 `glm-4.5-air`；每个模型内部的两种方法只改变证据上下文。
- 证据预算：每条回答最多 4 个证据片段。
- 推理模式：开发性候选比较统一关闭 thinking，并以 `inference_profile` 隔离缓存身份。

## 安全门控

默认命令只生成执行记录，不调用外部 API：

```powershell
$env:PYTHONPATH='.'
python -m experiments.phase7_formal_experiments.run_generation_calls
```

真实调用必须同时显式提供两个参数：

```powershell
python -m experiments.phase7_formal_experiments.run_generation_calls `
  --execute `
  --confirm-external-call
```

执行真实调用前还必须人工确认：免费额度、模型可用性、调用数、输入/输出 token 上限和输出目录。相同 cache key 已成功运行时直接复用缓存；失败重跑使用 `--retry-failed-from <run_dir>`，不整批重跑。

当前 A2 已完成 3 条开发样本 × 2 种方法 × 2 个候选模型，共 12 次真实调用。Qwen 配置透传 `enable_thinking=false`，GLM 配置透传 `thinking.type=disabled`；厂商字段不得覆盖模型、消息、温度和最大输出等核心请求字段。

真实 smoke 结果：

- `glm-4.5-air`：6/6 成功，input/output tokens 为 4662/197。
- `qwen3.7-plus`：6/6 成功，input/output tokens 为 4898/615。
- 两组均无空输出和 reasoning 泄漏。
- GLM 暂列 Frozen15 主候选，Qwen 作为次级稳健性候选；主模型尚未冻结。
- 详细人工审计见 `revision/phase7/phase7a2_candidate_model_comparison_smoke3_v0_1.md`。

Phase 7-A2.1 新增运行记录字段：

- `latency_ms`：从单条外部模型调用开始到成功返回或最终失败的累计模型阶段延迟，包含该条调用内部的重试等待。
- `latency_source`：区分真实测量、缓存回放、旧缓存无延迟和非执行状态。
- `attempt_count`：该条外部调用的实际尝试次数。

两次独立缓存的单调用探针均成功：GLM 为 `1921.09 ms`，Qwen 为 `2126.38 ms`。每个模型只有 1 个观测，不能据此推断稳定延迟差异或模型速度排名。

对既有生成输出执行确定性 claim-level 证据审计：

```powershell
$env:PYTHONPATH='.'
python -m experiments.phase7_formal_experiments.audit_generation_claims `
  --generation-run-dir <generation_run_dir> `
  --output-dir <new_output_dir> `
  --evidence-budget 4
```

审计器根据生成方法分别恢复 `rank_before` 或 `rank_after` 证据顺序，并检查缓存键集合、父资产不可变、无 gold-only 字段泄漏和两次运行确定性。审计结果用于开发诊断；由于当前 claim 对齐器对回答长度和措辞敏感，状态计数不得直接当作论文 hallucination rate。

## 输出资产

每次运行都会保存：

- `run_config_effective.json`
- `prompts.jsonl`
- `model_call_plan.jsonl`
- `raw_model_outputs.jsonl`
- `evaluation_metadata.jsonl`
- `failed_cases.jsonl`
- `token_usage_actual.csv`
- `summary.md`

其中真实新调用会在 `raw_model_outputs.jsonl`、缓存 JSON 和 `token_usage_actual.csv` 中同时保存延迟与尝试次数。2026-07-22 以前生成的旧缓存没有历史延迟，回放时明确标记为 `unavailable_legacy_cache`，不会补造数据。

模型输入与评测 metadata 继续物理分离。`expected_decision`、gold evidence、risk labels 和 forbidden claims 不进入模型 prompt。

## 研究边界

Phase 7-A 的 Dev50/Frozen15 运行是工程与方法 smoke，不是正式论文 Test Set。正式效果比较必须等待独立构建并冻结的 Benchmark-v1，且需要配对统计检验和置信区间。
