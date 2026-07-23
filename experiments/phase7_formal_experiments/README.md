# Phase 7 Formal Experiments

本目录用于运行生成侧配对实验。当前已完成 Phase 7-A1 执行器验证和 Phase 7-A2 双候选模型 3 样本真实 smoke；12/12 次调用成功，但仍未形成 Graph-enhanced 效果结论。

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

模型输入与评测 metadata 继续物理分离。`expected_decision`、gold evidence、risk labels 和 forbidden claims 不进入模型 prompt。

## 研究边界

Phase 7-A 的 Dev50/Frozen15 运行是工程与方法 smoke，不是正式论文 Test Set。正式效果比较必须等待独立构建并冻结的 Benchmark-v1，且需要配对统计检验和置信区间。
