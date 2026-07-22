# Phase 7 Formal Experiments

本目录用于运行生成侧配对实验。当前仅完成 Phase 7-A1：真实调用执行器、预算门控、缓存复用和离线验证；尚未执行真实模型调用，也没有形成 Graph-enhanced 效果结论。

## 当前输入

- 开发集：`Dev50-v1.0`，只用于 smoke 和失败分析。
- 知识库：`KB-medium-v1`，22 份准入资料。
- 方法：`vector_only_rag` 与 `graph_enhanced_rag`。
- 生成模型：配置中的同一模型；两种方法只改变证据上下文。
- 证据预算：每条回答最多 4 个证据片段。

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
