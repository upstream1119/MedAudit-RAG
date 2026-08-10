# Phase 7 Formal Experiments

本目录用于运行生成侧配对实验。当前已完成 Phase 7-A1 执行器验证、Phase 7-A2 双候选模型 3 样本真实 smoke、Phase 7-A2.1 延迟与生成回答 claim-level 审计、Phase 7-A3 Frozen15 离线预检、Phase 7-A3.1 的 24 次缺失调用与 30 条回答审计，以及 Phase 7-A3.2 的通用约束语义校准。Frozen15 仍是开发集，尚未形成 Graph-enhanced 正式效果结论。

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

## Phase 7-A3 Frozen15 离线预检

先生成 Frozen15 的模型输入资产，不调用外部 API：

```powershell
$env:PYTHONPATH='.'
python -m experiments.phase6_evidence_graph.generation_contrast_builder `
  --config experiments/phase6_evidence_graph/configs/phase6b_generation_contrast_frozen15_glm45air_nonthinking_v0_1.json `
  --output-dir revision/phase7/source_runs/phase7_source_glm45air_nonthinking_frozen15_v0_1
```

再运行 Phase 7 非执行预检：

```powershell
python -m experiments.phase7_formal_experiments.run_generation_calls `
  --config experiments/phase7_formal_experiments/configs/phase7_generation_frozen15_glm45air_nonthinking_v0_1.json
```

2026-07-24 离线预检结果：

- 15 条 Frozen15 开发样本形成 30 个唯一调用键，`vector_only_rag` 与 `graph_enhanced_rag` 各 15 个。
- 总计划预算为 24,151 input tokens、9,000 output tokens。
- Smoke3 已完成的 6 个调用直接命中缓存，因此后续只需执行 24 个新调用；新增调用估算为 20,092 input tokens、7,200 output tokens。
- `PMSQA_DEV_003`、`PMSQA_DEV_006`、`PMSQA_DEV_039` 在两种方法下都保持无证据输入，共 6 个失败闭合调用。
- gold-only 字段搜索为 0，离线运行 `failed=0`、`external_model_calls=0`。
- dry-run 汇总中的 4,662/197 input/output tokens 来自 6 个历史缓存的原始用量元数据，不代表本次离线预检产生了新 token 消耗。

## Phase 7-A3.1 Frozen15 真实生成

用户明确确认外部传输风险后，使用以下命令只执行 24 个缺失缓存键：

```powershell
$env:PYTHONPATH='.'
python -m experiments.phase7_formal_experiments.run_generation_calls `
  --config experiments/phase7_formal_experiments/configs/phase7_generation_frozen15_glm45air_nonthinking_v0_1.json `
  --execute `
  --confirm-external-call
```

2026-07-24 真实运行结果：

- 30 个逻辑调用中复用 6 个缓存，新增 24 个调用全部成功，失败和空回答均为 0。
- 24 个新增调用实际 input/output tokens 为 20,483/814，估算费用为 0。
- 24 个新调用均首次尝试成功；延迟中位数为 2,643.48 ms，P95 为 5,643.37 ms。
- 30 条总实际 input tokens 为 25,145，比 25,000 的估算门控高 145；后续正式配置需给 token 估算保留至少 5% 余量。
- 确定性 claim 审计完成 30/30，但当前对齐器对回答长度、否定措辞和 claim 拆分敏感，状态计数不得作为论文 hallucination rate。
- Codex 辅助逐题证据复核草案与限制见 `revision/phase7/phase7a3_1_frozen15_generation_audit_v0_1.md`，仍需研究者最终确认。

真实运行目录：

`experiments/phase7_formal_experiments/runs/phase7_generation_glm45air_nonthinking_frozen15_v0_1_20260724_164806/`

## Phase 7-A3.2 约束语义校准

本步骤不重新调用模型，而是修复开发审计中暴露的通用语义问题，并重新构建 Frozen15 重排和生成输入资产：

```powershell
$env:PYTHONPATH='.'
python -m experiments.phase6_evidence_graph.batch_runner `
  --config experiments/phase6_evidence_graph/configs/phase6b_rerank_frozen15_v0_4.json `
  --output-dir revision/phase6/graph_runs/phase6b_rerank_frozen15_v0_4_20260725_02

python -m experiments.phase6_evidence_graph.generation_contrast_builder `
  --config experiments/phase6_evidence_graph/configs/phase6b_generation_contrast_frozen15_glm45air_nonthinking_v0_2.json `
  --output-dir revision/phase7/source_runs/phase7_source_glm45air_nonthinking_frozen15_v0_2_20260725_02
```

随后对 A3.1 已有 30 条 GLM 输出进行离线复审：

```powershell
python -m experiments.phase7_formal_experiments.audit_generation_claims `
  --generation-run-dir experiments/phase7_formal_experiments/runs/phase7_generation_glm45air_nonthinking_frozen15_v0_1_20260724_164806 `
  --output-dir revision/phase7/claim_audits/phase7a3_2_frozen15_semantic_calibration_v0_1_20260725_02 `
  --evidence-budget 4
```

校准覆盖静脉途径方向层级、联合用药指代、局部立场和复合拒绝。输出级 contradicted 从旧审计的 3 条降为 1 条，但这只说明已知评测器误判减少，不是模型 hallucination rate 或 Graph-enhanced 方法效果。

详细报告：

`revision/phase7/phase7a3_2_constraint_semantic_calibration_v0_1.md`

## Phase 7-B1 Benchmark-v1 证据锚点构建

Phase 7-B1 使用 22 份准入资料独立构建正式 Benchmark 的证据锚点，不能复用 Dev50 的问题、事实簇或证据锚点：

- B1.0：校验 source manifest、PDF、文件大小和 SHA-256。
- B1.1：生成 3497 条 `candidate_unverified` 高召回候选；自动候选不是 gold evidence。
- B1.2a：生成 100 条待作者逐页核验记录，并隔离 85 条与 Dev50 同来源同页的候选。
- B1.2b：已完成 100/100 条作者逐页核验，其中 72 条升级为 `author_verified_anchor`、28 条按范围或可追溯性要求拒绝；85 条 Dev50 同来源同页候选继续作为隔离轨迹保留。

生成或安全重建人工核验队列：

```powershell
$env:PYTHONPATH='.'
python -m experiments.phase7_formal_experiments.benchmark_anchor_review `
  --mode prepare
```

`prepare` 会生成 `anchor_review_queue_v0_1.csv`、空的 `evidence_anchor_pool_v0_1.jsonl`、`source_coverage_matrix_v0_2.csv` 和核验报告。如果现有队列已包含人工核验时间，命令会拒绝覆盖。

作者逐页核对 PDF 并填写队列中的全部人工字段后，才可执行升级校验：

```powershell
$env:PYTHONPATH='.'
python -m experiments.phase7_formal_experiments.benchmark_anchor_review `
  --mode promote
```

升级器默认 fail-closed：人工字段不完整、Dev50 同来源同页重叠、来源标题或哈希不一致、页码越界、文本过短/乱码/无法回溯、支持范围不是 `within_can_support` 时均拒绝升级。B1.2 不调用外部模型或 API。

2026-07-29 B1.2b 收尾结果：185 条审核队列由 72 条已升级锚点、28 条范围拒绝和 85 条 Dev50 重叠隔离记录构成，待核验为 0；锚点 ID 无重复，22 份来源均具有明确 coverage status。

2026-07-29 B1.3 元数据完整性修复：修复 10 条 Batch 07 人工核验记录中的中文编码损坏，收窄 4 条锚点的 `supported_claim_types`，并在晋升器中增加 `rejected_metadata_quality` fail-closed 门禁。修复后仍为 72 条唯一锚点，四个关键人工元数据字段的乱码计数为 0；定向测试 `22 passed`、Phase 7 全量测试 `52 passed`、跨层回归 `216 passed`。下一步进入 B2.1 候选问题构建与 Dev50/事实簇/锚点重叠审计；Benchmark-v1 尚未冻结，也尚未形成 Graph-enhanced 正式效果结论。

## Phase 7-B2.1 Benchmark-v1 候选问题池

B2.1 只负责基于已核验锚点构建候选草案，不执行与 Dev50 或候选池内部的近重复/事实簇泄漏判定。后者属于 B2.2。

```powershell
$env:PYTHONPATH='.;backend'
python experiments/phase7_formal_experiments/benchmark_candidate_builder.py
```

2026-07-29 生成结果：

- 以 72 条 `author_verified_anchor` 确定性生成 144 条候选，每个锚点对应 `direct_support` 和 `scope_boundary` 两个角色。
- 22 份准入资料全部保留 coverage status；其中 `SRC-004`、`SRC-009` 为 `scope_limited`，其余 20 个来源实际产生可用于造题的核验锚点。
- 暂定决策分布：`answer=72`、`review_required=41`、`insufficient_evidence=16`、`boundary_refusal=15`。
- 候选 ID、问题文本均为 144/144 唯一；无乱码、无未核验锚点引用，边界拒答均绑定 `POLICY-SAFETY-001`，证据不足均声明缺失证据类型。
- 模型输入导出仅包含 `sample_id` 和 `question`，不会泄露决策、风险标签、证据来源或 gold metadata。
- 生成过程不调用外部模型：input/output tokens 为 0/0，估算费用为 0。
- 验证：B2.1 定向测试 `12 passed`，Phase 7 全量测试 `64 passed`，后端 + Phase 6 + Phase 7 跨层回归 `228 passed`。

产物：

- `revision/benchmark/benchmark_v1/benchmark_candidates_v0_1.jsonl`
- `revision/benchmark/benchmark_v1/candidate_distribution_v0_1.md`

当前候选状态仍为 `draft_candidate_unverified`。B2.2 完成 Dev50 精确重复、中文 3-gram 近重复、事实簇和证据锚点重叠审计前，不得将其称为冻结 Benchmark-v1。

## Phase 7-B2.2 重复与泄漏审计

B2.2 对 B2.1 的 144 条候选执行确定性重叠审计，不调用外部模型或 API：

```powershell
$env:PYTHONPATH='.;backend'
python experiments/phase7_formal_experiments/benchmark_overlap_audit.py
```

2026-07-29 审计结果：

- 144 条候选问题、候选 ID 均唯一；与 Dev50 的精确问题、复用事实簇、复用证据锚点和中文字符 3-gram 阈值重叠均未触发拒绝或人工复核。
- 144 条候选全部保留为 `overlap_audited_draft`，人工复核队列为 0，拒绝记录为 0。
- 每个已核验锚点的 `direct_support` 与 `scope_boundary` 变体属于同一统计依赖组，因此 144 条表达只对应 72 个 `independence_unit_id`；后续切分不得拆散同组变体。
- 分组分布为 72 个大小为 2 的独立单元；不能把 144 条候选当成 144 个相互独立样本进行统计。
- 审计器、配置和 10 项定向测试已完成；Phase 7 专项回归 `74 passed`，后端 + Phase 6 + Phase 7 跨层回归 `238 passed`。
- 外部模型调用、input/output tokens 与估算费用均为 0。

产物：

- `revision/benchmark/benchmark_v1/overlap_audit_v0_1.json`
- `revision/benchmark/benchmark_v1/overlap_review_queue_v0_1.csv`
- `revision/benchmark/benchmark_v1/benchmark_candidates_v0_2_deduplicated.jsonl`

当前产物仍是通过重叠审计的候选草稿。下一步为 B3 双轮作者标注核验；完成 B4 分组切分和 B5 最终审计前，不得称为冻结 Benchmark-v1。

## Phase 7-B3.1 首轮作者核验队列准备

B3.1 只生成并校验第一轮作者核验队列，不代替作者逐题核验，也不调用外部模型或 API：

```powershell
$env:PYTHONPATH='.;backend'
python experiments/phase7_formal_experiments/benchmark_annotation_validator.py --mode prepare-pass1
```

2026-07-29 准备结果：

- 对 B2.2 保留的 144 条候选执行来源标题、文件哈希、页码、证据 span、重叠状态和安全规则绑定的 fail-closed 校验。
- 生成 144 条第一轮队列记录，对应 72 个统计独立单元；`direct_support=72`、`scope_boundary=72`。
- 暂定决策分布仍为 `answer=72`、`review_required=41`、`insufficient_evidence=16`、`boundary_refusal=15`，这些字段必须由作者逐题确认，不能视为最终 gold 标签。
- B3.1 生成时队列为 `completed=0`、`pending=144`，状态为 `pass1_pending`；当时尚未生成第二轮盲化队列、verified 数据集或分歧 resolution 记录。
- 若作者修改问题文本，必须重新执行 B2.2 重叠审计；来源、页码、span 或安全规则不一致时不得进入完成状态。
- 外部模型调用、input/output tokens 与估算费用均为 0。
- 验证：B3.1 定向测试 `15 passed`，Phase 7 专项回归 `89 passed`，后端 + Phase 6 + Phase 7 跨层回归 `253 passed`。

产物：

- `revision/benchmark/benchmark_v1/annotation_pass1_queue_v0_1.csv`
- `revision/benchmark/benchmark_v1/annotation_pass1_summary_v0_1.json`

该阶段只完成核验工具和队列准备。后续由作者按小批次填写第一轮人工字段并逐批运行完成性校验；同一作者的两轮复核应表述为 `two-pass author verification`，不是独立标注者一致性、专家验证或临床验证。

## Phase 7-B3.2 第一轮作者核验 Batch 01

Batch 01 使用离线、可恢复的批次应用模式，不调用外部模型或 API：

```powershell
$env:PYTHONPATH='.;backend'
python experiments/phase7_formal_experiments/benchmark_annotation_validator.py --mode apply-pass1-batch --batch revision/benchmark/benchmark_v1/annotation_pass1_batches/batch01_decisions_v0_1.json
python experiments/phase7_formal_experiments/benchmark_annotation_validator.py --mode validate-pass1-progress
```

2026-07-29 执行结果：

- 完成 10/144 条第一轮作者核验，`accepted=9`、`revise=1`、`pending=134`，当前可进入第二轮候选 9 条。
- 修改题 `PMSQA-BV1C-b097d2cf93138e86fd81` 用于收窄特殊人群证据边界；其 question 已修改，必须返回 B2.2 重跑重叠审计后才能晋升。
- 批次应用执行原子写入、重复批次保护、部分进度校验和人工字段完整性校验；连续问号或替换字符会触发中文元数据质量 fail-closed 门禁。
- 产物为 `annotation_pass1_batches/batch01_decisions_v0_1.json`、更新后的 `annotation_pass1_queue_v0_1.csv` 和 `annotation_pass1_progress_v0_1.json`。
- 队列 SHA-256 为 `a2b4d608221cde3da936f68d171196bed5ba8afeb26b9a24a88ee70fc177be76`，Batch 01 SHA-256 为 `2429d87753c70da924463204e11d2ae2322f72f7f13c6943f9dee134d4f13ac8`。
- 验证：Phase 7 专项回归 `98 passed`，后端 + Phase 6 + Phase 7 跨层回归 `262 passed`。
- 本批 input/output tokens 为 0/0，估算费用为 0。

该时点仍是第一轮核验的部分进度，不是已完成双轮核验或已冻结 Benchmark-v1。

### 修订题复审、可审计纠错与 Batch 02-04（2026-08-01）

修改 question 后必须生成独立复审报告，再由标注器验证报告与当前记录、父队列和内容哈希一致：

```powershell
$env:PYTHONPATH='.;backend'
python experiments/phase7_formal_experiments/benchmark_overlap_audit.py --mode revision --candidate-id PMSQA-BV1C-662420ff4ec8565be29d --pass1-queue revision/benchmark/benchmark_v1/annotation_pass1_queue_v0_1.csv --parent-audit revision/benchmark/benchmark_v1/overlap_audit_v0_1.json --reaudit-output revision/benchmark/benchmark_v1/annotation_pass1_reaudits/PMSQA-BV1C-662420ff4ec8565be29d_v0_1.json
python experiments/phase7_formal_experiments/benchmark_annotation_validator.py --mode apply-pass1-batch --batch revision/benchmark/benchmark_v1/annotation_pass1_batches/batch02_decisions_v0_1.json
python experiments/phase7_formal_experiments/benchmark_annotation_validator.py --mode apply-pass1-reaudit --reaudit-report revision/benchmark/benchmark_v1/annotation_pass1_reaudits/PMSQA-BV1C-662420ff4ec8565be29d_v0_1.json
python experiments/phase7_formal_experiments/benchmark_annotation_validator.py --mode apply-pass1-correction --correction revision/benchmark/benchmark_v1/annotation_pass1_corrections/batch02_source_title_correction_v0_1.json
python experiments/phase7_formal_experiments/benchmark_annotation_validator.py --mode apply-pass1-batch --batch revision/benchmark/benchmark_v1/annotation_pass1_batches/batch03_decisions_v0_1.json
python experiments/phase7_formal_experiments/benchmark_annotation_validator.py --mode apply-pass1-batch --batch revision/benchmark/benchmark_v1/annotation_pass1_batches/batch04_decisions_v0_1.json
python experiments/phase7_formal_experiments/benchmark_overlap_audit.py --mode revision --candidate-id PMSQA-BV1C-0719dda589c36bf349a2 --pass1-queue revision/benchmark/benchmark_v1/annotation_pass1_queue_v0_1.csv --parent-audit revision/benchmark/benchmark_v1/overlap_audit_v0_1.json --reaudit-output revision/benchmark/benchmark_v1/annotation_pass1_reaudits/PMSQA-BV1C-0719dda589c36bf349a2_v0_1.json
python experiments/phase7_formal_experiments/benchmark_annotation_validator.py --mode apply-pass1-reaudit --reaudit-report revision/benchmark/benchmark_v1/annotation_pass1_reaudits/PMSQA-BV1C-0719dda589c36bf349a2_v0_1.json
python experiments/phase7_formal_experiments/benchmark_annotation_validator.py --mode validate-pass1-progress
```

当前执行结果：

- Batch 01、Batch 02 和 Batch 04 的三条修改题复审结果均为 `clear`，待复审记录为 0。
- Batch 02 结果为 `accepted=5`、`revise=1`、`reject=4`；Batch 03 结果为 `accepted=8`、`reject=2`、`revise=0`。
- Batch 04 结果为 `accepted=7`、`revise=1`、`reject=2`；诊断性临床表现题被排除，修订后的青霉素过敏背景题通过独立重叠复审。
- 元数据纠错只允许精确更新 `pass1_review_reason`，需绑定父队列 SHA-256 与旧值，并将 correction 文件、哈希和前后队列哈希写入进度记录；不得用它改问题、证据或决策。
- 累计为 `completed=40`、`pending=104`、`accepted=29`、`revise=3`、`reject=8`；32 条记录当前可进入第二轮候选。
- 当前队列 SHA-256 为 `188e241959e14337abfe22768ac794f87a247da7598d15a5e85b79d580a8ec69`；Batch 04 SHA-256 为 `640a5e2d69c185621ba71fb85705bf63cd83579c847ef58b536a4788b923a27d`；第 38 题复审报告 SHA-256 为 `7fea80ffedf4213fcbdb810bcdab1a3bc0f273850a080aec9aa8194bb4a9cbdd`。
- 正确设置 `PYTHONPATH=backend` 后，Phase 7 专项回归 `114 passed`。
- 外部模型调用、input/output tokens 与估算费用均为 0。

以上为 Batch 01-04 的历史阶段快照；后续批次继续沿用同一原子应用、重叠复审和 fail-closed 校验机制。

## Phase 7-B3.2 第一轮作者核验完成（2026-08-02）

- Batch 01-15 已覆盖 annotation order 1-144，`completed=144`、`pending=0`，进度状态为 `pass1_completed`。
- 第一轮结果为 `accepted=110`、`revise=5`、`reject=29`，其中 `promotable_to_pass2=115`。
- 决策分布为 `answer=69`、`boundary_refusal=15`、`insufficient_evidence=21`、`review_required=39`。
- 五条修订题均已通过 B2.2 重叠复审，当前 `overlap_reaudit_required=0`；原始批次、纠错和复审记录均保留，不覆盖历史结论。
- 队列 SHA-256 为 `5690caa28e765efad435a0b526590950737e4c06c065de0916eeb45d80391f47`，进度文件 SHA-256 为 `551c5da8d89006a2578f9eaa253aabd06424d94f32b99587fcd3065b37ea2e07`。
- `validate-pass1-progress` 通过；设置 `PYTHONPATH=backend` 后 Phase 7 专项回归为 `114 passed`。
- 第一轮没有调用外部模型或 API，input/output tokens 与估算费用均为 0。
- 当前 115 条可晋升候选低于 120 条冻结目标。下一步先补足至少 5 条合格候选并完成相同审计，再生成第二轮盲化队列；Benchmark-v1 尚未冻结。

## Phase 7-B3.2 范围复核与补充池第一轮完成（2026-08-02）

- 对原 144 条第一轮队列执行只读范围一致性审计，标记 1 条“4 周龄以下婴儿体温测量方法”非用药任务；原始第一轮队列和历史结论未被覆盖。
- 从已核验锚点构建 24 条补充候选，决策分布为 `review_required=14`、`insufficient_evidence=5`、`boundary_refusal=5`；来源、页码、锚点、Dev50 重叠、内部重叠和父审计均通过校验。
- 补充池第一轮作者核验为 `completed=24`、`accepted=24`、`pending=0`、`promotable_to_pass2=24`，状态为 `pass1_completed`。
- 排除 1 条范围标记后，原池有 114 条可晋升记录；与补充池合并后共有 138 条第二轮候选，能够覆盖 120 题目标及四类决策配额。
- 补充队列 SHA-256 为 `6d7e3112b5b3a52ab19efc518a48cb2ab2b9057c29e5ba7940ebb743cc349dbb`，补充批次 SHA-256 为 `66a4a22ad30d3e9e3c1722baa797658d5b114671a82f0d339356728ba5421c29`。
- 本步骤没有调用外部模型或 API，input/output tokens 与估算费用均为 0。下一步生成隐藏第一轮结论的第二轮盲化队列；Benchmark-v1 尚未冻结。

## Phase 7-B3.3 第二轮盲化队列就绪（2026-08-02）

```powershell
$env:PYTHONPATH='.;backend'
python experiments/phase7_formal_experiments/benchmark_pass2_queue.py --config experiments/phase7_formal_experiments/configs/benchmark_pass2_queue_v0_1.json
```

- 合并原池 114 条与补充池 24 条，生成 138 条确定性随机顺序的第二轮作者核验队列，对应 59 个 `independence_unit_id`。
- 审阅队列隐藏原候选 ID、`pass1_*`、`provisional_*`、第一轮理由和第一轮支持状态；source/page/span 等核验证据字段继续保留。
- 单独的 linkage manifest 保存 blind ID 与原记录、第一轮结论、独立单元和来源行哈希的映射；第二轮审阅时不应查看该文件。
- 当前第一轮决策池为 `answer=54`、`review_required=43`、`insufficient_evidence=25`、`boundary_refusal=16`，可满足最终 `40/40/24/16` 目标；这只是可行性门禁，不是最终入选结果。
- review queue SHA-256 为 `da529c4d79e94ea3334516009328c8d2850b64e11d1fcf08cbccc22e133f8393`，linkage SHA-256 为 `f05c77d953c07b904c3b9a1e6fa3c1b3fd755b5030ca25dbcd4349d544438b40`。
- 真实输出审计为 138 个唯一 blind ID、138 个唯一 linkage ID、一一映射、0 个泄露列、0 个空问题、0 个空证据片段；Phase 7 专项回归 `127 passed`。
- 本步骤未调用外部模型或 API，input/output tokens 与估算费用均为 0。生成当时第二轮尚未执行，Benchmark-v1 尚未冻结。

## Phase 7-B3.4 第二轮作者核验 Batch 01（2026-08-02）

- 新增第二轮批次校验器与配置，强制执行 blind ID 连续顺序、父队列 SHA-256、版本、字段、决策/支持状态映射、安全政策绑定、乱码和零调用门禁。
- 在不打开 linkage manifest、不查看第一轮结论的条件下，完成 `pass2_order=1-10` 的第二轮作者核验；结果为 `accepted=10`、`pending=128`。
- 当前第二轮决策分布为 `answer=3`、`review_required=1`、`insufficient_evidence=4`、`boundary_refusal=2`。10 条均保持原问题文本，没有修订题或静默晋升。
- 写后队列 SHA-256 为 `20913120a6f9a93efd88cd29a50cbadd3a4c9107c2270fb2e1df30b699fe72e5`；真实输出审计为 138 行、138 个唯一 blind ID、0 个乱码字段、0 个列表解析错误。
- Phase 7 专项回归为 `131 passed`。本批未调用外部模型或 API，input/output tokens 与估算费用均为 0。
- 当前只完成同一作者第二轮的 10/138 条，不构成双轮核验完成、独立专家验证、临床验证或冻结 Benchmark-v1。

## Phase 7-B3.4 第二轮作者核验 Batch 02（2026-08-02）

- 完成 `pass2_order=11-20` 的第二轮盲化作者核验；Batch 02 为 `accepted=9`、`reject=1`，累计 `completed=20`、`pending=118`、`promotable=19`。
- 第 15 条虽有可定位的肾脏专科监测证据，但不属于儿科用药、给药风险或处方边界任务，保留为 `reject` 并记录范围排除理由。
- 累计第二轮决策分布为 `answer=6`、`review_required=4`、`insufficient_evidence=7`、`boundary_refusal=3`；没有修改题。
- 写后队列 SHA-256 为 `19448a0f691ea8a03c3f843dcafff6707cb5ee8a872ea3cc143fe88b00c3ab39`，Batch 02 SHA-256 为 `d2e283059c877922371b54bc725b69f0bb1606e159fac55f39b9956c66319e78`。
- 真实产物审计确认 138 个唯一 blind ID、0 个乱码、0 个列表解析错误；Phase 7 专项回归 `131 passed`。
- 本批未打开 linkage manifest，未调用外部模型或 API，input/output tokens 与估算费用均为 0；Benchmark-v1 尚未冻结。

## Phase 7-B3.4 第二轮作者核验 Batch 03（2026-08-02）

- 在 reviewer-visible 队列上完成 `pass2_order=21-30` 的盲化作者核验，全程未打开 linkage manifest 或读取第一轮结论。
- Batch 03 共 10 条，均为 `accepted` 且保持原问题文本；累计 `completed=30`、`pending=108`、`promotable=29`、`revision_required=0`。
- 当前第二轮决策分布为 `answer=11`、`review_required=6`、`insufficient_evidence=10`、`boundary_refusal=3`。
- 本批继续区分监测原则、检查期预防方案、给药途径条件和个体处方证据；一般人群或国外来源仅在问题主动要求解释适用边界时形成窄范围回答。
- 父队列 SHA-256 为 `19448a0f691ea8a03c3f843dcafff6707cb5ee8a872ea3cc143fe88b00c3ab39`，写后队列 SHA-256 为 `214cee4fb14c75c8c31d7eb2efbdac39dc3f380c8d725ca5ede74a4e8af4cc34`，Batch 03 SHA-256 为 `07d1905f35108b745f76add043e1ce6ff11afa37b659209d38c6d3f8a1cb50dd`。
- 真实产物审计通过，第二轮标注器定向测试 `4 passed`，Phase 7 专项回归 `131 passed`；本批未调用外部模型或 API，input/output tokens 与估算费用均为 0。Benchmark-v1 尚未冻结，下一步继续 Batch 04。

## Phase 7-B3.4 第二轮作者核验 Batch 04（2026-08-05）

- 在 reviewer-visible 队列上完成 `pass2_order=31-40` 的盲化作者核验，全程未打开 linkage manifest 或读取第一轮结论。
- Batch 04 共 10 条，均为 `accepted` 且保持原问题文本；本批决策为 `answer=5`、`insufficient_evidence=3`、`review_required=1`、`boundary_refusal=1`。
- 累计进度为 `completed=40`、`pending=98`、`promotable=39`、`revision_required=0`，累计 outcome 为 `accepted=39`、`reject=1`。
- 累计决策分布为 `answer=16`、`review_required=7`、`insufficient_evidence=13`、`boundary_refusal=4`。
- 父队列 SHA-256 为 `214cee4fb14c75c8c31d7eb2efbdac39dc3f380c8d725ca5ede74a4e8af4cc34`，写后队列 SHA-256 为 `5a34d351d7e26530b6a559b0dcbfe3a1cfc42ea7723f3bdcc21757141b73a529`，Batch 04 SHA-256 为 `bee3e0a0b6b4c878407a3026776dbbef5e5bee323623028d1d1c07eadfb892cb`。
- 第二轮标注器定向测试 `4 passed`，正确设置 `PYTHONPATH=backend` 后 Phase 7 专项回归 `131 passed`；外部模型调用、input/output tokens 与估算费用均为 0。Benchmark-v1 尚未冻结，下一步继续 Batch 05。

## Phase 7-B3.4 第二轮作者核验 Batch 05（2026-08-05）

- 在 reviewer-visible 队列上完成 `pass2_order=41-50` 的盲化作者核验，全程未打开 linkage manifest 或读取第一轮结论。
- Batch 05 共 10 条，均为 `accepted` 且保持原问题文本；本批决策为 `answer=4`、`insufficient_evidence=5`、`review_required=1`、`boundary_refusal=0`。
- 累计进度为 `completed=50`、`pending=88`、`promotable=49`、`revision_required=0`，累计 outcome 为 `accepted=49`、`reject=1`。
- 累计决策分布为 `answer=20`、`review_required=8`、`insufficient_evidence=18`、`boundary_refusal=4`。
- 10 条 source/page/span 锚点均与原始 PDF 复核一致，预期页均为最佳匹配页；回答范围继续限制在已读取证据内。
- 父队列 SHA-256 为 `5a34d351d7e26530b6a559b0dcbfe3a1cfc42ea7723f3bdcc21757141b73a529`，写后队列 SHA-256 为 `3cd4c38f39328c96aedec9352fd0eef1ccb3296af08afdbd4dcda3b8fb5ba851`，Batch 05 SHA-256 为 `dfe2f7879b3ab941ff00550b27f6b631cc0bc0508f36bb74575d94e6375a177d`。
- 第二轮标注器定向测试 `4 passed`，Phase 7 专项回归 `131 passed`；外部模型调用、input/output tokens 与估算费用均为 0。Benchmark-v1 尚未冻结，下一步继续 Batch 06。

## Phase 7-B3.4 第二轮作者核验 Batch 06（2026-08-08）

- 在 reviewer-visible 队列上完成 `pass2_order=51-60` 的盲化作者核验，全程未打开 linkage manifest 或读取第一轮结论。
- Batch 06 共 10 条，均为 `accepted` 且保持原问题文本；本批决策为 `answer=2`、`insufficient_evidence=8`、`review_required=0`、`boundary_refusal=0`。
- 累计进度为 `completed=60`、`pending=78`、`promotable=59`、`revision_required=0`，累计 outcome 为 `accepted=59`、`reject=1`。
- 累计决策分布为 `answer=22`、`review_required=8`、`insufficient_evidence=26`、`boundary_refusal=4`。
- 10 条 source/page/span 锚点均与原始 PDF 精确匹配，文件 SHA-256 一致且预期页均为最佳匹配页；本批重点阻止把国外路径、疗程 rationale、复评原则、混合病原范围和培养复核原则外推为儿童个体处方。
- 父队列 SHA-256 为 `3cd4c38f39328c96aedec9352fd0eef1ccb3296af08afdbd4dcda3b8fb5ba851`，写后队列 SHA-256 为 `31f684e800424d808aa6c27f5e4ec47391036ad8b36a42eb196d16cdf3beb0c3`，Batch 06 SHA-256 为 `b8078b65f39ae9578fab231712ed7bd49d7aedcba5fe2e4ee5cd7ffd155d040c`。
- 第二轮标注器定向测试 `4 passed`，Phase 7 专项回归 `131 passed`；外部模型调用、input/output tokens 与估算费用均为 0。Benchmark-v1 尚未冻结，下一步继续 Batch 07。

## Phase 7-B3.4 第二轮作者核验 Batch 07（2026-08-08）

- 在 reviewer-visible 队列上完成 `pass2_order=61-70` 的盲化作者核验，全程未打开 linkage manifest 或读取第一轮结论。
- Batch 07 共 10 条，均为 `accepted` 且保持原问题文本；本批决策为 `answer=3`、`insufficient_evidence=4`、`review_required=3`、`boundary_refusal=0`。
- 累计进度为 `completed=70`、`pending=68`、`promotable=69`、`revision_required=0`，累计 outcome 为 `accepted=69`、`reject=1`。
- 累计决策分布为 `answer=25`、`review_required=11`、`insufficient_evidence=30`、`boundary_refusal=4`。
- 10 份 PDF 的文件 SHA-256 均与队列一致；8 条锚点在标注页逐字命中，第 66、67 条因项目符号、换行和加粗标记未逐字命中，但标注页词项覆盖率均为 100%，且仍是全文最佳匹配页。
- 本批继续区分给药途径与剂量、禁忌提醒与替代药剂量、结构化记录与确诊、药敏调整原则与具体联合方案，以及国外指南与中国处方标准。
- 父队列 SHA-256 为 `31f684e800424d808aa6c27f5e4ec47391036ad8b36a42eb196d16cdf3beb0c3`，写后队列 SHA-256 为 `fe729dc38c0a101dea5bbc2313107f435734a77be600617fcec72fb272f247bb`，Batch 07 SHA-256 为 `e5c47a756aad6e3ff67b53eadbc30a69400416b5f9e27e883ba263dc7066c749`。
- 批次生成首次触发中文乱码门禁，标注器在队列写回前 fail closed；改用 UTF-8 补丁重建批次后预检和写后审计通过，正式队列未受失败尝试污染。
- 第二轮标注器定向测试 `4 passed`，Phase 7 专项回归 `131 passed`；唯一警告为 Windows 权限阻止 pytest 写入 `.pytest_cache`，不影响测试结果。
- 本批未调用外部模型或 API，input/output tokens 与估算费用均为 0。Benchmark-v1 尚未冻结，下一步继续 Batch 08。

## Phase 7-B3.4 第二轮作者核验 Batch 08（2026-08-09）

- 在 reviewer-visible 队列上完成 `pass2_order=71-80` 的盲化作者核验，全程未打开 linkage manifest 或读取第一轮结论。
- Batch 08 共 10 条，均为 `accepted` 且保持原问题文本；本批决策为 `answer=6`、`insufficient_evidence=3`、`review_required=1`、`boundary_refusal=0`。
- 累计进度为 `completed=80`、`pending=58`、`promotable=79`、`revision_required=0`，累计 outcome 为 `accepted=79`、`reject=1`。
- 累计决策分布为 `answer=31`、`review_required=12`、`insufficient_evidence=33`、`boundary_refusal=4`。
- 7 份来源 PDF 的文件 SHA-256 均与队列一致；7 条锚点逐字命中，第 75、77、80 条因项目符号、换行和空白差异未逐字命中，但标注页关键语义完整且为全文最佳匹配页。
- 本批重点限制超说明书用药背景、定性剂量、特殊人群剂量行、普通人群过敏背景和国外指南方案的外推；只有来源明确支持的窄范围结论可标为 `answer`。
- 父队列 SHA-256 为 `fe729dc38c0a101dea5bbc2313107f435734a77be600617fcec72fb272f247bb`，写后队列 SHA-256 为 `ba0e7bb04d30b6fde5270abb2053e77f656fca623376fa81244f049ca9b81a2e`，Batch 08 SHA-256 为 `f2a3b44b3995cfe17d87cdd29ff8034f998ed2e159832ad5ea865aebb7834f73`。
- 本批未调用外部模型或 API，input/output tokens 与估算费用均为 0。Benchmark-v1 尚未冻结，下一步继续 Batch 09。

## Phase 7-B3.4 第二轮作者核验 Batch 09（2026-08-09）

- 在 reviewer-visible 队列上完成 `pass2_order=81-90` 的盲化作者核验，全程未打开 linkage manifest 或读取第一轮结论。
- Batch 09 共 10 条，均为 `accepted` 且保持原问题文本；本批决策为 `answer=6`、`insufficient_evidence=3`、`review_required=0`、`boundary_refusal=1`。
- 累计进度为 `completed=90`、`pending=48`、`promotable=89`、`revision_required=0`，累计 outcome 为 `accepted=89`、`reject=1`。
- 累计决策分布为 `answer=37`、`review_required=12`、`insufficient_evidence=36`、`boundary_refusal=5`。
- 10 份来源 PDF 的文件 SHA-256 均与队列一致，10 条 source/page/span 锚点均在标注页逐字命中。
- 本批重点区分一般不推荐与绝对禁忌、过敏误标背景与诊断操作、给药途径与剂量处方，以及分层治疗原则与无患者信息的个体化处方。
- 父队列 SHA-256 为 `ba0e7bb04d30b6fde5270abb2053e77f656fca623376fa81244f049ca9b81a2e`，写后队列 SHA-256 为 `0ea5226fb23a403f31d78a31269727510f793703e44a4547024a93fb72bc9a1e`，Batch 09 SHA-256 为 `6f0b2890e59e8872ebcfd8270b1f4a710975a3294dc7a10eac7363ffe60c28bc`。
- 本批未调用外部模型或 API，input/output tokens 与估算费用均为 0。Benchmark-v1 尚未冻结，下一步继续 Batch 10。

## Phase 7-B3.4 第二轮作者核验 Batch 10-14 收口（2026-08-09）

- 在 reviewer-visible 队列上完成 `pass2_order=91-138` 的剩余 48 条核验，全程未打开 linkage manifest 或读取第一轮结论。
- 最终进度为 `completed=138`、`pending=0`、`promotable=137`、`revision_required=0`；outcome 为 `accepted=137`、`reject=1`。
- 第二轮最终决策分布为 `answer=56`、`review_required=12`、`insufficient_evidence=56`、`boundary_refusal=14`。
- Batch 10-14 的 SHA-256 分别为 `b24a9156866505a8780cc6ca49f2136207112bb2e8ec18624e574d1147bfb5e9`、`e8793abe0b86c4241317b9b9afc65d13ff80ec222322309623818ba97298199c`、`90d25b62001980e05c0000fc29260b36a67de2f7fedc94491646acfe6d9aa69f`、`5b7d61441ce7521f07ae8cd0967e822474ed5882cdf2cd7cef3ffdcac9cc07f0`、`da8cca8fe3625fc94d14491be838ccca840eb8a49dfbf4a66f5232c4b902ce8e`。
- 最终 reviewer queue SHA-256 为 `95bcacbb07a93fce5e11d04d77d86ca7e9d52d4bc5f55109da7ec7265a1224ad`；14 个批次覆盖 138 个唯一 blind ID，进度文件与实际队列哈希一致。
- 全量证据审计确认 22 份正式来源可用、文件哈希和页码有效；133 条规范化精确命中，5 条排版回退项的词项覆盖率为 95.0%-97.7%，对应页均为全文最佳匹配页。
- 结构审计为 0 个乱码、0 个问题漂移、0 个非法边界标签、0 个 support-map 不一致；Phase 7 全量回归 `131 passed`。
- B3.4 收口时仅完成同一作者的第二轮核验，不是独立专家或临床验证；后续 linkage 与裁决队列状态见下一节，Benchmark-v1 仍未冻结。

## Phase 7-B3.5 两轮关联与裁决队列（2026-08-09）

- 新增 `benchmark_pass2_resolution.py` 和 `configs/benchmark_pass2_resolution_v0_1.json`，只在 Pass 2 状态完整、输入哈希匹配且 blind ID 一一关联时生成裁决队列。
- 138 条候选关联结果为：103 条可晋升核心一致、34 条可晋升核心分歧、1 条范围排除。34 条分歧来自原池 29 条、补充池 5 条。
- 主要转移为 `review_required->insufficient_evidence=30`；其余核心分歧为 `boundary_refusal->insufficient_evidence=2`、`review_required->answer=1`、`insufficient_evidence->answer=1`。
- `annotation_resolution_queue_v0_1.csv` 保留两轮原始判断、理由、证据定位和独立单元信息，所有最终裁决字段为空；脚本不会自动选择任一轮结论。
- 可晋升样本第二轮原始分布为 `answer=55`、`review_required=12`、`insufficient_evidence=56`、`boundary_refusal=14`。与目标 `40/40/24/16` 不匹配，必须在真实裁决后补充候选或记录偏差，不能强改标签。
- 裁决队列 SHA-256 为 `645ed72b93006379bdeff39c2c344a1469a985082dd405fa99dcd7c5a4322284`，摘要 SHA-256 为 `d7d46a6cf34df8f9a3c2b512a49c77d9f13bb13068965b6a062d5e688c46d92b`；重复生成字节一致。
- 定向测试 `5 passed`，Phase 7 专项回归 `136 passed`。本步骤外部调用、input/output tokens 和费用均为 0。
- 下一步逐条裁决 34 条分歧；裁决完成前不得选择 120 题、冻结 Validation/Test 或启动正式效果比较。

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

## Phase 7-B3.5b 作者确认工作流（2026-08-09）

生成作者审核包：

```powershell
$env:PYTHONPATH='backend'
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\benchmark_resolution_confirmation.py --mode prepare --config experiments\phase7_formal_experiments\configs\benchmark_resolution_confirmation_v0_1.json
```

作者逐条填写 `annotation_resolution_author_review_v0_1.csv` 的全部 `author_*` 字段后，才允许执行：

```powershell
$env:PYTHONPATH='backend'
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\benchmark_resolution_confirmation.py --mode apply --config experiments\phase7_formal_experiments\configs\benchmark_resolution_confirmation_v0_1.json
```

- `prepare` 固定校验正式裁决队列与 AI 草稿哈希，按高置信 20 条、中置信 14 条生成审核 CSV 和中文指南；全部作者字段保持为空。
- `apply` 要求 34 条均有明确审核者、作者角色、确认状态、最终决策、KB 支持、证据状态、理由和时间。任一条缺失、枚举非法、不可变字段漂移或源哈希变化都会 fail closed。
- 正式输入 `annotation_resolution_queue_v0_1.csv` 永不覆盖；完整确认后只写出新的 `annotation_resolution_resolved_v0_1.csv` 和摘要。
- `34/34` 条作者确认已完成并通过 fail-closed 校验。作者审核 CSV SHA-256 为 `03da5dd3e38fb0ce3b8318d868e2549d2f3937842f0e6052273683138fe59fa2`。
- 已生成 `annotation_resolution_resolved_v0_1.csv` 和摘要；resolved CSV SHA-256 为 `bf1f6506511c4853a7a76a5787af785ddf95b7766f23015b8dd044b53ab7a31d`，摘要 SHA-256 为 `5a64d09c6d08e85b82df0d81e09297787cabb383d5429ca71369faca70a69388`。
- 裁决后的 138 条候选分布为：`answer=88`、`review_required=12`、`insufficient_evidence=24`、`boundary_refusal=14`。这仍是候选池状态，不等同于冻结 benchmark。
- 定向测试 `6 passed`，当时 Phase 7 全量回归为 `142 passed`。本步骤没有实验外部模型/API 调用，实验 token 与费用均为 0。

## Phase 7-B3.6a 第二批独立证据锚点候选队列（2026-08-09）

生成候选队列：

```powershell
$env:PYTHONPATH='.;backend'
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\benchmark_anchor_expansion.py
```

- 输入为 `KB-medium-v1` 的 3497 条候选块；排除旧审核队列、现有 evidence anchors 与 Dev50 使用过的来源页，共排除 113 个来源页、738 条候选块。
- 在预选阶段过滤参考文献、目录/标题噪声和缺少儿科相关性的片段，共过滤 1884 条；剩余 875 条满足候选准入条件。
- 最终生成 58 条互不重复的 `source_id + page_number` 候选，覆盖 20 个可产出候选的正式来源。19 个来源各 3 条，`SRC-014` 只有 1 条满足质量约束。
- 配置目标为 60 条，实际短缺 2 条。程序在摘要中显式记录 `target_met=false`、`target_shortfall=2`；没有为凑数而放宽医学质量过滤。
- 审计结果：与旧队列/现有锚点/Dev50 的来源页重叠为 0，参考文献标题噪声为 0，58/58 条具备儿科相关性，所有作者审核字段保持为空。
- 队列 SHA-256 为 `423198aefbb6110fb988f14317493c4c7a55802e1f50c7231947e26224bd9b1b`，确定性重跑哈希一致。
- 输出文件为 `anchor_expansion_review_queue_v0_2.csv`、`anchor_expansion_summary_v0_2.json` 和 `anchor_expansion_review_guide_v0_2.md`。候选队列不是 gold anchor，下一步必须逐条进行作者核验与晋升。
- 本步骤未调用外部模型/API，`input_tokens=0`、`output_tokens=0`、`estimated_cost=0`。

## Phase 7-B3.6b 第二批锚点 AI 辅助核验草稿（2026-08-10）

准备 6 个核验批次：

```powershell
$env:PYTHONPATH='.;backend'
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\benchmark_anchor_expansion_review.py --mode prepare
```

填写独立草稿文件后执行 fail-closed 校验：

```powershell
$env:PYTHONPATH='.;backend'
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\benchmark_anchor_expansion_review.py --mode validate
```

- 父队列固定为 58 条，SHA-256 为 `423198aefbb6110fb988f14317493c4c7a55802e1f50c7231947e26224bd9b1b`；批次规模为 `10/10/10/10/10/8`。
- 最终 AI 辅助草稿分布为 `accepted_draft=27`、`revision_required=25`、`rejected_draft=6`，接受草稿覆盖 15 个来源。
- 草稿 SHA-256 为 `62826dfa7f01c2ec039ad659a743ce3868e1a0a43b980ee1d9226d2ad3924a3a`；作者确认计数为 0。
- `matched_topics` 只用于候选检索解释，不能直接作为证据支持的 claim type。核验脚本要求接受记录具有至少 40 个字符的可核对证据跨度，并拒绝作者字段预填、父队列漂移和非法枚举。
- 输出为 `anchor_expansion_assistant_draft_v0_2.csv`、`anchor_expansion_review_audit_v0_2.json` 和 `anchor_expansion_review_summary_v0_2.md`。
- 定向测试为 `6 passed`；标准 `PYTHONPATH=backend` 下后端/Phase 6/Phase 7 完整回归为 `320 passed`。
- 本步骤未调用外部模型/API，`external_api_calls=0`、`input_tokens=0`、`output_tokens=0`、`estimated_cost=0`。
- 这些输出只是 AI 辅助核验草稿，不是作者确认、Gold evidence、独立专家验证、临床验证或 Graph-enhanced 方法效果证据。
