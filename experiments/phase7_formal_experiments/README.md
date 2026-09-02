# Phase 7 Formal Experiments

本目录用于运行冻结 Benchmark、检索配置选择与生成侧配对实验。当前已完成 Benchmark120 的作者双轮核验与冻结、Validation40 输入隔离、Phase 7-C1c-4d 强非图基线选择，以及 `C1c-4e-3a` 的前缀稳定 matched-compute `F24/G1-v0.4` 配对实验。`G1-v0.4` 在 Validation40 开发集上的候选严格来源页召回@24由 `21/40` 提高至 `26/40`，最终严格来源页召回@4由 `20/40` 提高至 `22/40`，但当前不声称统计显著性。下一步先补齐 graph path provenance，再进入 `C1c-4e-3b` G2 图重排；G3 图一致性审计、完整 Graph-enhanced 方法和临床验证结论尚未形成，Pilot Test80 仍未读取。

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

## Phase 7-B3.6c 第二批锚点作者核验门禁（2026-08-10）

生成 6 个作者核验批次：

```powershell
$env:PYTHONPATH='.;backend'
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\benchmark_anchor_expansion_author_review.py --mode prepare
```

逐批填写作者字段后，可先校验单个批次：

```powershell
$env:PYTHONPATH='.;backend'
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\benchmark_anchor_expansion_author_review.py --mode validate-batch --review-batch revision\benchmark\benchmark_v1\anchor_expansion_author_review_batches_v0_2\batch_01.csv
```

只有 58 条均完成显式作者核验后，才允许执行汇总：

```powershell
$env:PYTHONPATH='.;backend'
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\benchmark_anchor_expansion_author_review.py --mode finalize
```

- 配置锁定父队列和 AI 草稿 SHA-256，任一来源、页码、原文或草稿字段漂移都会 fail closed。
- `prepare` 生成 `10/10/10/10/10/8` 六批工作包，所有作者字段保持为空；再次运行不会覆盖已经填写作者结论的批次。
- `accepted` 必须提供可追溯的证据跨度、非空 claim type、适用人群、适用条件和 `within_can_support` 范围判断；`rejected` 必须写明理由。
- 未覆盖全部 58 条、存在重复候选或任一批次字段不完整时，`finalize` 不会生成作者核验汇总。
- 本步骤只完成作者核验门禁，不执行 evidence anchor 晋升；也不得表述为独立专家验证、临床验证或 Graph-enhanced 方法效果。

## Phase 7-B3.6d 第二批证据锚点独立晋升门禁（2026-08-11）

先执行只读校验，不写正式产物：

```powershell
$env:PYTHONPATH='.;backend'
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\benchmark_anchor_expansion_promotion.py --mode validate
```

校验通过后执行原子晋升：

```powershell
$env:PYTHONPATH='.;backend'
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\benchmark_anchor_expansion_promotion.py --mode promote
```

- 配置锁定候选队列、作者核验结果、作者核验审计、来源覆盖矩阵、旧锚点池和 Dev50 注册表的 SHA-256；父资产漂移时 fail closed。
- 晋升时重新检查正式 PDF 的来源、标题、文件名、SHA-256、页码、证据跨度、适用范围、作者核验元数据和 claim type。
- 58 条作者核验记录中仅 46 条 `accepted` 被晋升，12 条 `rejected` 被排除并保留在决策审计中。
- 新增锚点为 46 条，合并锚点池为 118 条，即旧池 72 条加新增 46 条；旧 v0.1 文件及其 SHA-256 保持不变。
- 新增锚点与旧池、Dev50 的 `source_id + page_number` 重叠均为 0；合并池 schema 一致，无重复 ID 或重复来源页。
- 主要产物：`evidence_anchor_expansion_v0_2.jsonl`、`evidence_anchor_pool_v0_2.jsonl`、`anchor_expansion_promotion_decisions_v0_2.csv`、`anchor_expansion_promotion_audit_v0_2.json` 和 `anchor_expansion_promotion_summary_v0_2.md`。
- 关键 SHA-256：新增锚点 `bf5d22d48ebf7f2a182e958deef7d98831821cd1c1cbefc1fbee41d386293c7d`，合并池 `a11b9e552e3b0de126f5a8cfdc20dc508e4355a1a4e8dcb1301ac984a01d8fa5`，决策审计 `462efc1ada3169b442dbe320e6dd3ab161e7bb59c16066c8bb470ba2e0ff9494`。
- 确定性重跑后五个产物的文件哈希一致。定向测试 `8 passed`，Phase 7 回归 `170 passed`，完整回归 `334 passed`。
- 本步骤没有调用外部模型/API，`external_api_calls=0`、`input_tokens=0`、`output_tokens=0`、`estimated_cost=0`。
- 118 条记录是单一作者核验、公开资料可追溯的 evidence anchors，不是冻结 Benchmark-v1 的最终 gold labels、独立专家验证、临床验证或 Graph-enhanced 方法效果证据。

## Phase 7-B3.6e 非 `answer` 补充问题候选构建（2026-08-11）

运行候选构建与独立性审计：

```powershell
$env:PYTHONPATH='.;backend'
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\benchmark_anchor_supplement_candidates.py --config experiments\phase7_formal_experiments\configs\benchmark_anchor_supplement_candidates_v0_2.json
```

- 输入锁定 B3.6d 新增的 46 条独立 evidence anchors，并为每个锚点构建 1 条补充问题候选；候选总数为 46，独立单元数为 46，覆盖 19 个正式来源。
- 临时候选分布为 `review_required=36`、`boundary_refusal=10`，用于补足已核验候选池中非 `answer` 场景不足；这些标签仍是 `provisional_expected_decision`，不是最终 `expected_decision`。
- 10 条处方边界候选均绑定 `POLICY-SAFETY-001`；全部候选保持 `annotation_status=pending_author_review`、`freeze_status=draft`，没有写入最终 `forbidden_claims`。
- 与 Dev50、Frozen15、既有候选、候选内部和既有 `source_id + page_number` 的重叠均为 0；无效页码和未解决审计项均为 0。
- 候选 JSONL SHA-256 为 `4083e6c41efbc9dd9ace72b73a783bde6c930310dec0d8e1a0afc767ec26d678`；作者审核队列为 `2e25f89295cad29f725636ea08d591a9c239e0475fbbc03c0eee47e01381321d`；审计 JSON 为 `663c7f4968bf34e59af63faba59caf760bfc08542b3f2f2c074fb8ea62315ae3`；摘要为 `9ec4a6d36ee07e2fd66a857180eb7fcbdf8bccff971e7f2af34d4bacadb0079c`。
- 确定性重跑后四个产物哈希全部一致。定向测试 `6 passed`、Phase 7 回归 `176 passed`、后端/Phase 6/Phase 7 联合回归 `340 passed`。
- 本步骤未调用外部模型/API，`external_model_calls=0`、`input_tokens=0`、`output_tokens=0`、`estimated_cost=0`。
- 下一步必须逐条完成作者审核，确认问题表述、证据范围、最终决策、风险标签和禁止主张；审核完成前不得合并到最终候选池、选择 120 题或冻结 Benchmark-v1。

## Phase 7-B3.6f 补充候选作者审核门禁（2026-08-11）

生成五批作者审核工作包：

```powershell
$env:PYTHONPATH='.;backend'
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\benchmark_anchor_supplement_author_review.py --mode prepare
```

逐批填写作者字段后校验，例如：

```powershell
$env:PYTHONPATH='.;backend'
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\benchmark_anchor_supplement_author_review.py --mode validate-batch --batch-file revision\benchmark\benchmark_v1\supplement_author_review_batches_v0_2\batch_01.csv
```

仅当五批全部通过校验后才能收口：

```powershell
$env:PYTHONPATH='.;backend'
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\benchmark_anchor_supplement_author_review.py --mode finalize --batch-dir revision\benchmark\benchmark_v1\supplement_author_review_batches_v0_2
```

- 配置固定锁定 B3.6e 的候选 JSONL、审核队列、独立性审计和摘要四个父资产 SHA-256；任一输入漂移都会 fail closed。
- 46 条候选按 `10/10/10/10/6` 拆分为五批，覆盖 46 个唯一候选和 19 个正式来源；工作包包含原问题、证据跨度、临时决策、风险标签、策略规则、来源文件及哈希等审核上下文。
- 作者接受候选时必须同时填写最终问题、最终决策、风险标签、允许回答范围、禁止主张、理由、审核人和时间；`revision_required` 与 `rejected` 不能携带可直接晋升的完整字段。
- 历史门禁快照：工作包初始生成时五批作者字段均为空、审核进度为 `0/46`；空白批次执行 `validate-batch`、五批空白执行 `finalize` 均按预期失败。
- 当前状态：五批已完成 `46/46` 条作者审核并通过 `10/10/10/10/6` 官方校验；B3.6g 已执行 `finalize`，正式结果为 `accepted=44`、`rejected=2`，接收项决策为 `review_required=34`、`boundary_refusal=10`。
- 正式作者审核 CSV 保留全部 46 条审计轨迹；两条 rejected 的可晋升字段均为空。CSV、审计 JSON 和摘要 Markdown 的 SHA-256 分别为 `ce2c7443e258af2ab53aab9bcb27707f494d6d567567d241b1d7aa60933180b0`、`8b2feb4b59ae4b1f2070649edef88b8b6ea6f6226bd425da3e90b6ac5bc20f22`、`7561d1e5066ba55830231d27ccb426e8527ede1b2746f9bab314a172888d0f22`；确定性重跑保持一致。
- 定向测试 `9 passed`、Phase 7 回归 `185 passed`、后端/Phase 6/Phase 7 联合回归 `349 passed`。受限沙箱中的 `_ctypes` DLL 加载失败属于收集权限问题，同一命令脱离受限沙箱后全部通过。
- 本步骤没有调用外部模型/API，`external_model_calls=0`、`input_tokens=0`、`output_tokens=0`、`estimated_cost=0`。
- B3.6g 只完成作者审核结果收口，审计明确记录 `candidate_merge_performed=false`、`gold_promotion_performed=false`、`freeze_performed=false`；不能将 `finalize` 等同于 Gold 晋升或 Benchmark 冻结。

## Phase 7-B3.6h：已审核补充候选池构建与再审计

```powershell
$env:PYTHONPATH='.;backend'
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\benchmark_anchor_supplement_reviewed_pool.py --config experiments\phase7_formal_experiments\configs\benchmark_anchor_supplement_reviewed_pool_v0_2.json
```

- 固定锁定 B3.6g 正式作者审核 CSV、审核审计、审核摘要，以及 B3.6e 原始候选、既有候选池和 Dev50 的 SHA-256；任一父资产漂移都会 fail closed。
- 正式候选池仅纳入 44 条 `accepted`，明确排除 2 条 `rejected`；通过 `candidate_id` 一一关联被锁定的原始候选，恢复事实簇、独立单元和来源页字段。
- 44 条记录对应 44 个唯一候选 ID、44 个唯一事实簇、44 个唯一 `independence_unit_id` 和 44 个唯一来源页；最终决策分布为 `review_required=34`、`boundary_refusal=10`。
- 最终问题与 Dev50、既有候选和本池内部的精确/近重复均为 0；来源页复用、事实簇复用和 rejected 泄漏均为 0，未解决重叠为 0。
- 正式候选 JSONL、审计 JSON 和摘要 Markdown 的 SHA-256 分别为 `09266a42377557fba647079d95f1103706b8407c9008cf09fe07ebc7b71d9c6d`、`b9ae0c94c4124d834693ad98ab145b7cfe870bff209f5e1e9ab6fcb09199c1ee`、`e945e1a741c7ae37b7aa0a89aec52646b6eb793393e7a132397feb184a3e0d83`；确定性重跑保持一致。
- 定向测试 `9 passed`、Phase 7 回归 `194 passed`、后端/Phase 6/Phase 7 联合回归 `358 passed`。受限沙箱中的 `_ctypes` DLL 加载失败属于收集权限问题，同一命令脱离受限沙箱后全部通过。
- 本步骤没有调用外部模型/API，`external_model_calls=0`、`input_tokens=0`、`output_tokens=0`、`estimated_cost=0`。
- 当前仍为 `author_reviewed_candidate` 草稿池：`benchmark_merge_performed=false`、`gold_promotion_performed=false`、`freeze_performed=false`。下一步 B3.6i 才进行 Benchmark-120 选择和按独立单元分组的切分设计。

## Phase 7-B3.6i：Benchmark-120 选择与分层草案（2026-08-14）

运行固定配置的选择与分层工具：

```powershell
$env:PYTHONIOENCODING='utf-8'
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\benchmark120_selection.py --config experiments\phase7_formal_experiments\configs\benchmark120_selection_v0_1.json --repo-root .
```

- 选择器锁定 8 类输入资产 SHA-256，任一父资产漂移都会 fail closed；不调用外部模型或 API。
- 可选池共 146 条：102 条来自旧池两轮一致记录，44 条来自 B3.6h 单轮作者终审池。另隔离 36 条，其中 34 条裁决理由文本损坏、1 条两轮结论不一致、1 条第二轮未接受。
- 确定性选择 120 条，决策分布为 `answer=40`、`review_required=40`、`insufficient_evidence=24`、`boundary_refusal=16`；其中 89 条为两轮一致，31 条为单轮作者终审且待同一作者、与第一轮逐条结论隔离的第二轮盲化复核。
- 分层草案为 Validation 40 / Pilot Test 80。Validation 分布为 `13/13/8/6`，Pilot Test 分布为 `27/27/16/10`；对应两轮一致/单轮待二审分别为 `30/10` 与 `59/21`。
- 跨分层事实簇、证据锚点、独立单元、来源页和中文 3-gram 近重复泄漏均为 0；与 Dev50 的问题重叠为 0。
- 五个正式产物确定性复跑 SHA-256 完全一致。定向测试 `8 passed`、Phase 7 回归 `202 passed`、后端/Phase 6/Phase 7 联合回归 `366 passed`。
- 外部模型调用、input/output tokens 与估算费用均为 0。当前状态为 `draft_selection_ready_for_second_pass`，并明确记录 `gold_promotion_performed=false`、`freeze_performed=false`。

主要产物：

- `revision/benchmark/benchmark_v1/benchmark_selectable_pool_v0_1.jsonl`
- `revision/benchmark/benchmark_v1/benchmark120_selection_draft_v0_1.jsonl`
- `revision/benchmark/benchmark_v1/benchmark120_split_proposal_v0_1.json`
- `revision/benchmark/benchmark_v1/benchmark120_selection_audit_v0_1.json`
- `revision/benchmark/benchmark_v1/benchmark120_selection_summary_v0_1.md`

下一步 B3.6j 只对选中的 31 条单轮样本执行同一作者、与第一轮逐条结论隔离的第二轮盲化复核。第二轮分歧解决、Gold promotion gate 和正式冻结必须继续分步执行；当前 40/80 仅是可审计切分草案。

## Phase 7-B3.6j-a：Benchmark-120 第二轮盲化复核队列准备（2026-08-14）

```powershell
$env:PYTHONIOENCODING='utf-8'
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\benchmark120_second_pass_queue.py --config experiments\phase7_formal_experiments\configs\benchmark120_second_pass_queue_v0_1.json --repo-root .
```

- 锁定 Benchmark-120 选择草案、选择审计和医疗安全策略三项父资产 SHA-256；任一父资产漂移都会 fail closed。
- 从 120 条草案中仅抽取 31 条 `requires_second_pass=true` 记录，形成 Validation 10 / Pilot Test 21 的盲化复核队列，并确定性拆分为 `10/10/10/1` 四个空白批次模板。
- reviewer-visible 队列只暴露问题、来源、页码、证据跨度或安全策略证据，以及空白 Pass 2 字段；`candidate_id`、split、第一轮决策、支持范围、风险标签、禁止主张和其他作者派生字段均不进入可见队列。
- 私有 linkage 仅用于复核完成后的关联与分歧识别；31 个 blind ID 与 31 条第一轮记录一一对应，可见字段泄漏为 0，31 条 Pass 2 复核字段全部为空。
- 复核模式是 `same-author row-level first-pass-label-blinded second review`。它不是独立专家复核、临床验证或独立标注者一致性研究。
- 队列、linkage、指南及四批模板均通过确定性复跑审计。生成审计 SHA-256 为 `10873d75e8f54586a0eca6602814459bc6d279178496db52a0bca3f440c5fdb9`。
- 新增定向测试 `7 passed`；相关选择/队列回归 `21 passed`；Phase 7 全量回归 `209 passed`；后端与 Phase 7 联合回归在脱离受限沙箱后为 `281 passed`。受限沙箱中的 `_ctypes` DLL 加载失败属于环境权限问题，不是代码回归。
- 本步骤没有调用外部模型/API，`external_model_calls=0`、`input_tokens=0`、`output_tokens=0`、`estimated_cost=0`。
- 当前状态为 `second_pass_queue_ready_review_pending`：只完成队列准备，31 条实际复核、两轮关联、真实分歧裁决、Gold 晋升和正式冻结均未执行。

## Phase 7-B3.6j-b：Benchmark-120 第二轮复核 Batch 01（2026-08-15）

- 作者 `PYH` 仅依据 reviewer-visible 工作包完成 Batch 01 的 10 条复核；10 条均为 `accepted`，其中 `answer=9`、`boundary_refusal=1`，原问题均未改写。
- 对“待审回答能否通过”的证据审计题，若当前证据可直接确认错误外推、适用范围错配、关键禁忌遗漏或不受支持的具体化，则允许形成受限的否定审计结论，记为 `answer`；个体化完整处方请求单独按 `POLICY-SAFETY-001` 记为 `boundary_refusal`。
- 标注器原子写回后的进度为 `completed=10`、`pending=21`、`status=in_progress`；队列 SHA-256 为 `acd2fe8a8ea2f3c138f1dfacb0df8c08c63769bdddd16241c0e99634d792054b`。
- 空白 Batch 01 模板 SHA-256 仍为 `52c08318030c1d57a1adc6c383ccfba6e5cb8e9fe62c78fb7034ea893b475f36`，后 21 条 Pass 2 字段保持为空，私有 linkage 未读取。
- 本轮是 AI 辅助草案与同一作者确认，不是独立专家、独立标注者或临床验证；辅助检索过程也不宣称完全盲态。
- 定向测试为 `11 passed`；使用既定 `PYTHONPATH=backend` 后，Phase 7 全量回归为 `209 passed`。
- 本批没有调用外部模型/API，`external_model_calls=0`、`input_tokens=0`、`output_tokens=0`、`estimated_cost=0`。

## Phase 7-B3.6j-b：Benchmark-120 第二轮复核收口（2026-08-16）

- Batch 02、03、04 分别完成 `10/10/1` 条作者复核；四批累计 `31/31`，结果为 `accepted=31`、`answer=29`、`boundary_refusal=2`，所有 accepted 记录均保留原问题。
- 两条 `boundary_refusal` 均绑定版本化安全策略 `POLICY-SAFETY-001`；其余 29 条只允许在已定位 source/page/span 的范围内形成受限审计结论。
- 最终队列状态为 `complete`，SHA-256 为 `9a1845330d852cca24f4b7a178698418e335529474329dffd2fb3410f933fefa`；进度 JSON SHA-256 为 `ad23cbaac031239166754b4a20042dd08d82553c5a7df874912c4868627f458a`。
- 独立结构审计确认 31 个 blind ID 唯一、必填字段和列表字段合法、问题文本未被静默改写、边界策略绑定正确、模板哈希未变化且无乱码。
- 定向测试 `4 passed`，Phase 7 全量回归 `209 passed`。本阶段外部模型/API 调用、input/output tokens 与估算费用均为 0。
- B3.6j-b 仍属于同一作者的第二轮复核，不是独立专家或临床验证。私有 linkage 在本步骤中未读取，Gold promotion 与 Validation/Pilot Test 冻结均未执行；下一步是 B3.6j-c 的两轮关联与真实分歧识别。

## Phase 7-B3.6j-c：Benchmark-120 双轮关联与真实分歧队列（2026-08-16）

```powershell
$env:PYTHONIOENCODING='utf-8'
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\benchmark120_second_pass_resolution.py --config experiments\phase7_formal_experiments\configs\benchmark120_second_pass_resolution_v0_1.json
```

- 配置固定锁定 120 条选择草案、31 条私有 linkage、第二轮最终队列和完成进度的 SHA-256；任一父资产漂移都会 fail closed。
- 关联器重新计算每条 `selection_row_sha256`，并核对 blind ID、candidate ID、问题、来源、页码、版本和第一轮字段，避免直接信任私有映射。
- 10 个比较字段包括 decision、KB support、evidence status、required evidence、required/allowed/forbidden claims、missing evidence/information 和 risk labels；数组字段按顺序无关的规范形式比较，但在产物中保留原始双轮文本。
- 真实结果为 linked 31、完全一致 0、待裁决 31。29 条从 `review_required` 转为 `answer`，2 条维持 `boundary_refusal`；不得自动采用第二轮结果。
- 主要产物为 `benchmark120_second_pass_linked_comparison_v0_1.jsonl`、`benchmark120_second_pass_resolution_queue_v0_1.csv`、`benchmark120_second_pass_resolution_summary_v0_1.json` 和 `benchmark120_second_pass_resolution_guide_v0_1.md`。
- 待裁决队列 SHA-256 为 `5d10786a259b5a71ecb4da85c5092bdfba0c6bc09c469ee14def31213c90487a`；31 条记录全部存在真实分歧，全部 `resolution_*` 字段为空。
- 定向测试 `5 passed`，使用 `PYTHONPATH=backend` 后 Phase 7 全量回归 `214 passed`；确定性重跑输出哈希一致。
- 本步骤不调用外部模型/API，也不执行 Gold promotion 或 freeze。下一步 B3.6j-d 是作者 `PYH` 对 31 条分歧进行逐条裁决，完成后才能进入独立 Gold promotion gate。

## Phase 7-B3.6j-d：分歧裁决门禁与作者裁决完成（2026-08-16）

- 以 TDD 新增 `benchmark120_disagreement_adjudication.py`、固定配置和 6 项定向测试；工具固定锁定 B3.6j-c 的待裁决队列与摘要 SHA-256，并将 31 条记录确定性拆分为 `8/8/8/7` 四批。
- 四批工作包保持与 AI 建议分离；收到作者 `reviewer_id=PYH` 对 31 条建议的显式确认后，才原子写入正式裁决字段。
- 另生成 31 条只读 AI 裁决建议，用于降低作者逐条复核成本。建议分布为 `answer=29`、`boundary_refusal=2`，`reviewer_id` 保持为空，不能视为作者裁决或 Gold 标签。
- AI 建议 JSONL 与 Markdown 的 SHA-256 分别为 `3ece5f6f35af1dff038448a0eec546a39939a3cf9be6192bba907f9575b56bfd` 和 `86d76b284d2659275c41e9e047ae2b213b58d5fb66866d3a9953510f41ae5f85`。
- 裁决门禁要求 31/31 唯一完整覆盖、不可变来源字段无漂移、所有列表字段为合法 JSON 数组；`boundary_refusal` 必须同时满足 `policy_rule + safety_policy`。
- 四批以 `8/8/8/7` 全部通过校验并完成 `finalize`；正式分布为 `answer=29`、`boundary_refusal=2`。两条边界题仅将同义枚举 `medical_safety_policy` 规范化为协议值 `safety_policy`，未改变问题、来源、页码或 claim 边界。
- 正式裁决 CSV SHA-256 为 `9471dff8afb1a6f1c9e4f03e7677d28b429dc3788192bd78f9a444e16c33865f`，裁决审计 SHA-256 为 `f7a32d11a9cb6f2fb9d143e2335bcbab9639cab2e7800df850f1fb751b285de2`。

## Phase 7-B3.6j-e：Gold promotion 与冻结完成（2026-08-16）

- 以 TDD 新增并补强 `benchmark120_gold_freeze.py`，8 项定向测试覆盖输入哈希、正式裁决 CSV、裁决审计状态/哈希、31 条裁决完整覆盖、来源/页码漂移、跨 split 泄漏、政策边界以及冻结文件不可覆盖。
- Gold freeze 只允许把已接受的正式裁决映射回 `requires_second_pass=true` 记录；Validation 40 / Pilot Test 80 必须精确覆盖 120 条，且跨 split 的事实簇、证据锚点组、独立单元、来源页和问题文本不得泄漏。
- 冻结产物采用原子写入与 immutable preflight：同内容可幂等复跑，已有文件字节不同时 fail closed，禁止静默覆盖。
- 14 条未进入第二轮的旧边界记录在冻结转换中完成可审计 schema 规范化：`page_span_located -> policy_rule`、`medical_safety_policy -> safety_policy`；2 条作者裁决边界记录已是规范格式。任何不满足旧 schema 前提的记录仍 fail closed。
- 正式冻结得到 Gold 120 / Validation 40 / Pilot Test 80；决策分布为 `answer=69`、`review_required=11`、`insufficient_evidence=24`、`boundary_refusal=16`。
- Gold、Validation、Pilot Test SHA-256 分别为 `34ce42a62f004da18d9132baa47980fcf4210ee353f86c865d7e08cc469c0348`、`8f8e0c63e8fa829109df98f3e51d37324c978bf49462ad38f14bfe6f67c2e1f7`、`14afa2988d9ff579471f2d082f9803ce3bfc9e030eb439e7cf2d4c6fa55d5da9`。
- 独立内容审计和字节级幂等复跑均通过；Gold freeze 定向测试 `8 passed`，Phase 7 全量回归 `228 passed`。`gold_promotion_performed=true`、`freeze_performed=true`，但 `clinically_validated=false`，且本步骤外部模型/API、token 和费用均为 0。

## Phase 7-C1a：Validation40 运行时输入隔离（2026-08-17）

在任何检索、Prompt 构建或模型调用前，先从冻结 Validation40 生成严格 Gold 隔离的运行时投影：

```powershell
$env:PYTHONIOENCODING='utf-8'
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\validation40_runtime_projection.py --config experiments\phase7_formal_experiments\configs\validation40_runtime_projection_v0_1.json --repo-root .
```

- 固定源文件为 `benchmark120_validation40_v1_0.jsonl`，SHA-256 为 `8f8e0c63e8fa829109df98f3e51d37324c978bf49462ad38f14bfe6f67c2e1f7`。
- 运行时行严格只含 `sample_id`、`question`、`dataset_version`、`kb_version`；不携带 decision、claims、risk labels、source/page/span 等评测字段。
- 输出 40 条唯一记录和独立 selection manifest；Gold 字段泄漏为 0，`pilot_test_accessed=false`。
- 输出采用 immutable preflight：同内容可幂等复跑，已有文件字节冲突时在写入前 fail closed。
- 定向测试 `6 passed`，Phase 7 全量回归 `234 passed`；外部模型调用、input/output tokens 和估算费用均为 0。

主要产物：

- `revision/phase7/validation40_inputs/validation40_runtime_projection_v0_1.jsonl`
- `revision/phase7/validation40_inputs/validation40_selection_manifest_v0_1.jsonl`
- `revision/phase7/validation40_inputs/validation40_projection_audit_v0_1.json`
- `revision/phase7/validation40_inputs/validation40_projection_summary_v0_1.md`

下一步 Phase 7-C1b 只基于运行时投影构建离线检索与方法调用计划。Pilot Test 80 继续保持未见，真实模型调用仍需单独批准。

## Phase 7-C1b：Validation40 离线实验预检（2026-08-17）

在执行检索或调用外部模型前，先展开固定方法矩阵并锁定缓存键：

```powershell
$env:PYTHONIOENCODING='utf-8'
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\validation40_experiment_preflight.py --config experiments\phase7_formal_experiments\configs\validation40_experiment_preflight_v0_1.json --repo-root .
```

- 输入只允许来自 C1a 的 40 条 Gold 隔离运行时投影，SHA-256 固定为 `d6d425dddd5411ea4ea731e17780f1d2406bebadaf99a96a313e28c80feffbc5`。
- 固定五种方法：Vanilla LLM、Naive RAG、Multi-granularity RAG、Trust-gated RAG 和 Graph-enhanced Full Method。
- 生成 160 个离线检索任务和 200 个方法调用计划；每种方法恰好 40 条，检索键与调用键均唯一。
- 配置强制 `execute_retrieval=false`、`execute_model_calls=false`；本步骤不执行检索、不构建最终证据上下文，也不调用模型。
- 产物未出现 Gold-only 字段，Pilot Test 80 执行前后 SHA-256 均为 `14afa2988d9ff579471f2d082f9803ce3bfc9e030eb439e7cf2d4c6fa55d5da9`。
- 输出采用 immutable preflight，同内容可幂等复跑，字节冲突时整体 fail closed。
- 定向测试 `10 passed`，Phase 7 全量回归 `244 passed`，后端与 Phase 7 联合回归 `316 passed`；外部模型调用、input/output tokens 和估算费用均为 0。

主要本地产物：

- `revision/phase7/validation40_preflight/validation40_retrieval_plan_v0_1.jsonl`
- `revision/phase7/validation40_preflight/validation40_method_call_plan_v0_1.jsonl`
- `revision/phase7/validation40_preflight/validation40_experiment_preflight_audit_v0_1.json`
- `revision/phase7/validation40_preflight/validation40_experiment_preflight_summary_v0_1.md`

下一步 Phase 7-C1c 只执行和缓存 Validation40 的离线检索结果，继续禁止读取 Pilot Test 80 和调用外部生成模型。完成证据上下文审计后，真实模型调用必须另设批准门禁。

## Phase 7-C1c-1：Validation40 离线检索执行与缓存冻结（2026-08-21）

使用固定的本地 BGE embedding 和 22 源 ChromaDB 索引执行检索：

```powershell
$env:DEBUG='true'
$env:PYTHONPATH='backend'
$env:PYTHONIOENCODING='utf-8'
$env:EMBEDDING_PROVIDER='local'
$env:EMBEDDING_MODEL='BAAI/bge-small-zh-v1.5'
$env:CHROMA_PERSIST_DIR=(Resolve-Path 'backend\data\chroma_db_local').Path
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\validation40_retrieval_execution.py --config experiments\phase7_formal_experiments\configs\validation40_retrieval_execution_v0_1.json --repo-root .
```

- 配置同时锁定 embedding provider/model、Chroma 持久化目录和 `index_status.json` SHA-256；运行目录或索引身份不一致时 fail closed，避免 512 维查询误连 1024 维旧索引。
- 40 个 Validation 样本形成 160 个逻辑检索任务；Naive RAG 使用固定 512 粒度，Multi-granularity / Trust-gated / Graph-enhanced 共享多粒度物理检索，因此实际执行 80 个物理检索。
- 物理结果为 `completed=72`、`insufficient_evidence=8`、`failed=0`；每个上下文保留 0-4 个真实证据片段，不补造证据。来源、页码或正文无效、纯标题占位和完全重复片段会被过滤。
- 缓存身份包含 execution、dataset、KB、embedding、Chroma 路径和索引状态哈希；已完成/证据不足结果可复用，技术失败只重跑失败项，并保存 `error_type/error_message`。
- Pilot Test 80 仅做哈希门禁，执行前后 SHA-256 均为 `14afa2988d9ff579471f2d082f9803ce3bfc9e030eb439e7cf2d4c6fa55d5da9`；`pilot_test_accessed=false`，Gold-only 字段泄漏为 0。
- 五个正式输出 SHA-256 分别为 physical `36b1ba74...e704a`、task results `d99ad50d...bb4a`、failures `e3b0c442...b855`、audit `295c52d1...b5f`、summary `0d5f34b7...2639`。
- smoke3 两次运行的五个输出哈希逐项一致；定向测试 `8 passed`、Phase 7 回归 `252 passed`、后端/Phase 6/Phase 7 联合回归 `416 passed`。
- 本步骤没有调用外部模型/API，`external_model_calls=0`、`input_tokens=0`、`output_tokens=0`、`estimated_cost=0`；`graph_reranking_executed=false`、`clinically_validated=false`。

当前完成的是可复现的原始检索与证据缓存，不是检索质量达标或方法效果结论。人工抽样发现部分同来源同页片段存在语义重叠，且个别 Top-4 结果相关性偏弱；下一步应在 Validation40 Gold 上计算 retrieval recall/precision、来源页命中和冗余率，再决定去重、重排与阈值调整。真实生成模型调用继续等待独立批准。

## Phase 7-C1c-2：Validation40 冻结检索质量评测（2026-08-21）

对 C1c-1 已冻结的 80 条物理检索结果做离线评测，不重新检索、不调用模型、不读取 Pilot Test80：

```powershell
$env:PYTHONPATH='backend'
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\validation40_retrieval_evaluation.py --config experiments\phase7_formal_experiments\configs\validation40_retrieval_evaluation_v0_1.json
```

| 物理检索配置 | Gold 来源召回@K | Gold 来源-页码精确召回@K | 来源-页码 MRR | ±1 页诊断召回@K | 平均 Gold 锚点字符覆盖 | 证据不足率 | 词法冗余证据对比例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `single_granularity` | 0.4500 | 0.2250 | 0.1458 | 0.3250 | 0.2371 | 0.1500 | 0.0816 |
| `multi_granularity` | 0.4750 | 0.2000 | 0.1250 | 0.3000 | 0.1785 | 0.0500 | 0.1060 |

- 主指标严格使用冻结 Gold 的来源与页码精确匹配；`±1` 页只用于诊断，不能并入主指标。
- 字符 n-gram 覆盖和冗余率是确定性词法代理，不是语义相似度、claim-level support 或临床正确性指标。
- 当前只存在 `single_granularity` 与 `multi_granularity` 两种真实物理检索配置。Trust-gated 与 Graph-enhanced 尚共享多粒度检索，且 `graph_reranking_executed=false`，因此本步骤不生成虚假的四方法检索对比。
- 每题只冻结一个 Gold 主锚点，精确来源-页码召回只能作为单锚点代理；它可能低估可接受的替代权威证据，也不能视为完整证据精确率。
- 单粒度精确命中 9/40，多粒度精确命中 8/40。两者都没有达到可直接进入最终模型实验的理想检索水平；多粒度仅降低了证据不足率，但没有改善精确页码召回、MRR、词法覆盖或冗余率。
- 3 个物理结果虽然命中冻结 Gold 来源页，仍被上游标记为 `insufficient_evidence`。这说明“命中事实”和“安全状态”需要分开审计，下一步应检查最小证据数、过滤和状态路由，而不是直接改 Gold 或放宽结论。
- 四个输入资产 SHA-256 均通过锁定校验；两次正式运行的五个输出 SHA-256 完全一致。定向测试 `5 passed`、Phase 7 回归 `257 passed`、跨后端/Phase 6/Phase 7 回归 `421 passed`。
- 本步骤 `external_model_calls=0`、`input_tokens=0`、`output_tokens=0`、`estimated_cost=0`，且 `pilot_test_read=false`、`clinically_validated=false`。

主要产物：

- `revision/phase7/validation40_retrieval_evaluation/validation40_retrieval_sample_metrics_v0_1.jsonl`
- `revision/phase7/validation40_retrieval_evaluation/validation40_retrieval_profile_summary_v0_1.json`
- `revision/phase7/validation40_retrieval_evaluation/validation40_retrieval_failure_cases_v0_1.jsonl`
- `revision/phase7/validation40_retrieval_evaluation/validation40_retrieval_evaluation_audit_v0_1.json`
- `revision/phase7/validation40_retrieval_evaluation/validation40_retrieval_evaluation_summary_v0_1.md`

下一步 Phase 7-C1c-3 应只在 Validation40 上分层审计 `gold_source_miss`、`gold_page_miss`、相邻页命中、冗余片段与命中但证据不足案例，再提出版本化修复方案。任何检索器、阈值、去重或图重排改动都需另设审批和新版本；Pilot Test80 与真实模型调用继续保持关闭。

## Phase 7-C1c-3：候选恢复诊断（2026-08-22）

C1c-3 在 Validation40 上扩大每个粒度的候选池至 20，并加入确定性 RRF、相邻页扩展和最终 Top-4 证据窗口。该实验不调用外部模型，也不读取 Pilot Test80。

- Gold 来源召回@K：`0.3750`。
- Gold 来源-页码精确召回@K：`0.2000`。
- ±1 页诊断召回@K：`0.3250`。
- 来源-页码 MRR：`0.1833`。
- 证据不足率由旧多粒度配置的 `0.0500` 降至 `0.0000`，但严格来源、来源页和 MRR 均未形成整体改善。
- `query_rewrite_executed=false`、`cross_encoder_executed=false`、`external_model_calls=0`、`pilot_test_accessed=false`。

因此 C1c-3 是有效的负结果：单纯扩大候选池、相邻页扩展和确定性融合不足以解决严格召回问题。该结果用于支撑 C1c-4 方案 B，而不是性能提升结论。

主要产物：

- `revision/phase7/validation40_retrieval_recovery_v0_1/validation40_retrieval_recovery_results_v0_1.jsonl`
- `revision/phase7/validation40_retrieval_recovery_v0_1/validation40_retrieval_recovery_profile_summary_v0_1.json`
- `revision/phase7/validation40_retrieval_recovery_v0_1/validation40_retrieval_recovery_audit_v0_1.json`
- `revision/phase7/validation40_retrieval_recovery_v0_1/validation40_retrieval_recovery_summary_v0_1.md`

## Phase 7-C1c-4a/4b：CUDA、本地模型与项目配置（2026-08-23）

- 按研究者明确选择，直接升级现有 `verimind_MedAudit_env`，未新建独立 CUDA 环境；该选择降低了环境切换成本，但后续需要保留包清单和可回滚记录。
- PyTorch 已升级为 `2.12.1+cu130`，`torch.cuda.is_available()` 为 `True`，设备为 NVIDIA GeForce RTX 4060 Laptop GPU。
- `BAAI/bge-m3` 已保存到 `D:\AI_Models\huggingface\BAAI\bge-m3`；通过项目索引器离线编码，输出维度为 1024。
- `BAAI/bge-reranker-v2-m3` 已保存到 `D:\AI_Models\huggingface\BAAI\bge-reranker-v2-m3`；CUDA 离线打分 smoke 通过。
- 新增 `phase7_c1c4_local_models_v0_1.json`，锁定模型身份、1024 维新索引目录、候选 Top-20、最终证据 Top-4 和旧索引不可复用门禁。
- `backend/.env` 只保存本机模型路径和设备配置，不进入 Git；公开实验配置只保存模型 ID 和环境变量名。
- 当前只完成 dense embedding 与 reranker 的配置接入和 smoke。`sparse_retrieval.status=planned_not_implemented`，正式 Cross-Encoder 重排尚未执行，BGE-M3 新索引尚未构建。
- 验证：本轮定向测试 `20 passed`，Phase 7 全量回归 `262 passed`，后端全量回归 `74 passed`。

下一步是 C1c-4c：在 `backend/data/chroma_db_bge_m3` 构建独立 1024 维 BGE-M3 dense 索引并完成 22/22 来源审计。旧 `backend/data/chroma_db_local` 继续作为只读 BGE-small 基线；不得混用两个维度的索引。稀疏检索和 reranker 正式接入属于后续独立任务。

## Phase 7-C1c-4c：BGE-M3 独立索引构建与审计（2026-08-23）

- 使用 `BAAI/bge-m3`、CUDA 和 batch size 4，在独立目录 `backend/data/chroma_db_bge_m3` 完成 1024 维 dense 索引构建；旧 `chroma_db` 与 `chroma_db_local` 未被覆盖。
- 22/22 份已准入 PDF 均进入 `detail_128`、`concept_512`、`context_1024` 三个 collection；三层 collection 的实际向量维度均为 1024。
- collection 记录数分别为 `11884 / 7403 / 11070`；缺失来源、扫描件偏重来源、参考文献噪声、图片占位噪声和维度不匹配均为 0。
- `backend/data/chroma_db_bge_m3/index_status.json` 与 `docs/index_audit_report_chroma_db_bge_m3.json` 均显示 `ready=true`；非默认索引使用独立、带目录名的报告文件，避免覆盖旧基线审计资产。
- 3 道定向检索 smoke 分别命中 CAP 48-72 小时再评估、阿奇霉素 `qd` 和 5 岁以下发热退热药相关权威正文及真实页码。
- 旧索引状态、资料清单和冻结 Pilot Test80 的 SHA-256 在构建前后保持一致；本步骤没有读取 Pilot Test80 内容，也没有调用外部模型/API。
- 验证结果：新增索引构建/审计定向测试 `8 passed`，后端全量回归 `80 passed`，Phase 7 全量回归 `262 passed`。

本步骤证明 BGE-M3 独立索引的来源完整性、维度一致性和运行可用性，不证明严格召回已经改善。smoke 候选尾部仍可见弱相关片段，下一步 C1c-4d 需要在 Validation40 上实现并评测 sparse、dense+sparse hybrid、Cross-Encoder 重排和最终证据去冗余；在这些配置冻结前继续禁止读取 Pilot Test80。

## Phase 7-C1c-4d：强非图检索配置选择与冻结（2026-08-24）

先以本地 BGE-M3 对三个 collection 构建 learned lexical sparse CSR 索引：

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\bge_m3_sparse_index.py `
  --chroma-path backend\data\chroma_db_bge_m3 `
  --model-path D:\AI_Models\huggingface\BAAI\bge-m3 `
  --output-dir revision\phase7\c1c4d\bge_m3_sparse_index_v0_1 `
  --batch-size 4 `
  --device cuda:0
```

随后在冻结 Validation40 上运行四种真实配置，并执行独立 Gold 评测：

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:PYTHONPATH='backend'
$env:LOCAL_EMBEDDING_MODEL_PATH='D:\AI_Models\huggingface\BAAI\bge-m3'
$env:RERANKER_MODEL_PATH='D:\AI_Models\huggingface\BAAI\bge-reranker-v2-m3'
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\validation40_hybrid_retrieval.py `
  --config experiments\phase7_formal_experiments\configs\validation40_hybrid_retrieval_v0_1.json `
  --repo-root .

D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\validation40_hybrid_retrieval_evaluation.py `
  --config experiments\phase7_formal_experiments\configs\validation40_hybrid_retrieval_evaluation_v0_1.json `
  --repo-root .
```

| 配置 | 来源@20 | 严格来源页@20 | MRR@20 | 来源@4 | 严格来源页@4 | ±1页@4 | 冗余率 | 平均端到端延迟/题 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bge_m3_dense` | 0.9000 | 0.4500 | 0.2546 | 0.7500 | 0.3750 | 0.4500 | 0.0083 | 0.0376 s |
| `bge_m3_sparse` | 0.4000 | 0.4000 | 0.3500 | 0.4000 | 0.4000 | 0.4000 | 0.0000 | 0.0069 s |
| `dense_sparse_rrf` | 0.7500 | 0.5250 | 0.2670 | 0.5500 | 0.3500 | 0.3750 | 0.0000 | 0.0450 s |
| `hybrid_reranker_dedup` | 0.7500 | 0.5250 | 0.3883 | 0.7250 | 0.5000 | 0.5500 | 0.0083 | 1.0253 s |

- 按预声明顺序，以最终严格来源页召回@4、候选严格 MRR、最终来源召回@4、冗余率和延迟依次选择，冻结强非图配置为 `hybrid_reranker_dedup`。
- Cross-Encoder 相对 dense 新增 7 个严格 Top-4 命中、丢失 2 个；融合候选严格命中的 21 题中有 20 题进入最终 Top-4。仍有 19/40 题在融合 Top-20 缺少严格 Gold 页，候选生成仍是主要瓶颈。
- dense Chroma 与 sparse CSR 索引大小约为 216.9 MB 与 20.3 MB；dense/sparse/reranker 峰值显存约为 2.14/2.12/2.34 GiB。
- 原始查询向量、六路候选排名、RRF 分数、重排前后顺序和去冗余审计均已保存；五个评测产物幂等复跑哈希一致。
- 冻结配置同时锁定模型 ID、dense/sparse 索引哈希、Top-K、RRF、去冗余参数、Validation40 Gold 与检索结果哈希，以及未见 Pilot Test80 哈希。
- 本阶段外部模型/API 调用、input/output tokens 和费用均为 0；Phase 7 回归 `284 passed`，后端回归 `80 passed`。
- `0.5000` 是 Validation40 的单 Gold 主锚点严格来源页代理指标，不是临床正确率、幻觉率或独立测试集结论，也不能证明 Graph-enhanced 已改善。

主要产物：

- `revision/phase7/c1c4d/bge_m3_sparse_index_v0_1/manifest.json`
- `revision/phase7/c1c4d/validation40/validation40_hybrid_retrieval_results_v0_1.jsonl`
- `revision/phase7/c1c4d/validation40/validation40_hybrid_method_summary_v0_1.json`
- `revision/phase7/c1c4d/validation40/validation40_strong_non_graph_config_v0_1.json`
- `revision/phase7/c1c4d/validation40/validation40_hybrid_evaluation_audit_v0_1.json`

## Phase 7-C1c-4e：失败驱动的图引导候选扩展

C1c-4d 的 19 个融合候选严格缺页样本并非单一瓶颈：9 个样本已出现 Gold 来源但页码未命中，5 个只命中 `±1` 页，10 个 Gold 来源未出现，另有 3 个样本的 Gold 页已由 Dense 召回但被 RRF 丢失。因此 C1c-4e 不直接把图重排接到现有 Top-20，而按以下顺序执行：

1. `C1c-4e-0`：冻结逐题失败 taxonomy；
2. `C1c-4e-1`：运行候选预算曲线、路由并集、来源页聚合和来源拥挤度非图控制；
3. `C1c-4e-2`：实现不读取 Gold-only 字段的 graph-guided candidate expansion；
4. `C1c-4e-3`：比较 `F / G1 / G2 / G3`；
5. `C1c-4e-4`：完成图扩展、图重排、图审计和权威约束消融后冻结配置。

严格来源页召回仍为主指标，`±1` 页只作诊断。Graph 收益必须与扩大 K、路由并集和融合修复带来的收益分开报告。完整方法、生成参数、Trust Gate 和统计方案全部冻结后，才允许对 Pilot Test80 进行一次性最终评测。

阶段设计见：

- `revision/phase7/c1c4e/phase7_c1c4e_candidate_expansion_design_v0_1.md`

### C1c-4e-0 正式失败审计

```powershell
$env:PYTHONPATH='backend'
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments/phase7_formal_experiments/validation40_candidate_failure_analysis.py --config experiments/phase7_formal_experiments/configs/validation40_candidate_failure_analysis_v0_1.json --repo-root .
```

正式结果：19 个 RRF 严格缺页样本的唯一主类型为相邻页 `5`、融合丢失 `3`、来源缺失 `8`、同源错页 `3`；三路 Top-20 均缺页 `16`。路由并集恢复 `3` 题，来源页聚合降低重复候选但不增加严格页命中。真实单路 Top-40/80 尚未运行，不能用路由并集替代。

验证结果：定向测试 `9 passed`、Phase 7 全量 `293 passed`、后端全量 `80 passed`；正式产物字节级幂等复跑一致。

正式产物：

- `revision/phase7/c1c4e/failure_analysis_v0_1/candidate_failure_cases_v0_1.jsonl`
- `revision/phase7/c1c4e/failure_analysis_v0_1/candidate_failure_taxonomy_v0_1.json`
- `revision/phase7/c1c4e/failure_analysis_v0_1/non_graph_candidate_controls_v0_1.json`
- `revision/phase7/c1c4e/failure_analysis_v0_1/candidate_failure_audit_v0_1.json`
- `revision/phase7/c1c4e/failure_analysis_v0_1/candidate_failure_summary_v0_1.md`

### C1c-4e-1 Exact Dense 确定性候选预算控制

为避免 Chroma HNSW 独立进程近邻次序漂移影响配对实验，本阶段增加不依赖近似索引的 Exact Dense 控制。该分支从既有 BGE-M3 collection 导出向量，使用 NumPy 穷举平方 L2 距离和稳定 tie-break；它是实验复现控制，不替代在线 HNSW，也不构成新的性能方法。

```powershell
$env:PYTHONPATH='backend'
$env:LOCAL_EMBEDDING_MODEL_PATH='D:\AI_Models\huggingface\BAAI\bge-m3'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'

D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\build_bge_m3_exact_dense_assets.py `
  --chroma-path backend\data\chroma_db_bge_m3 `
  --output-dir backend\data\chroma_db_bge_m3_exact_control_v0_1 `
  --batch-size 256

D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\validation40_candidate_budget_retrieval.py `
  --config experiments\phase7_formal_experiments\configs\validation40_candidate_budget_retrieval_v0_3.json `
  --repo-root .

D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\validation40_candidate_budget_evaluation.py `
  --config experiments\phase7_formal_experiments\configs\validation40_candidate_budget_evaluation_v0_2.json `
  --repo-root .
```

| 方法 | Strict@20 | Strict@40 | Strict@80 | Source@80 | Adjacent@80 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bge_m3_dense_exact` | 0.4250 | 0.6750 | 0.7750 | 0.9750 | 0.8500 |
| `bge_m3_sparse` | 0.4000 | 0.4000 | 0.4000 | 0.4250 | 0.4000 |
| `dense_exact_sparse_rrf` | 0.5250 | 0.5250 | 0.7250 | 0.9250 | 0.8500 |

- Exact 资产覆盖 `30,357` 条 `1024` 维切片，约 `136.79 MiB`；manifest SHA-256 为 `66d07e2e98b3ca2d51e2d1314574c50832f59fd673fa28997c19d5b3ba7fa9ad`。
- 两个独立进程的结果 JSONL 字节级一致，`results_sha256=8a96107f091197cf3f2af2e8a7920489a0c0e2ed5bee88ace3b932eeb970806a`。
- 原 HNSW Dense Strict@20/40/80 为 `0.4500/0.6750/0.7750`；RRF 命中数与 Exact 分支完全相同。Exact 的作用是确定性控制，不是性能提升。
- Gold 与 Pilot Test80 哈希未变；外部模型/API 调用、input/output tokens 和费用均为 0。
- 完整 Phase 7 回归按 `PYTHONPATH=backend` 执行为 `305 passed`。
- C1c-4e-1 至此完成；C1c-4e-2 将在相同 Exact Dense 控制后端与候选预算下开发 graph-guided candidate expansion，避免把检索后端或扩大 K 的收益误计为 Graph 收益。

### C1c-4e-2b-2：Exact-control F/G1 配对实验

本实验从已冻结的 `dense_exact_sparse_rrf` Top-20 读取同一份候选源，比较：

- `F_exact = exact RRF Top-20 + bge-reranker-v2-m3 + dedup Top-4`；
- `G1_exact = exact RRF Top-20（预留 4 个图候选替换槽位）+ 同一 reranker + 同一 dedup Top-4`。

图候选只能替换基线尾部候选，不能把候选总预算从 20 扩大。运行阶段不读取 Validation40 Gold；Gold 只在独立评估脚本中打开。Pilot Test80 始终只做字节级 SHA-256 校验，不解析内容。

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:RERANKER_MODEL_PATH='D:\AI_Models\huggingface\BAAI\bge-reranker-v2-m3'
$env:RERANKER_DEVICE='cuda'
$env:PYTHONPATH='.;backend'

# 独立运行 A
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\validation40_graph_candidate_paired_retrieval.py `
  --config experiments\phase7_formal_experiments\configs\validation40_graph_candidate_paired_retrieval_v0_1.json `
  --repo-root .

# 独立运行 B：必须写入空目录
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\validation40_graph_candidate_paired_retrieval.py `
  --config experiments\phase7_formal_experiments\configs\validation40_graph_candidate_paired_retrieval_v0_1.json `
  --repo-root . `
  --output-dir revision\phase7\c1c4e\validation40_f_g1_exact_v0_1\run_b

# A/B 核心结果哈希一致后，只评估 run_a
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\validation40_graph_candidate_paired_evaluation.py `
  --config experiments\phase7_formal_experiments\configs\validation40_graph_candidate_paired_evaluation_v0_1.json `
  --repo-root .
```

冻结 `G1` 的必要条件是候选严格来源页召回@20高于 `F_exact`，且最终严格来源页召回@4不低于 `F_exact`。若条件不满足，保留负结果并继续完善疾病、药物、适应证与来源门控，不得通过扩大 K 或改用不同后端包装为 Graph 收益。

#### C1c-4e-2b-2 正式结果

- 两个独立进程的核心结果 JSONL 均为 40 条，SHA-256 同为 `7d49bd1c2e2c9e3d21bc8049d3ef9c4d097d9d28d35dfec902eac269401192cd`，逐字节一致。
- `F_exact`：候选严格来源页召回@20 `0.5250`，候选来源召回@20 `0.7500`，最终严格来源页召回@4 `0.5000`，最终来源召回@4 `0.7250`。
- `G1_exact`：上述指标与 `F_exact` 完全相同；新增命中 `0`、新增丢失 `0`、图新增候选 `0`、零扩展 `40/40`。
- 根因只读诊断：Validation40 中 29 题未抽取到运行时约束，11 题仅抽取到 1 类约束；图扩展规则要求至少命中 2 类约束，因此没有样本满足路径门槛。
- 冻结建议为 `do_not_freeze_g1`。该结果证明实验控制和负结果审计有效，但不构成“Graph 无效”的方法结论；下一步先修复 Gold-free 约束抽取覆盖，再注册新的 `G1-v0.2`。
- 两次运行总耗时分别约 `119.84 s` 与 `109.14 s`，峰值 CUDA 显存均约 `2620.69 MB`；Pilot Test80 未解析，检索阶段未读取 Gold，外部模型调用、token 和费用均为 0。
- 完整 Phase 7 回归为 `329 passed`。

### C1c-4e-2e-2b：G1-v0.2 Gold-only 配对评测

```powershell
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\validation40_graph_candidate_paired_evaluation.py `
  --config experiments\phase7_formal_experiments\configs\validation40_graph_candidate_paired_evaluation_v0_2.json `
  --repo-root .
```

- 冻结输入检索结果 SHA-256 为 `66861d4a83a182a170a4ea0b6d0d34788446d9ba96e8beb282b85bde235d3c45`，检索 manifest SHA-256 为 `e34e6960be5ebb48a8cadf18082d052a76cb94d98579fbeb3060b4d52459068c`。
- 候选 Top-20 严格来源页召回由 F 的 `21/40`（`0.5250`）提高到 G1-v0.2 的 `24/40`（`0.6000`）；来源召回由 `30/40` 提高到 `36/40`。
- 最终 Top-4 严格来源页召回由 `20/40`（`0.5000`）提高到 `22/40`（`0.5500`）；来源召回由 `29/40` 提高到 `32/40`。
- 候选阶段新增/丢失命中为 `5/2`，最终阶段为 `4/2`。预声明裁决为 `freeze_g1_candidate_expansion`，只冻结 Validation40 开发配置中的 G1 候选扩展。
- 正式产物位于 `revision/phase7/c1c4e/validation40_f_g1_exact_v0_2/evaluation/`；评测读取 Validation40 Gold，但未读取 Pilot Test80，且未调用外部模型/API。
- 评测器定向回归为 `8 passed`；按 `PYTHONPATH=backend` 执行的 Phase 7 全量回归为 `359 passed`。
- 下一步先审计候选和最终阶段的新增丢失样本，再设计 `C1c-4e-3` 的 G2 图重排；当前不得写成完整 Graph-enhanced、统计显著性、幻觉率或临床安全收益。

### C1c-4e-2e-2c：正式损失审计

```powershell
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\validation40_graph_candidate_loss_audit.py `
  --config experiments\phase7_formal_experiments\configs\validation40_graph_candidate_loss_audit_v0_1.json `
  --repo-root .
```

- 冻结 F/G1-v0.2 的候选严格来源页命中为 `21/40` 与 `24/40`，最终为 `20/40` 与 `22/40`。
- 两个候选丢失与两个最终丢失为同一组样本；Gold 在 F pre-rerank rank 19/20，可被现有 reranker 提升到 rank 4/5，但在 G1 固定预算替换时被移出候选池。
- 共享候选共 `730` 条，内容差异为 0；最大重排分数差 `2.2649765e-06` 低于 `1e-5` 容差。
- late-union 诊断达到候选 `26/40`、最终 `22/40`，但 reranker 输入增至 20-24 条，只能作为瓶颈诊断。
- 当前直接入口为 `C1c-4e-3a`：先运行同预算非图 `F24` 控制，再比较 `F20/F24/G1` 并设计 G2；Pilot Test80 与外部生成模型继续关闭。

### C1c-4e-3a：前缀稳定的 F24/G1-v0.4 公平预算实验

Gold-free 配对检索：

```powershell
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\validation40_graph_candidate_paired_retrieval.py `
  --config experiments\phase7_formal_experiments\configs\validation40_graph_candidate_paired_retrieval_v0_4.json `
  --repo-root .
```

Gold-only 离线评测：

```powershell
D:\anaconda\envs\verimind_MedAudit_env\python.exe experiments\phase7_formal_experiments\validation40_graph_candidate_paired_evaluation.py `
  --config experiments\phase7_formal_experiments\configs\validation40_graph_candidate_paired_evaluation_v0_4.json `
  --repo-root .
```

- `F24` 固定保留 Exact RRF Top-20，并从同一路 Top-40 顺序补入前 4 个未见候选；`G1-v0.4` 保护同一 Top-20，只允许图候选替换 4 个预留槽位。
- run_a/run_b 的 40 条核心结果逐字节一致，SHA-256 均为 `79e56ffb5347ccb19df1cd1647a9acaf27ed1e055f68712df32c2f90fddfca22`，结构审计违规为 0。
- 候选严格来源页召回@24由 `21/40` 提高到 `26/40`，added/lost=`5/0`；最终严格来源页召回@4由 `20/40` 提高到 `22/40`，added/lost=`2/0`。
- 预声明裁决为 `freeze_g1_candidate_expansion`，仅冻结 Validation40 开发配置中的 G1 候选扩展。候选层双侧 McNemar `p=0.0625`，最终层 `p=0.5`，当前不声称统计显著性。
- 5 个新增 Gold 候选均由图扩展加入，但输出候选尚未携带完整 graph path trace。下一步先补齐 provenance，再进入 `C1c-4e-3b` G2 图重排。
- 正式产物位于 `revision/phase7/c1c4e/validation40_f24_g1v04_prefix_stable/`；Pilot Test80 未读取，外部模型/API、input/output tokens 和费用均为 0。
