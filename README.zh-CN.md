# MedAudit-RAG

[English README](./README.md)

MedAudit-RAG 是一个面向儿科用药问答的证据链审计 RAG 研究原型，用于检查模型回答是否受到指南、共识、药品说明书、目录等已准入权威资料的证据支持。

它不是临床诊断系统，不生成真实处方，也不能替代医生。当前仓库已包含 vector RAG + TrustScore baseline、开发阶段的 Graph-enhanced evidence auditing 方法原型，以及用于正式评测的 guideline-grounded 冻结基准；方法的经验效果尚未建立。

## 核心能力

- 将儿科用药问题路由为 `DETAIL`、`CONCEPT`、`CONTEXT` 等审计意图。
- 从多粒度知识库索引中检索证据片段。
- 要求生成回答只能基于已检索证据。
- 审计检索相关性、回答忠实度和资料权威度。
- 通过 TrustScore Gate 判断：有证据支持、需要复核、证据不足或安全边界拒答。
- 在前端展示审计状态、得分分解、引用来源、页码和证据片段。

## 医学安全边界

本项目仅用于科研、教学和证据审计方法验证。

系统不提供真实临床诊断、个体化处方或治疗建议。所有医学结论都必须回到已检索到的证据。如果证据不足、证据不匹配、资料不完整，或用户请求超出允许回答边界，系统应拒答或提示人工复核。

## 系统架构

```text
用户问题
    |
    v
路由器 / 意图标准化
    |
    v
多粒度检索器
    |
    v
受约束生成器
    |
    v
证据审计器
    |
    v
TrustScore 门控
    |
    +--> 有证据支持
    +--> 需要人工复核
    +--> 证据不足
    +--> 安全边界拒答
```

TrustScore 基于检索相关性、回答忠实度和资料权威度：

```text
T = alpha * S_ret + beta * S_faith
TrustScore = T * W_authority
```

## 技术栈

- 后端：Python, FastAPI
- 工作流编排：LangGraph
- 向量数据库：ChromaDB
- 前端：React, Ant Design, Vite
- 流式输出：Server-Sent Events
- 测试：pytest

## 仓库范围

当前仓库包含：

- 健康检查、审计问答、SSE 流式接口
- router、retriever、generator、auditor 和 TrustScore gate 逻辑
- 指南资料准入和 manifest 追踪脚本
- 向量索引重建与审计脚本
- 用于展示审计过程和证据链的 React 前端
- parser、retriever、流式序列化和 TrustScore 相关单测

仓库不提交原始指南 PDF、本地 ChromaDB 索引、API Key 或个人规划笔记。

## 知识库与资料准入

正式资料进入知识库前，需要经过 manifest 记录和准入检查。manifest 是判断资料是否进入正式知识库的事实来源。

```text
data/guidelines/source_manifest.json
backend/data/chroma_db/
backend/data/chroma_db/index_status.json
```

资料准入流程会记录 source type、准入状态、解析诊断、checksum 和索引状态。生成的 ChromaDB 索引属于可复现本地资产，不进入 Git。

## 快速启动

安装后端依赖：

```powershell
pip install -r backend/requirements.txt
```

运行后端测试：

```powershell
$env:DEBUG='true'
$env:PYTHONPATH='backend'
python -m pytest backend/tests -q
```

重建向量索引：

```powershell
$env:DEBUG='true'
$env:PYTHONPATH='backend'
python backend/rebuild_index.py
```

如果需要离线或隐私更敏感的检索实验，可以启用可选的本地 embedding 模式：

```powershell
pip install -r backend/requirements-local-embedding.txt
$env:EMBEDDING_PROVIDER='local'
$env:EMBEDDING_MODEL='BAAI/bge-small-zh-v1.5'
$env:CHROMA_PERSIST_DIR='backend/data/chroma_db_local'
python backend/rebuild_index.py
```

启动后端：

```powershell
$env:PYTHONPATH='backend'
python -m uvicorn app.main:app --reload
```

启动前端：

```powershell
Set-Location frontend
npm install
npm run dev
```

Windows 下也可以使用便捷脚本：

```powershell
.\start-dev.ps1
```

## API 接口

```text
GET  /api/health
POST /api/audit/query
POST /api/audit/query/stream
```

审计接口会返回标准化问题、意图、回答或拒答文本、TrustScore、得分分解、证据片段、引用来源、页码和最终门控结论。

## 评测方向

后续评测集定位为 guideline-grounded pediatric medication safety QA。每道题应包含：

- gold evidence 的来源、页码和原文片段
- expected decision
- allowed answer scope
- forbidden claims
- risk labels
- dataset、prompt、model 和 knowledge-base 版本

计划关注 hallucination rate、unsupported claim rate、unsafe suggestion rate、refusal correctness、claim-evidence alignment 和 evidence-source mismatch rate。未来如果声明错误率下降，必须提供原始输出、审计轨迹、置信区间和统计检验支撑。

## 实验纪律

所有模型实验都应控制成本、可缓存、可复现：

1. 每次模型或 judge 调用记录 `input_tokens`、`output_tokens` 和 `estimated_cost`。
2. 保留 raw model outputs，再做后处理。
3. 相同 `sample_id + method + model + prompt_version` 已经成功运行时，不重复调用。
4. 重跑时优先只重跑 failed cases，不整批重跑。
5. evidence context 控制在 2-4 个证据片段，不整页塞入 prompt。
6. 每个输出文件写入 `prompt_version`、`dataset_version` 和 `kb_version`。

## 当前研究状态

本仓库是持续开发中的研究原型。详细阶段状态、任务追踪、实验记录和发现统一维护在 `revision/`，不放在 README 首页中反复更新。

## Roadmap

- 提升检索稳定性和 source/page 精度。
- 扩展并审计儿科用药权威知识库。
- 在已冻结的 guideline-grounded benchmark split 上执行可复现方法对比。
- 对比 vanilla LLM、naive RAG、multi-granularity RAG、TrustScore Gate 和 graph-enhanced evidence auditing。
- 保存 raw outputs、失败样本、置信区间和统计检验结果，为论文写作提供可审计证据。
