"""
Medaudit-RAG API 路由
"""

import time
import logging
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from app.config import get_settings
from app.models.schemas import (
    AuditQueryRequest,
    AuditQueryResponse,
    HealthResponse,
    TrustScoreDetail,
    TrustLevel,
    IntentType,
)
from app.services.llm_client import get_llm_client
from app.services.safety_boundary import is_direct_prescription_request

logger = logging.getLogger(__name__)
router = APIRouter()

# Backward-compatible alias retained for existing callers and regression tests.
_is_direct_prescription_request = is_direct_prescription_request


def _blocked_prescription_answer() -> str:
    return (
        "已拦截：当前系统不提供个体化处方开具。儿童发热咳嗽需结合年龄、体重、体温、"
        "查体、病原学、过敏史、肝肾功能等信息，由医生线下评估。"
        "可改问某指南对儿童发热咳嗽初步评估有哪些建议，或某药物在儿童中的说明书/指南依据。"
    )


def _rejected_trust_score() -> TrustScoreDetail:
    return TrustScoreDetail(
        s_ret=0.0,
        s_faith=0.0,
        w_authority=0.0,
        trust_score=0.0,
        trust_level=TrustLevel.REJECTED,
    )


def _serialize_evidence_chunks(chunks):
    """将检索结果转换为前端稳定消费的证据结构。"""
    payload = []
    for chunk in chunks:
        page_number = getattr(chunk, "page_number", None)
        payload.append(
            {
                "content": chunk.content,
                "source": chunk.source_file,
                "page": page_number,
            }
        )
    return payload


def _resolve_answer_from_state(state: dict) -> str:
    """兼容旧字段 answer 与当前图状态 draft_answer。"""
    return (
        state.get("draft_answer")
        or state.get("answer")
        or "内部错误: 引擎未生成回答"
    )


@router.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """健康检查 — 返回系统状态和当前模型配置"""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        llm_provider=settings.LLM_PROVIDER,
        models={
            "generator": settings.LLM_MODEL_GENERATOR,
            "judge": settings.LLM_MODEL_JUDGE,
            "router": settings.LLM_MODEL_ROUTER,
        },
    )


@router.post("/audit/query", response_model=AuditQueryResponse, tags=["审计"])
async def audit_query(
    request: AuditQueryRequest,
):
    """
    医药合规审计查询 (非流式)

    完整的 Agent DAG 流程:
    用户提问 → 知识对齐 → 意图路由 → 三粒度检索 → 推理生成 → 审计门控 → 返回
    """
    from app.agents.graph import audit_engine
    start_time = time.time()

    try:
        # 同步等待整个图跑完
        if _is_direct_prescription_request(request.query):
            return AuditQueryResponse(
                query=request.query,
                normalized_query=request.query,
                intent=IntentType.DETAIL,
                answer=_blocked_prescription_answer(),
                trust_score=_rejected_trust_score(),
                evidence=[],
                processing_time=round(time.time() - start_time, 3),
            )

        final_state = await audit_engine.ainvoke({"original_query": request.query})

        processing_time = time.time() - start_time

        return AuditQueryResponse(
            query=request.query,
            normalized_query=final_state.get("normalized_query", request.query),
            intent=final_state.get("intent", IntentType.CONCEPT),
            answer=_resolve_answer_from_state(final_state),
            trust_score=final_state.get("trust_score", TrustScoreDetail(
                s_ret=0.0, s_faith=0.0, w_authority=0.0, trust_score=0.0, trust_level=TrustLevel.REJECTED
            )),
            evidence=_serialize_evidence_chunks(final_state.get("evidence", [])),
            processing_time=round(processing_time, 3),
        )
    except Exception as e:
        logger.error(f"审计查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询处理失败: {str(e)}")


@router.post("/audit/query/stream", tags=["审计"])
async def audit_query_stream(
    request: AuditQueryRequest,
):
    """
    医药合规审计查询 (SSE 流式)

    前端通过 EventSource 接收图流转状态，用于展示“执行轨迹”及最终结果
    """
    import json
    from app.agents.graph import audit_engine

    async def event_generator():
        try:
            # 发送开始事件
            yield f"data: {json.dumps({'type': 'start', 'query': request.query})}\n\n"

            # astream_events 能够深度捕获所有图内 LLM 调用的实时 Token
            if _is_direct_prescription_request(request.query):
                payload = {
                    "type": "node_update",
                    "node": "auditor",
                    "intent": IntentType.DETAIL.value,
                    "normalized_query": request.query,
                    "evidence_count": 0,
                    "evidence": [],
                    "answer": _blocked_prescription_answer(),
                    "trust_score": _rejected_trust_score().dict(),
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            async for event in audit_engine.astream_events({"original_query": request.query}, version="v2"):
                kind = event["event"]

                # 1. 不透传底层 LLM token，避免 Router/Judge 的结构化 JSON 泄漏到聊天气泡。
                # 前端通过 node_update.answer 展示 Generator/Auditor 的最终审计结论。
                if kind == "on_chat_model_stream":
                    continue

                # 2. 捕获每个大节点跑完后的状态更新
                elif kind == "on_chain_end":
                    # name 对应了我们的 router, retriever, generator, auditor
                    node_name = event["name"]
                    if node_name in ["router", "retriever", "generator", "auditor"]:
                        state_update = event["data"].get("output")
                        if isinstance(state_update, dict):
                            payload = {
                                "type": "node_update",
                                "node": node_name,
                            }
                            if "intent" in state_update:
                                payload["intent"] = state_update["intent"].value if hasattr(state_update["intent"], "value") else state_update["intent"]
                            if "normalized_query" in state_update:
                                payload["normalized_query"] = state_update["normalized_query"]

                            # 携带证据核心字段给前端展示
                            if "evidence" in state_update:
                                payload["evidence_count"] = len(state_update["evidence"])
                                payload["evidence"] = _serialize_evidence_chunks(state_update["evidence"])

                            if "draft_answer" in state_update:
                                payload["answer"] = state_update["draft_answer"]

                            if "trust_score" in state_update:
                                ts = state_update["trust_score"]
                                payload["trust_score"] = ts.dict() if hasattr(ts, "dict") else ts

                            # 仅当有实际状态更新时下发
                            if len(payload) > 2:
                                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            # 发送完成事件
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.error(f"流式查询失败: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
