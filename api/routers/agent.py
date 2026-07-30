"""Agent 对话 API"""

import json
from typing import Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

import config
from agent.token_tracker import get_tracker
from api.dependencies import get_agent
from api.schemas.agent import AgentChatRequest, AgentChatResponse, SessionCloseRequest, SessionCloseResponse

router = APIRouter(prefix="/api/v1/agent", tags=["Agent"])


@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(req: AgentChatRequest, agent=Depends(get_agent)):
    """Agent 对话（非流式）"""
    tokens = []
    for token in agent.execute_stream(req.query, req.session_id, req.tenant_id):
        tokens.append(token)
    return AgentChatResponse(answer="".join(tokens))


@router.post("/stream")
async def agent_stream(req: AgentChatRequest, agent=Depends(get_agent)):
    """Agent 流式对话（SSE）"""

    if config.STREAM_MODE == "stream":
        # 真流式: astream_events 逐 token 推送
        async def generate():
            async for token in agent.aexecute_stream(req.query, req.session_id, req.tenant_id):
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # 原有路径: invoke 模式，节点级粒度
        def generate():
            for token in agent.execute_stream(req.query, req.session_id, req.tenant_id):
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )


@router.post("/invoke", response_model=AgentChatResponse)
async def agent_invoke(req: AgentChatRequest, agent=Depends(get_agent)):
    """Agent 异步全量调用（测试/批处理路径）"""
    answer = await agent.ainvoke(req.query, req.session_id, req.tenant_id)
    return AgentChatResponse(answer=answer, mode="async")


@router.get("/cost")
async def get_cost_report(
    start_date: str | None = Query(None, description="起始日期 ISO 格式，如 2026-07-01"),
    end_date: str | None = Query(None, description="结束日期"),
    module: str | None = Query(None, description="按模块过滤"),
):
    """Token 成本历史报告（从 SQLite 查询）"""
    tracker = get_tracker()
    rows = tracker.cost_report(start_date=start_date, end_date=end_date, module=module)
    session_report = tracker.report()
    return {
        "history": rows,
        "current_session": {k: v for k, v in session_report.items() if k != "__total__"},
        "session_total": session_report.get("__total__", {}),
    }


@router.post("/session/close", response_model=SessionCloseResponse)
async def close_session(req: SessionCloseRequest, agent=Depends(get_agent)):
    """关闭会话，触发 LLM 生成排障过程摘要并写入 ChromaDB

    前端在用户离开对话时调用。摘要聚焦 JSON 画像 schema 外的碎片化排障细节。
    """
    if not req.messages:
        return SessionCloseResponse(success=False, summary=None, summary_length=0)

    summary = agent.close_session(req.messages, req.session_id, req.tenant_id)
    if summary:
        return SessionCloseResponse(success=True, summary=summary, summary_length=len(summary))
    return SessionCloseResponse(success=False, summary=None, summary_length=0)
