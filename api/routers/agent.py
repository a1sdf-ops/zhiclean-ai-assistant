"""Agent 对话 API"""

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from api.dependencies import get_agent
from api.schemas.agent import AgentChatRequest, AgentChatResponse

router = APIRouter(prefix="/api/v1/agent", tags=["Agent"])


@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(req: AgentChatRequest, agent=Depends(get_agent)):
    """Agent 对话（非流式）"""
    tokens = []
    for token in agent.execute_stream(req.query):
        tokens.append(token)
    return AgentChatResponse(answer="".join(tokens))


@router.post("/stream")
async def agent_stream(req: AgentChatRequest, agent=Depends(get_agent)):
    """Agent 流式对话（SSE）"""

    def generate():
        for token in agent.execute_stream(req.query):
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
    answer = await agent.ainvoke(req.query)
    return AgentChatResponse(answer=answer, mode="async")
