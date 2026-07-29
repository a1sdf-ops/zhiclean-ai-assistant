"""Agent 请求/响应模型"""

from pydantic import BaseModel, Field


class AgentChatRequest(BaseModel):
    query: str = Field(..., description="用户输入", min_length=1)
    session_id: str = Field(default="", description="会话标识，不传则自动生成")
    tenant_id: str = Field(default="", description="租户标识，为空时自动从 session_id 派生")


class AgentChatResponse(BaseModel):
    answer: str = Field(..., description="Agent 的最终回答")
    mode: str = Field(default="stream", description="执行模式: stream / async")
