"""Agent 请求/响应模型"""

from pydantic import BaseModel, Field


class AgentChatRequest(BaseModel):
    query: str = Field(..., description="用户输入", min_length=1)
    session_id: str = Field(default="", description="会话标识，不传则自动生成")
    tenant_id: str = Field(default="default", description="租户标识，用于多租户记忆隔离")


class AgentChatResponse(BaseModel):
    answer: str = Field(..., description="Agent 的最终回答")
    mode: str = Field(default="stream", description="执行模式: stream / async")
