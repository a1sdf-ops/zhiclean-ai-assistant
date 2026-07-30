"""Agent 请求/响应模型"""

from pydantic import BaseModel, Field


class AgentChatRequest(BaseModel):
    query: str = Field(..., description="用户输入", min_length=1)
    session_id: str = Field(default="", description="会话标识，不传则自动生成")
    tenant_id: str = Field(default="", description="租户标识，为空时自动从 session_id 派生")


class AgentChatResponse(BaseModel):
    answer: str = Field(..., description="Agent 的最终回答")
    mode: str = Field(default="stream", description="执行模式: stream / async")


class SessionCloseRequest(BaseModel):
    session_id: str = Field(..., description="会话标识")
    tenant_id: str = Field(default="", description="租户标识，为空时自动从 session_id 派生")
    messages: list[str] = Field(..., description="完整对话文本列表，格式: ['用户: ...', '助手: ...']")


class SessionCloseResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    summary: str | None = Field(default=None, description="生成的会话摘要")
    summary_length: int = Field(default=0, description="摘要字数")
