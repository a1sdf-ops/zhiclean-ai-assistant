"""Agent 请求/响应模型"""

from pydantic import BaseModel, Field


class AgentChatRequest(BaseModel):
    query: str = Field(..., description="用户输入", min_length=1)


class AgentChatResponse(BaseModel):
    answer: str = Field(..., description="Agent 的最终回答")
    mode: str = Field(default="stream", description="执行模式: stream / async")
