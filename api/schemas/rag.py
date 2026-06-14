"""RAG 请求/响应模型"""

from pydantic import BaseModel, Field


class RagQueryRequest(BaseModel):
    question: str = Field(..., description="用户问题", min_length=1)
    session_id: str = Field(default="default", description="会话ID，用于保持对话历史")


class RagQueryResponse(BaseModel):
    answer: str = Field(..., description="RAG 生成的回答")
    session_id: str = Field(default="default", description="会话ID")
