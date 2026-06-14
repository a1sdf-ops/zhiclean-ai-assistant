"""知识库 请求/响应模型"""

from pydantic import BaseModel, Field


class KnowledgeUploadRequest(BaseModel):
    content: str = Field(..., description="要上传的文本内容", min_length=1)
    filename: str = Field(..., description="文档名称", min_length=1)


class KnowledgeUpdateRequest(BaseModel):
    source_name: str = Field(..., description="要更新的文档名称", min_length=1)
    content: str = Field(..., description="新的文本内容", min_length=1)


class KnowledgeDeleteRequest(BaseModel):
    source_name: str = Field(..., description="要删除的文档名称", min_length=1)


class KnowledgeListResponse(BaseModel):
    data: list[dict] = Field(default_factory=list, description="文档列表")
    total: int = Field(default=0, description="文档总数")
    page: int = Field(default=1, description="当前页码")
    page_size: int = Field(default=10, description="每页条数")
