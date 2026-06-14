"""知识库管理 API"""

import os

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.dependencies import get_kb_service
from api.schemas.knowledge import (
    KnowledgeListResponse,
    KnowledgeUpdateRequest,
    KnowledgeUploadRequest,
)

router = APIRouter(prefix="/api/v1/knowledge", tags=["Knowledge"])


@router.post("/upload")
async def upload_text(req: KnowledgeUploadRequest, kb=Depends(get_kb_service)):
    """上传文本内容到知识库"""
    result = kb.upload_by_str(req.content, req.filename)
    return {"message": result}


@router.post("/upload-file")
async def upload_file(file: UploadFile = File(...), kb=Depends(get_kb_service)):
    """上传文件到知识库"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("gbk")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="文件编码不支持，请使用 UTF-8 或 GBK")

    result = kb.upload_by_str(text, file.filename)
    return {"message": result, "filename": file.filename}


@router.get("/list", response_model=KnowledgeListResponse)
async def list_documents(page: int = 1, page_size: int = 10, kb=Depends(get_kb_service)):
    """分页列出知识库文档"""
    result = kb.list_knowledge(page=page, page_size=page_size)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return KnowledgeListResponse(**result)


@router.put("/{source_name}")
async def update_document(
    source_name: str,
    req: KnowledgeUpdateRequest,
    kb=Depends(get_kb_service),
):
    """更新知识库文档（用请求体中的 source_name 可覆盖路径参数）"""
    name = req.source_name or source_name
    result = kb.update_knowledge(name, req.content)
    return {"message": result}


@router.delete("/{source_name}")
async def delete_document(source_name: str, kb=Depends(get_kb_service)):
    """删除知识库文档"""
    result = kb.delete_knowledge(source_name)
    return {"message": result}
