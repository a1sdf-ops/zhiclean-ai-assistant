"""RAG 问答 API"""

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from api.dependencies import get_rag_service
from api.schemas.rag import RagQueryRequest, RagQueryResponse

router = APIRouter(prefix="/api/v1/rag", tags=["RAG"])


@router.post("/query", response_model=RagQueryResponse)
async def rag_query(req: RagQueryRequest, rag=Depends(get_rag_service)):
    """单次 RAG 问答（非流式）"""
    tokens = []
    for token in rag.ask_stream(req.question, session_id=req.session_id):
        tokens.append(token)
    return RagQueryResponse(answer="".join(tokens), session_id=req.session_id)


@router.post("/stream")
async def rag_stream(req: RagQueryRequest, rag=Depends(get_rag_service)):
    """流式 RAG 问答（SSE）"""

    def generate():
        for token in rag.ask_stream(req.question, req.session_id):
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
