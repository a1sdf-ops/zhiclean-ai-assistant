"""FastAPI 应用入口"""

import os
import sys
import time
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.routers import agent, knowledge, rag
from utils.logger_handler import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("应用启动，预加载服务...")
    from api.dependencies import get_agent, get_kb_service, get_rag_service

    get_rag_service()
    get_kb_service()
    get_agent()
    try:
        from agent部分.mcp_client import get_mcp_manager

        get_mcp_manager().connect_all()
    except Exception as e:
        logger.warning("MCP 服务连接失败（非致命）: %s", e)
    logger.info("所有服务预加载完成")
    yield
    logger.info("应用关闭")


app = FastAPI(
    title="知识库 RAG + Agent API",
    description="基于 LangChain + Chroma + DashScope 的知识库问答系统",
    version="2.0.0",
    lifespan=lifespan,
)

# ---------- 中间件（后加的先执行） ----------


@app.middleware("http")
async def log_request(request: Request, call_next):
    """请求日志：记录方法、路径、状态码、耗时"""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %d (%.0fms)", request.method, request.url.path, response.status_code, elapsed)
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag.router)
app.include_router(agent.router)
app.include_router(knowledge.router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}
