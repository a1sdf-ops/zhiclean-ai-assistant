"""
知识库 MCP Server

将 RAG 知识库能力封装为 MCP Tools，供 Claude Code、OpenClaw 等 Agent 调用。

支持的 Tools:
  - search_knowledge: 语义搜索知识库
  - upload_knowledge: 上传文本到知识库
  - upload_knowledge_file: 上传 txt 文件到知识库
  - list_knowledge: 分页列出知识库内容
  - update_knowledge: 更新知识库文档（文本输入）
  - update_knowledge_file: 更新知识库文档（文件输入）
  - delete_knowledge: 删除知识库中的文档

启动方式:
  python mcp_server.py

在 Claude Code 中配置:
  claude mcp add knowledge-base -- python "D:\\桌面\\...\\RAG部分\\mcp_server.py"
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from RAG部分.knowledge_base import KnowledgeBaseService
from RAG部分.rag import RagService
from utils.logger_handler import logger

app = Server("knowledge-base-mcp")

# 全局服务实例（懒加载）
_rag_service = None
_kb_service = None


def get_rag():
    global _rag_service
    if _rag_service is None:
        _rag_service = RagService()
    return _rag_service


def get_kb():
    global _kb_service
    if _kb_service is None:
        _kb_service = KnowledgeBaseService()
    return _kb_service


@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="search_knowledge",
            description="语义搜索知识库，输入自然语言查询，返回最相关的知识内容。例如：搜索'春天'可以找到朱自清《春》的内容。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询，支持自然语言语义搜索",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "会话ID，用于保持对话历史，默认 'default'",
                        "default": "default",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="upload_knowledge",
            description="上传文本内容到知识库，支持直接输入文本。",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "要上传的文本内容",
                    },
                    "filename": {
                        "type": "string",
                        "description": "文档名称，用于标识知识来源",
                    },
                },
                "required": ["content", "filename"],
            },
        ),
        Tool(
            name="upload_knowledge_file",
            description="上传本地 txt 文件到知识库。",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "txt 文件的绝对路径",
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="list_knowledge",
            description="分页列出知识库中的所有文档。",
            inputSchema={
                "type": "object",
                "properties": {
                    "page": {
                        "type": "integer",
                        "description": "页码，从 1 开始，默认 1",
                        "default": 1,
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "每页条数，默认 10",
                        "default": 10,
                    },
                },
            },
        ),
        Tool(
            name="update_knowledge",
            description="更新知识库中的文档，传入新的文本内容替换旧文档。",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "要更新的文档名称",
                    },
                    "content": {
                        "type": "string",
                        "description": "新的文本内容",
                    },
                },
                "required": ["source_name", "content"],
            },
        ),
        Tool(
            name="update_knowledge_file",
            description="更新知识库中的文档，从 txt 文件读取新内容替换旧文档。",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "要更新的文档名称",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "新内容的 txt 文件绝对路径",
                    },
                },
                "required": ["source_name", "file_path"],
            },
        ),
        Tool(
            name="delete_knowledge",
            description="按文档名称删除知识库中的文档。",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "要删除的文档名称",
                    },
                },
                "required": ["source_name"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        if name == "search_knowledge":
            query = arguments.get("query", "").strip()
            if not query:
                return [TextContent(type="text", text="[错误] query 不能为空")]

            session_id = arguments.get("session_id", "default")
            rag = get_rag()

            # 收集流式输出
            full_response = []
            for token in rag.ask_stream(query, session_id=session_id):
                full_response.append(token)

            result = "".join(full_response)
            return [TextContent(type="text", text=result)]

        elif name == "upload_knowledge":
            content = arguments.get("content", "").strip()
            filename = arguments.get("filename", "").strip()

            if not content:
                return [TextContent(type="text", text="[错误] content 不能为空")]
            if not filename:
                return [TextContent(type="text", text="[错误] filename 不能为空")]

            kb = get_kb()
            result = kb.upload_by_str(content, filename)
            return [TextContent(type="text", text=result)]

        elif name == "upload_knowledge_file":
            file_path = arguments.get("file_path", "").strip()

            if not file_path:
                return [TextContent(type="text", text="[错误] file_path 不能为空")]
            if not os.path.exists(file_path):
                return [TextContent(type="text", text=f"[错误] 文件不存在: {file_path}")]

            kb = get_kb()
            result = kb.upload_by_file(file_path)
            return [TextContent(type="text", text=result)]

        elif name == "list_knowledge":
            page = max(1, int(arguments.get("page", 1)))
            page_size = max(1, min(100, int(arguments.get("page_size", 10))))

            kb = get_kb()
            result = kb.list_knowledge(page=page, page_size=page_size)
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

        elif name == "update_knowledge":
            source_name = arguments.get("source_name", "").strip()
            content = arguments.get("content", "").strip()

            if not source_name:
                return [TextContent(type="text", text="[错误] source_name 不能为空")]
            if not content:
                return [TextContent(type="text", text="[错误] content 不能为空")]

            kb = get_kb()
            result = kb.update_knowledge(source_name, content)
            return [TextContent(type="text", text=result)]

        elif name == "update_knowledge_file":
            source_name = arguments.get("source_name", "").strip()
            file_path = arguments.get("file_path", "").strip()

            if not source_name:
                return [TextContent(type="text", text="[错误] source_name 不能为空")]
            if not file_path:
                return [TextContent(type="text", text="[错误] file_path 不能为空")]
            if not os.path.exists(file_path):
                return [TextContent(type="text", text=f"[错误] 文件不存在: {file_path}")]

            kb = get_kb()
            result = kb.update_knowledge_by_file(source_name, file_path)
            return [TextContent(type="text", text=result)]

        elif name == "delete_knowledge":
            source_name = arguments.get("source_name", "").strip()

            if not source_name:
                return [TextContent(type="text", text="[错误] source_name 不能为空")]

            kb = get_kb()
            result = kb.delete_knowledge(source_name)
            return [TextContent(type="text", text=result)]

        else:
            return [TextContent(type="text", text=f"[错误] 未知工具: {name}")]

    except Exception as e:
        return [TextContent(type="text", text=f"[异常] 执行 {name} 时出错: {e}")]


async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
