"""Agent 工具 —— 直接调用 RAG 服务（同进程），同时保留 MCP 远程调用能力"""

import os
import sys

from langchain_core.tools import tool

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---- 懒加载单例 ----
_rag_service = None
_kb_service = None


def _get_rag():
    global _rag_service
    if _rag_service is None:
        from RAG部分.rag import RagService

        _rag_service = RagService()
    return _rag_service


def _get_kb():
    global _kb_service
    if _kb_service is None:
        from RAG部分.knowledge_base import KnowledgeBaseService

        _kb_service = KnowledgeBaseService()
    return _kb_service


# ---- 7 个工具函数 ----


@tool(description="语义搜索知识库。当用户提出问题时，用用户的原始问题作为 query 来检索知识库中的相关内容。")
def search_knowledge(query: str) -> str:
    rag = _get_rag()
    tokens = list(rag.ask_stream(query, session_id="agent"))
    return "".join(tokens)


@tool(description="上传文本内容到知识库。content: 要上传的文本。filename: 文档名称，用于标识知识来源")
def upload_knowledge(content: str, filename: str) -> str:
    return _get_kb().upload_by_str(content, filename)


@tool(description="上传本地 txt 文件到知识库。file_path: 文件的绝对路径")
def upload_knowledge_file(file_path: str) -> str:
    return _get_kb().upload_by_file(file_path)


@tool(description="分页列出知识库中所有文档。page: 页码从1开始。page_size: 每页条数")
def list_knowledge(page: int = 1, page_size: int = 10) -> str:
    import json

    result = _get_kb().list_knowledge(page=page, page_size=page_size)
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool(description="更新知识库中的文档，用新文本替换旧内容。source_name: 文档名称。content: 新文本内容")
def update_knowledge(source_name: str, content: str) -> str:
    return _get_kb().update_knowledge(source_name, content)


@tool(description="用文件内容更新知识库文档。source_name: 文档名称。file_path: 新内容的文件路径")
def update_knowledge_file(source_name: str, file_path: str) -> str:
    return _get_kb().update_knowledge_by_file(source_name, file_path)


@tool(description="按文档名称删除知识库中的文档。source_name: 要删除的文档名称")
def delete_knowledge(source_name: str) -> str:
    return _get_kb().delete_knowledge(source_name)
