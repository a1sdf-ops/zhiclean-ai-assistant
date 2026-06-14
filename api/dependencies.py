"""FastAPI 依赖注入 —— 全局单例服务管理"""

import os
import sys
from functools import lru_cache

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


@lru_cache
def get_rag_service():
    from RAG部分.rag import RagService

    return RagService()


@lru_cache
def get_kb_service():
    from RAG部分.knowledge_base import KnowledgeBaseService

    return KnowledgeBaseService()


@lru_cache
def get_agent():
    from agent部分.react_agent import ReactAgent

    return ReactAgent()
