"""ReAct Agent —— 基于自定义 LangGraph StateGraph（specific 模式）

生产路径: execute_stream() → graph.stream() → SSE
测试路径: ainvoke() → graph.ainvoke() → 全量返回
"""

import asyncio
import os
import sys
from collections.abc import AsyncIterator, Iterator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage

from utils.logger_handler import logger


class ReactAgent:
    def __init__(self):
        from agent部分.graph import build_graph

        self.graph = build_graph()
        logger.info("ReactAgent 初始化完成 (LangGraph StateGraph)")

    def execute_stream(self, query: str) -> Iterator[str]:
        """流式执行，逐 token yield（生产路径）"""
        for chunk in self.graph.stream(
            {"messages": [HumanMessage(content=query)]},
            stream_mode="values",
        ):
            if "messages" not in chunk:
                continue
            msgs = chunk["messages"]
            if not msgs:
                continue
            last = msgs[-1]
            if hasattr(last, "content") and last.content:
                text = last.content
                if isinstance(text, str):
                    yield text

    async def ainvoke(self, query: str) -> str:
        """异步全量执行，返回完整回答（测试路径）"""
        result = await self.graph.ainvoke(
            {"messages": [HumanMessage(content=query)]},
        )
        msgs = result.get("messages", [])
        for m in reversed(msgs):
            if hasattr(m, "content") and m.content:
                return m.content
        return ""
