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

from agent.state import generate_trace_id
from utils.logger_handler import logger


class ReactAgent:
    def __init__(self):
        from agent.graph import build_graph

        self.graph = build_graph()
        logger.info("ReactAgent 初始化完成 (LangGraph StateGraph)")

    def _build_initial_state(self, query: str, session_id: str = "", tenant_id: str = "") -> dict:
        """构造初始 state，注入身份标识"""
        if not session_id:
            session_id = f"session_{generate_trace_id()}"
        if not tenant_id:
            tenant_id = f"tenant_{generate_trace_id()}"

        initial_state = {
            "messages": [HumanMessage(content=query)],
            "session_id": session_id,
            "tenant_id": tenant_id,
            "trace_id": generate_trace_id(),
            "intent": "",
            "tool_name": "",
            "tool_args": {},
            "tool_result": "",
            "is_report": False,
            "memory_context": "",
            "user_query": "",
        }

        logger.info("Agent 入口: tenant=%s session=%s trace=%s",
                     tenant_id, session_id, initial_state["trace_id"])
        return initial_state

    def execute_stream(self, query: str, session_id: str = "", tenant_id: str = "") -> Iterator[str]:
        """流式执行，逐 token yield（生产路径）"""
        initial_state = self._build_initial_state(query, session_id, tenant_id)
        for chunk in self.graph.stream(initial_state, stream_mode="values"):
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

    async def ainvoke(self, query: str, session_id: str = "", tenant_id: str = "") -> str:
        """异步全量执行，返回完整回答（测试路径）"""
        from agent.token_tracker import get_report

        initial_state = self._build_initial_state(query, session_id, tenant_id)
        result = await self.graph.ainvoke(initial_state)

        # 输出 Token 成本报告
        report = get_report()
        total = report.get("__total__", {}).get("total_tokens", 0)
        modules = {k: v["total_tokens"] for k, v in report.items() if k != "__total__"}
        logger.info("Token报告: 总计=%d tokens | %s (trace=%s)",
                     total, modules, initial_state.get("trace_id", ""))

        msgs = result.get("messages", [])
        for m in reversed(msgs):
            if hasattr(m, "content") and m.content:
                return m.content
        return ""
