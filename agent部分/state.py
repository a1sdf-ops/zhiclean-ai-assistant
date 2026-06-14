"""Agent 状态定义 —— StateGraph 的 State 类型"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str
    tool_name: str
    tool_args: dict
    tool_result: str
    is_report: bool
    memory_context: str  # 长期记忆召回文本（注入 system prompt）
    user_query: str  # 原始用户输入（用于记忆提取）
