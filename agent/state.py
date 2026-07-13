"""Agent 状态定义 —— 强类型 AgentState + TraceID 生成

字段分层说明：
  [身份层] session_id / tenant_id / trace_id —— 会话隔离 & 全链路追踪
  [路由层] intent —— 意图标识，由 classify_intent 节点写入
  [工具层] tool_name / tool_args / tool_result —— 工具调用参数与结果
  [检索层] memory_context / user_query —— 记忆召回与用户输入
  [生成层] is_report —— 是否为报告模式，切换最终回答 prompt 模板
"""

import uuid
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

# ── 意图白名单 ──────────────────────────────
VALID_INTENTS = frozenset({
    "weather",
    "user_report",
    "knowledge_search",
    "knowledge_upload",
    "knowledge_list",
    "knowledge_delete",
    "general",
})


def generate_trace_id() -> str:
    """生成12位十六进制 TraceID，一次 invoke 一个"""
    return uuid.uuid4().hex[:12]


# ── AgentState: 强类型定义 ──────────────────

class AgentState(TypedDict):
    # 身份层
    session_id: str
    tenant_id: str
    trace_id: str

    # 对话
    messages: Annotated[list, add_messages]

    # 路由
    intent: str

    # 工具调用
    tool_name: str
    tool_args: dict
    tool_result: str

    # 检索 & 记忆
    memory_context: str
    user_query: str

    # 生成
    is_report: bool
