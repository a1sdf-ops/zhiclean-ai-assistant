"""LangGraph StateGraph —— 手动编排 Agent 工作流（specific 模式）

节点: recall_memory → classify_intent → (条件边) → 工具节点 → log_tool_call → generate_final → save_memory → END
"""

import json
import os
import sys
from typing import Literal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from agent部分.agent_tools import (
    delete_knowledge,
    list_knowledge,
    search_knowledge,
    update_knowledge,
    update_knowledge_file,
    upload_knowledge,
    upload_knowledge_file,
)
from agent部分.mcp_client import get_mcp_manager
from agent部分.state import AgentState
from agent部分.tools.external_tools import (
    fetch_external_data,
    fill_context_for_report,
    get_current_month,
    get_user_id,
    get_user_location,
    get_weather,
)
from model.factory import create_chat_model
from utils.logger_handler import logger
from utils.memory import MemoryManager

_memory_manager = None


def get_memory() -> MemoryManager:
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


# ---------- 意图分类 ----------

INTENT_CLASSIFIER_PROMPT = """分析用户输入，返回 JSON 格式的意图分类结果。仅输出 JSON，不要其他内容。

意图类型（intent）：
- "weather": 查询天气
- "user_report": 查询用户使用记录、生成报告、月度分析
- "knowledge_search": 搜索知识库、提问、查询文档内容
- "knowledge_upload": 上传文档到知识库
- "knowledge_list": 列出知识库中的文档
- "knowledge_delete": 删除知识库文档
- "general": 一般对话、闲聊、不需要工具的问题

额外字段：
- "tool_args": 工具参数（天气需要 city，搜索需要 query，上传需要 content+filename 等）
- "is_report": true 仅当用户明确要求生成使用报告或月度分析

示例输出：
{"intent": "weather", "tool_args": {"city": "北京"}, "is_report": false}
{"intent": "knowledge_search", "tool_args": {"query": "春天相关内容"}, "is_report": false}
{"intent": "user_report", "tool_args": {}, "is_report": true}
"""

INTENT_LABELS = (
    "weather",
    "user_report",
    "knowledge_search",
    "knowledge_upload",
    "knowledge_list",
    "knowledge_delete",
    "general",
)

# ---------- 最终回答 Prompt ----------

FINAL_ANSWER_PROMPT = """你是知识库助手，严格基于提供的工具调用结果回答用户问题。

工具名称: {tool_name}
工具参数: {tool_args}
工具结果: {tool_result}

{memory_context}
规则：
- 严格基于工具返回的内容回答，不编造信息
- 检索结果为空时如实告知
- 用简洁专业的语言回答
- 如果工具结果包含用户数据，以结构化格式呈现
- 记忆上下文中有相关信息时可以提及"""

REPORT_PROMPT = """你是用户使用报告生成助手。基于用户行为数据生成专业的月度使用报告。

工具结果: {tool_result}

{memory_context}
规则：
- 以报告格式呈现：标题 → 摘要 → 详细数据 → 建议
- 数据部分使用结构化格式
- 给出针对性的优化建议
- 语气专业但不生硬
- 记忆上下文中有用户历史偏好时可以引用"""


# ---------- 节点函数 ----------


def _parse_intent_response(raw: str) -> dict:
    """解析 LLM 返回的意图 JSON"""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[:-3]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"intent": "general", "tool_args": {}, "is_report": False}


def recall_memory(state: AgentState) -> dict:
    """节点: 召回相关长期记忆，注入 memory_context"""
    last_msg = state["messages"][-1]
    query = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    memory = get_memory()
    context = memory.recall(query)

    if context:
        context = "用户历史记忆（可能相关，供参考）:\n" + context + "\n"

    return {
        "memory_context": context,
        "user_query": query,
    }


def classify_intent(state: AgentState) -> dict:
    """节点: 分类用户意图"""
    last_msg = state["messages"][-1]
    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    model = create_chat_model(temperature=0.0)
    response = model.invoke(
        [
            SystemMessage(content=INTENT_CLASSIFIER_PROMPT),
            HumanMessage(content=content),
        ]
    )

    parsed = _parse_intent_response(response.content)
    intent = parsed.get("intent", "general")
    if intent not in INTENT_LABELS:
        intent = "general"

    logger.info("意图分类: %s | tool_args=%s | is_report=%s", intent, parsed.get("tool_args"), parsed.get("is_report"))

    return {
        "intent": intent,
        "tool_args": parsed.get("tool_args", {}),
        "is_report": parsed.get("is_report", False),
        "user_query": content,
    }


def handle_weather(state: AgentState) -> dict:
    """节点: 天气查询 —— 通过 MCP 协议调用 Go Weather Server"""
    city = state.get("tool_args", {}).get("city", "北京")

    mcp = get_mcp_manager()
    conn = mcp.get_connection("weather")
    if conn is not None:
        try:
            result = mcp.call_tool("weather", "get_weather", {"city": city})
            if not result.startswith("[错误]") and not result.startswith("[MCP错误]"):
                logger.info("MCP天气查询完成: city=%s (Go server)", city)
                return {"tool_name": "get_weather", "tool_result": result}
        except Exception as e:
            logger.warning("MCP天气调用失败，回退到Python: %s", e)

    result = get_weather(city)
    logger.info("天气查询完成: city=%s (Python fallback)", city)
    return {"tool_name": "get_weather", "tool_result": result}


def handle_user_report(state: AgentState) -> dict:
    """节点: 用户报告（串联调用多个工具）"""
    uid = get_user_id()
    month = get_current_month()
    data = fetch_external_data(uid, month)
    fill_context_for_report()

    result = json.dumps(
        {
            "用户ID": uid,
            "月份": month,
            "使用数据": json.loads(data) if data and data != "{}" else "无数据",
        },
        ensure_ascii=False,
        indent=2,
    )

    logger.info("用户报告生成: uid=%s month=%s", uid, month)
    return {
        "tool_name": "user_report",
        "tool_result": result,
        "is_report": True,
    }


def handle_knowledge_search(state: AgentState) -> dict:
    """节点: 知识库搜索"""
    query = state.get("tool_args", {}).get("query", "")
    if not query:
        last_msg = state["messages"][-1]
        query = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    result = search_knowledge.invoke({"query": query})
    logger.info("知识库搜索完成: query=%s", query[:50])
    return {"tool_name": "search_knowledge", "tool_result": result}


def handle_knowledge_upload(state: AgentState) -> dict:
    """节点: 知识库上传"""
    args = state.get("tool_args", {})
    content = args.get("content", "")
    filename = args.get("filename", "unknown.txt")

    if content:
        result = upload_knowledge.invoke({"content": content, "filename": filename})
    else:
        file_path = args.get("file_path", "")
        result = upload_knowledge_file.invoke({"file_path": file_path}) if file_path else "未提供上传内容"

    logger.info("知识库上传完成: filename=%s", filename)
    return {"tool_name": "knowledge_upload", "tool_result": result}


def handle_knowledge_list(state: AgentState) -> dict:
    """节点: 列出知识库文档"""
    page = state.get("tool_args", {}).get("page", 1)
    page_size = state.get("tool_args", {}).get("page_size", 10)
    result = list_knowledge.invoke({"page": page, "page_size": page_size})
    logger.info("知识库列表查询: page=%s", page)
    return {"tool_name": "list_knowledge", "tool_result": result}


def handle_knowledge_delete(state: AgentState) -> dict:
    """节点: 删除知识库文档"""
    source_name = state.get("tool_args", {}).get("source_name", "")
    if not source_name:
        return {"tool_name": "delete_knowledge", "tool_result": "未指定要删除的文档名称"}
    result = delete_knowledge.invoke({"source_name": source_name})
    logger.info("知识库删除: source=%s", source_name)
    return {"tool_name": "delete_knowledge", "tool_result": result}


def handle_general(state: AgentState) -> dict:
    """节点: 一般对话（无工具调用，直接回答）"""
    return {"tool_name": "general", "tool_result": ""}


def log_tool_call(state: AgentState) -> dict:
    """节点: 记录工具调用（原 monitor_tool 中间件逻辑）"""
    tool_name = state.get("tool_name", "unknown")
    tool_result = state.get("tool_result", "")
    result_preview = tool_result[:120] if tool_result else "(空)"
    logger.info("工具调用完成: %s | 结果预览: %s", tool_name, result_preview)
    return {}


def generate_final_answer(state: AgentState) -> dict:
    """节点: 生成最终回答（支持报告模式+记忆注入）"""
    is_report = state.get("is_report", False)
    tool_name = state.get("tool_name", "general")
    tool_result = state.get("tool_result", "")
    tool_args = state.get("tool_args", {})
    memory_context = state.get("memory_context", "")

    if is_report:
        prompt_template = REPORT_PROMPT
    else:
        prompt_template = FINAL_ANSWER_PROMPT

    system_msg = prompt_template.format(
        tool_name=tool_name,
        tool_args=json.dumps(tool_args, ensure_ascii=False),
        tool_result=tool_result or "(无工具调用，直接回答用户问题)",
        memory_context=memory_context or "",
    )

    model = create_chat_model()
    response = model.invoke(
        [
            SystemMessage(content=system_msg),
            *state["messages"],
        ]
    )

    logger.info("最终回答生成完成: is_report=%s has_memory=%s", is_report, bool(memory_context))
    return {"messages": [response]}


def save_memory(state: AgentState) -> dict:
    """节点: 从对话中提取关键事实，持久化到长期记忆"""
    user_query = state.get("user_query", "")
    last_msgs = state.get("messages", [])

    assistant_msg = ""
    if last_msgs:
        last = last_msgs[-1]
        assistant_msg = last.content if hasattr(last, "content") else str(last)

    if user_query and assistant_msg:
        memory = get_memory()
        facts = memory.save(user_query, assistant_msg)
        if facts:
            logger.info("记忆已提取: %d 条事实", len(facts))

    return {}


# ---------- 条件边 ----------


def route_by_intent(
    state: AgentState,
) -> Literal[
    "handle_weather",
    "handle_user_report",
    "handle_knowledge_search",
    "handle_knowledge_upload",
    "handle_knowledge_list",
    "handle_knowledge_delete",
    "handle_general",
]:
    intent = state.get("intent", "general")
    node_map = {
        "weather": "handle_weather",
        "user_report": "handle_user_report",
        "knowledge_search": "handle_knowledge_search",
        "knowledge_upload": "handle_knowledge_upload",
        "knowledge_list": "handle_knowledge_list",
        "knowledge_delete": "handle_knowledge_delete",
        "general": "handle_general",
    }
    return node_map.get(intent, "handle_general")


# ---------- 构建图 ----------


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("recall_memory", recall_memory)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("handle_weather", handle_weather)
    graph.add_node("handle_user_report", handle_user_report)
    graph.add_node("handle_knowledge_search", handle_knowledge_search)
    graph.add_node("handle_knowledge_upload", handle_knowledge_upload)
    graph.add_node("handle_knowledge_list", handle_knowledge_list)
    graph.add_node("handle_knowledge_delete", handle_knowledge_delete)
    graph.add_node("handle_general", handle_general)
    graph.add_node("log_tool_call", log_tool_call)
    graph.add_node("generate_final_answer", generate_final_answer)
    graph.add_node("save_memory", save_memory)

    # 入口 → 记忆召回 → 意图分类
    graph.set_entry_point("recall_memory")
    graph.add_edge("recall_memory", "classify_intent")

    # 条件边: 意图分类 → 对应处理节点
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "handle_weather": "handle_weather",
            "handle_user_report": "handle_user_report",
            "handle_knowledge_search": "handle_knowledge_search",
            "handle_knowledge_upload": "handle_knowledge_upload",
            "handle_knowledge_list": "handle_knowledge_list",
            "handle_knowledge_delete": "handle_knowledge_delete",
            "handle_general": "handle_general",
        },
    )

    # 所有工具节点 → 日志节点 → 最终回答 → 记忆保存
    for node in [
        "handle_weather",
        "handle_user_report",
        "handle_knowledge_search",
        "handle_knowledge_upload",
        "handle_knowledge_list",
        "handle_knowledge_delete",
        "handle_general",
    ]:
        graph.add_edge(node, "log_tool_call")

    graph.add_edge("log_tool_call", "generate_final_answer")
    graph.add_edge("generate_final_answer", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile()
