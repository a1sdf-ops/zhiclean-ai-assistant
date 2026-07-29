"""LangGraph StateGraph —— 手动编排 Agent 工作流（specific 模式）

节点: recall_memory → classify_intent → (条件边) → 工具节点 → log_tool_call → generate_final → save_memory → END
"""

import json
import os
import sys
import time
from typing import Literal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

import config
from agent.agent_tools import (
    delete_knowledge,
    list_knowledge,
    search_knowledge,
    upload_knowledge,
    upload_knowledge_file,
)
from agent.mcp_client import get_mcp_manager
from agent.state import AgentState
from agent.token_tracker import estimate_tokens, get_tracker
from agent.tools.external_tools import (
    fetch_external_data,
    fill_context_for_report,
    get_current_month,
    get_user_id,
    get_weather,
)
from model.factory import create_chat_model
from utils.logger_handler import logger, set_trace_id
from utils.memory import MemoryManager
from utils.profile import ProfileManager

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
- 检索结果提示"暂未收录专项文档"时，可结合产品通用知识给出排查建议，不强调"知识库未收录"
- 用简洁专业的语言回答
- 如果工具结果包含用户数据，以结构化格式呈现
- 记忆上下文中有相关信息时可以引用"""

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


# 耗材推荐更换周期（月），来自保养维护指南
_CONSUMABLE_CYCLE = {
    "hepa_filter": (1, 2),
    "side_brush": (3, 4),
    "main_brush": (6, 8),
    "mop_pad": (3, 3),
    "dust_bag": (2, 3),
}
_CONSUMABLE_LABEL = {
    "hepa_filter": "HEPA滤网",
    "side_brush": "边刷",
    "main_brush": "主刷",
    "mop_pad": "拖布",
    "dust_bag": "尘袋",
}


def _format_profile(profile: dict) -> str:
    """将用户画像格式化为 prompt 上下文字符串"""
    if not profile:
        return ""
    lines = []

    # ── 当前城市 ──
    current_loc = profile.get("current_location", "")
    locations = profile.get("locations", [])
    if current_loc:
        history_note = f"（历史: {', '.join(locations)}）" if len(locations) > 1 else ""
        lines.append(f"  当前城市: {current_loc} {history_note}".rstrip())
    elif locations:
        lines.append(f"  当前城市: {locations[-1]}")

    # ── 设备 ──
    devices = profile.get("devices", [])
    if devices:
        lines.append("  设备:")
        for d in devices:
            model = d.get("model", "")
            purchased = d.get("purchased", "")
            parts = [f"    • {model}"]
            if purchased:
                parts.append(f"购买: {purchased}")
            lines.append(" | ".join(parts))

            # 使用习惯
            usage = d.get("usage", {})
            if usage:
                u = []
                if usage.get("frequency"):
                    u.append(f"频率: {usage['frequency']}")
                if usage.get("primary_area"):
                    u.append(f"区域: {usage['primary_area']}")
                if usage.get("floor_type"):
                    u.append(f"地面: {usage['floor_type']}")
                if usage.get("has_pets") is True:
                    u.append("养宠物")
                if u:
                    lines.append(f"      使用习惯: {' | '.join(u)}")

            # 已知问题（结构化）
            issues = d.get("issues", [])
            if issues:
                lines.append("      已知问题:")
                for iss in issues:
                    if isinstance(iss, str):
                        lines.append(f"        - {iss}")
                    else:
                        problem = iss.get("problem", "")
                        status = iss.get("status", "未解决")
                        attempted = iss.get("attempted_solutions", [])
                        s = f"        - {problem} [{status}]"
                        if attempted:
                            s += f"  已尝试: {', '.join(attempted)}"
                        lines.append(s)

            # 耗材
            consumables = d.get("consumables", {})
            if consumables:
                lines.append("      耗材:")
                for key, val in consumables.items():
                    label = _CONSUMABLE_LABEL.get(key, key)
                    last = val.get("last_replaced", "?")
                    cycle = _CONSUMABLE_CYCLE.get(key)
                    if cycle:
                        lines.append(f"        - {label}: 上次更换 {last}（建议每{cycle[0]}-{cycle[1]}个月）")
                    else:
                        lines.append(f"        - {label}: 上次更换 {last}")

    # ── 偏好 ──
    prefs = profile.get("preferences", {})
    if prefs:
        p = []
        tech = prefs.get("tech_level", "")
        if tech:
            p.append(f"技术水平: {tech}")
        style = prefs.get("reply_style", "")
        if style:
            p.append(f"回复偏好: {style}")
        other = {k: v for k, v in prefs.items() if k not in ("tech_level", "reply_style")}
        for k, v in other.items():
            p.append(f"{k}: {v}")
        if p:
            lines.append(f"  偏好: {' | '.join(p)}")

    # ── 购买意向 ──
    purchase_intent = profile.get("purchase_intent", [])
    if purchase_intent:
        items = [f"{pi.get('product', '?')}(意向:{pi.get('level', '?')})" for pi in purchase_intent]
        lines.append(f"  购买意向: {', '.join(items)}")

    # ── 售后记录 ──
    service_history = profile.get("service_history", [])
    if service_history:
        lines.append("  售后记录:")
        for sh in service_history[-3:]:
            lines.append(f"    - [{sh.get('date', '?')}] {sh.get('type', '?')}: {sh.get('description', '')}")

    # ── 历史提问 ──
    question_history = profile.get("question_history", [])
    if question_history:
        total = len(question_history)
        resolved_count = sum(1 for q in question_history if q.get("resolved"))
        unresolved = [q for q in question_history if not q.get("resolved")]
        # 统计摘要
        lines.append(f"  历史提问: {total}次, 已解决{resolved_count}/{total}")
        if unresolved:
            unresolved_problems = [q.get("problem") or q.get("query_summary", "?") for q in unresolved]
            lines.append(f"  未解决问题: {', '.join(unresolved_problems[:5])}")
        # 最近 5 条
        recent = question_history[-5:]
        lines.append("  最近提问:")
        for q in recent:
            qdate = q.get("date", "?")[:10]
            cat = q.get("category", "?")
            summary = q.get("query_summary") or q.get("problem") or "?"
            status = "✓" if q.get("resolved") else "✗"
            lines.append(f"    [{qdate}] [{status}] {cat}: {summary}")

    return "[用户画像]\n" + "\n".join(lines) if lines else ""


def recall_memory(state: AgentState) -> dict:
    """用户画像 + ChromaDB 语义记忆，两层互补注入 memory_context"""
    set_trace_id(state.get("trace_id", "-"))
    last_msg = state["messages"][-1]
    query = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    tenant_id = state.get("tenant_id", "default")
    session_id = state.get("session_id", "default")

    context_parts = []
    profile_ctx = ""

    # ── 1. 用户画像（JSON 文件，全量加载，O(1)）──
    try:
        profile_mgr = ProfileManager()
        profile = profile_mgr.load(tenant_id)
        profile_ctx = _format_profile(profile)
        if profile_ctx:
            context_parts.append(profile_ctx)
    except Exception as e:
        logger.warning("画像加载失败: %s", e)

    # ── 2. ChromaDB 语义记忆（向量召回）──
    memory = get_memory()
    semantic_ctx = memory.recall(query, session_id=session_id, tenant_id=tenant_id)
    if semantic_ctx:
        context_parts.append(f"[语义记忆]\n{semantic_ctx}")

    context = "\n".join(context_parts) if context_parts else ""

    logger.info(
        "记忆召回: profile=%s semantic=%s trace=%s",
        "有" if profile_ctx else "无",
        "有" if semantic_ctx else "无",
        state.get("trace_id", ""),
    )

    return {
        "memory_context": context,
        "user_query": query,
    }


def classify_intent(state: AgentState) -> dict:
    """节点: 分类用户意图"""
    last_msg = state["messages"][-1]
    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    model = create_chat_model(temperature=0.0, model_name=config.INTENT_MODEL)
    t0 = time.time()
    response = model.invoke(
        [
            SystemMessage(content=INTENT_CLASSIFIER_PROMPT),
            HumanMessage(content=content),
        ]
    )
    latency = (time.time() - t0) * 1000
    # Token 埋点: 意图分类
    get_tracker().record(
        "llm_intent_classifier",
        input_tokens=estimate_tokens(INTENT_CLASSIFIER_PROMPT + content),
        output_tokens=estimate_tokens(response),
        latency_ms=latency,
    )

    parsed = _parse_intent_response(response.content)
    intent = parsed.get("intent", "general")
    if intent not in INTENT_LABELS:
        intent = "general"

    logger.info(
        "意图分类: %s | tool_args=%s | is_report=%s | latency=%.0fms",
        intent,
        parsed.get("tool_args"),
        parsed.get("is_report"),
        latency,
    )

    return {
        "intent": intent,
        "tool_args": parsed.get("tool_args", {}),
        "is_report": parsed.get("is_report", False),
        "user_query": content,
    }


def handle_weather(state: AgentState) -> dict:
    """节点: 天气查询 —— 通过 MCP 协议调用 Go Weather Server"""
    city = state.get("tool_args", {}).get("city", "北京")

    # 解析相对城市引用（"当前城市"、"这里"等）→ 从画像取 current_location
    relative_cities = {"当前城市", "这里", "本地", "我所在的城市", "我在的城市", "我这"}
    if city in relative_cities:
        try:
            profile_mgr = ProfileManager()
            profile = profile_mgr.load(state.get("tenant_id", "default"))
            current = profile.get("current_location", "")
            if current:
                logger.info("城市引用解析: '%s' → '%s'", city, current)
                city = current
        except Exception as e:
            logger.warning("城市解析失败: %s", e)

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
    """节点: 知识库搜索（检索失败时用原始提问回退一次）"""
    query = state.get("tool_args", {}).get("query", "")
    last_msg = state["messages"][-1]
    original_query = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    if not query:
        query = original_query

    if not config.RAG_LLM_ENABLED:
        # A1 优化：handler 只做检索，LLM 留给 generate_final_answer
        try:
            from rag.rag import RagService

            rag = RagService()
            docs = rag.retrieve_documents(query, top_k=getattr(config, "RAG_RETRIEVE_TOP_K", 5))
            # 第一次检索为空且 query 是提取的短词时，用原始提问回退一次
            if not docs and query != original_query:
                logger.info("首次检索为空(query=%s)，用原始提问回退", query[:50])
                docs = rag.retrieve_documents(original_query.strip(), top_k=getattr(config, "RAG_RETRIEVE_TOP_K", 5))
            if not docs:
                return {
                    "tool_name": "search_knowledge",
                    "tool_result": "知识库中暂未收录该问题的专项文档，请结合产品通用知识回答用户",
                }
            return {"tool_name": "search_knowledge", "tool_result": f"[检索到以下参考文档，请基于此回答]\n\n{docs}"}
        except Exception as e:
            return {"tool_name": "search_knowledge", "tool_result": str(e)}

    try:
        result = search_knowledge.invoke({"query": query})
    except Exception as e:
        return {"tool_name": "search_knowledge", "tool_result": str(e)}
    return {"tool_name": "search_knowledge", "tool_result": result}


def handle_knowledge_upload(state: AgentState) -> dict:
    """节点: 知识库上传"""
    args = state.get("tool_args", {})
    content = args.get("content", "")
    filename = args.get("filename", "unknown.txt")

    try:
        if content:
            result = upload_knowledge.invoke({"content": content, "filename": filename})
        else:
            file_path = args.get("file_path", "")
            if file_path:
                result = upload_knowledge_file.invoke({"file_path": file_path})
            else:
                result = "未提供上传内容"
    except Exception as e:
        return {"tool_name": "knowledge_upload", "tool_result": str(e)}
    return {"tool_name": "knowledge_upload", "tool_result": result}


def handle_knowledge_list(state: AgentState) -> dict:
    """节点: 列出知识库文档"""
    page = state.get("tool_args", {}).get("page", 1)
    page_size = state.get("tool_args", {}).get("page_size", 10)

    try:
        result = list_knowledge.invoke({"page": page, "page_size": page_size})
    except Exception as e:
        return {"tool_name": "list_knowledge", "tool_result": str(e)}
    return {"tool_name": "list_knowledge", "tool_result": result}


def handle_knowledge_delete(state: AgentState) -> dict:
    """节点: 删除知识库文档"""
    # 兼容 LLM 可能输出的两种参数名
    tool_args = state.get("tool_args", {})
    source_name = tool_args.get("source_name", "") or tool_args.get("filename", "")

    if not source_name:
        return {"tool_name": "delete_knowledge", "tool_result": "未指定要删除的文档名称"}

    try:
        result = delete_knowledge.invoke({"source_name": source_name})
    except Exception as e:
        return {"tool_name": "delete_knowledge", "tool_result": str(e)}
    return {"tool_name": "delete_knowledge", "tool_result": result}


def handle_general(state: AgentState) -> dict:
    """节点: 一般对话（无工具调用，直接回答）"""
    return {"tool_name": "general", "tool_result": ""}


def log_tool_call(state: AgentState) -> dict:
    """节点: 记录工具调用"""
    tool_name = state.get("tool_name", "unknown")
    tool_result = state.get("tool_result", "")
    trace_id = state.get("trace_id", "unknown")

    result_preview = tool_result[:120] if tool_result else "(空)"
    logger.info("工具调用完成: %s | trace=%s | 结果=%s", tool_name, trace_id, result_preview)

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
    t0 = time.time()

    if config.STREAM_MODE == "stream":
        # 真流式: 逐 token 产出，astream_events 捕获每个 chunk
        chunks = []
        full_content = []
        for chunk in model.stream(
            [
                SystemMessage(content=system_msg),
                *state["messages"],
            ]
        ):
            chunks.append(chunk)
            if chunk.content:
                full_content.append(chunk.content)
        # 合并为完整 AIMessage（Token 埋点需要）
        if chunks:
            response = chunks[0]
            for c in chunks[1:]:
                response += c
        else:
            response = chunks[0] if chunks else None
    else:
        # 原有路径: invoke 全量返回
        response = model.invoke(
            [
                SystemMessage(content=system_msg),
                *state["messages"],
            ]
        )

    latency = (time.time() - t0) * 1000
    # Token 埋点: 最终回答生成
    input_msgs = (
        system_msg
        + " "
        + " ".join(m.content if hasattr(m, "content") else str(m) for m in state.get("messages", [])[:5])
    )
    get_tracker().record(
        "llm_generation",
        input_tokens=estimate_tokens(input_msgs),
        output_tokens=estimate_tokens(response),
        latency_ms=latency,
    )

    logger.info(
        "最终回答生成完成: is_report=%s has_memory=%s | latency=%.0fms", is_report, bool(memory_context), latency
    )
    return {"messages": [response]}


def save_memory(state: AgentState) -> dict:
    """节点: 三层持久化记忆存储

    1. Redis Hash   → 短期会话上下文（最近N轮，72h TTL）
    2. ChromaDB     → 长期语义记忆（LLM提取事实 → Embedding → 向量检索）原有逻辑
    3. Redis Sorted Set → 长期偏好记忆（带权重衰减，用于快速TopK查询）
    """
    user_query = state.get("user_query", "")
    session_id = state.get("session_id", "default")
    tenant_id = state.get("tenant_id", "default")

    last_msgs = state.get("messages", [])
    assistant_msg = ""
    if last_msgs:
        last = last_msgs[-1]
        assistant_msg = last.content if hasattr(last, "content") else str(last)

    saved_facts = []
    if user_query and assistant_msg:
        try:
            memory = get_memory()
            saved_facts, profile_update = memory.save(
                user_query,
                assistant_msg,
                session_id=session_id,
                tenant_id=tenant_id,
            )
        except Exception as e:
            logger.warning("ChromaDB 记忆存储失败: %s", e)

    # ── 画像增量合并 ──
    if profile_update:
        try:
            profile_mgr = ProfileManager()
            profile_mgr.merge(tenant_id, profile_update)
        except Exception as e:
            logger.warning("画像更新失败: %s", e)

    if saved_facts:
        logger.info("记忆已提取: %d 条事实 (session=%s)", len(saved_facts), session_id)

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
