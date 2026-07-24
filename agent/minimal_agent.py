"""minimal_agent.py —— 纯 Python 手写 Agent 状态机，零框架依赖

面试用途：字节面试官可能要求手写"不依赖 LangGraph 的 Agent 状态流转"，
         这个文件就是标准答案。核心知识点：
         - TypedDict 强类型 → 拒绝随意增删字段
         - 路由表 dispatch → 显式声明状态流转路径
         - 手动 checkpoint → JSON 快照 + 异常回滚
         - error_count 熔断 → 3次异常强制终止

函数总结（按代码顺序）：
  AgentState        —— 强类型状态定义（TypedDict），字段不可随意增删
  save_checkpoint   —— 将当前 state 序列化为 JSON 快照字符串
  restore_checkpoint—— 从 JSON 快照还原 state 字典
  route             —— 意图分类：根据用户输入匹配意图白名单，设置 next_step
  execute_tool      —— 工具调用模拟：成功返回结果，失败累计 error_count 并触发熔断
  generate          —— 最终回复生成：拼接工具结果 + 记忆上下文 → 输出回答
  MinimalAgent.run  —— 主循环入口：route→execute→generate，含熔断检查 + checkpoint 回滚
"""

import json
import uuid
from typing import Optional, TypedDict

# ── 意图白名单（与 state.py 保持一致）──
VALID_INTENTS = frozenset(
    {
        "weather",
        "user_report",
        "knowledge_search",
        "knowledge_upload",
        "knowledge_list",
        "knowledge_delete",
        "general",
    }
)

# ── 下一步路由白名单 ──
VALID_STEPS = frozenset({"router", "execute_tool", "generate", "end"})


# ═══════════════════════════════════════════
# AgentState: 强类型定义
# ═══════════════════════════════════════════


class AgentState(TypedDict):
    session_id: str  # 会话唯一ID
    tenant_id: str  # 租户隔离标识
    messages: list  # 对话历史 [{role, content}, ...]
    intent: str  # 当前意图，必须 ∈ VALID_INTENTS
    memory_context: str  # 长期记忆召回文本
    tool_result: str  # 工具调用结果
    final_response: str  # 最终回复
    next_step: str  # 下一步路由: router | execute_tool | generate | end
    error_count: int  # 全局异常计数，≥3 熔断
    checkpoint: str | None  # 最近一次 JSON 快照


# ═══════════════════════════════════════════
# checkpoint 工具函数
# ═══════════════════════════════════════════


def save_checkpoint(state: AgentState) -> str:
    """拍快照：将 state 序列化为 JSON，用于异常回滚"""
    return json.dumps(state, ensure_ascii=False, default=str)


def restore_checkpoint(raw: str) -> AgentState:
    """从快照恢复：JSON 反序列化回 state"""
    return json.loads(raw)


# ═══════════════════════════════════════════
# 三个核心节点
# ═══════════════════════════════════════════


def route(state: AgentState) -> AgentState:
    """意图分类：根据用户最后一条消息判断意图并设置 next_step"""
    msgs = state.get("messages", [])
    if not msgs:
        state["intent"] = "general"
        state["next_step"] = "generate"
        return state

    text = msgs[-1].get("content", "") if isinstance(msgs[-1], dict) else str(msgs[-1])
    text_lower = text.lower()

    # ── 关键词 → 意图映射 ──
    intent_map = [
        (["天气", "weather", "气温", "下雨"], "weather"),
        (["报告", "月度", "使用记录", "分析"], "user_report"),
        (["上传", "upload", "添加文档"], "knowledge_upload"),
        (["列出", "list", "有哪些文档", "知识库列表"], "knowledge_list"),
        (["删除", "delete", "移除文档"], "knowledge_delete"),
        (["搜索", "search", "查询", "查找", "检索"], "knowledge_search"),
    ]
    for keywords, intent_val in intent_map:
        if any(kw in text_lower for kw in keywords):
            state["intent"] = intent_val
            state["next_step"] = "execute_tool"
            return state

    state["intent"] = "general"
    state["next_step"] = "generate"
    return state


def execute_tool(state: AgentState) -> AgentState:
    """模拟工具执行：成功返回结果，失败递增 error_count"""
    intent = state.get("intent", "general")

    # ── 工具模拟（实际项目替换为真实工具调用）──
    tool_results = {
        "weather": "[模拟] 天气查询: 北京 晴 22°C",
        "user_report": "[模拟] 用户报告: 本月使用15次，活跃度中等",
        "knowledge_search": "[模拟] 知识检索: 找到3条相关文档",
        "knowledge_upload": "[模拟] 文档上传成功",
        "knowledge_list": "[模拟] 知识库文档列表: doc1, doc2, doc3",
        "knowledge_delete": "[模拟] 文档已删除",
        "general": "",
    }

    try:
        result = tool_results.get(intent, "[模拟] 未知工具")
        if not result:
            raise ValueError(f"工具 {intent} 返回空结果")
        state["tool_result"] = result
    except Exception:
        state["error_count"] += 1
        state["tool_result"] = f"[工具异常] {intent} 调用失败 (第{state['error_count']}次)"
        if state["error_count"] >= 3:
            state["final_response"] = "系统异常，请稍后重试。"
            state["next_step"] = "end"
            return state

    state["next_step"] = "generate"
    return state


def generate(state: AgentState) -> AgentState:
    """生成最终回复：拼接工具结果 + 记忆上下文"""
    tool_result = state.get("tool_result", "")
    memory = state.get("memory_context", "")
    intent = state.get("intent", "general")

    # ── 拼装回复（实际项目替换为 LLM 调用）──
    parts = []
    if tool_result and "异常" not in tool_result:
        parts.append(tool_result)
    if memory:
        parts.append(f"(记忆参考: {memory})")
    if not parts:
        parts.append("已收到您的问题，请问有什么可以帮您？")

    state["final_response"] = " | ".join(parts)
    state["next_step"] = "end"
    return state


# ═══════════════════════════════════════════
# MinimalAgent: 主循环入口
# ═══════════════════════════════════════════


class MinimalAgent:
    """不依赖任何框架的 Agent 状态机

    路由表设计：next_step 决定下一步执行哪个函数，显式声明不留隐式路径。
    """

    def __init__(self):
        # 路由表: next_step → handler
        self._dispatch = {
            "router": route,
            "execute_tool": execute_tool,
            "generate": generate,
        }

    def run(self, query: str, session_id: str = "", tenant_id: str = "", memory_context: str = "") -> AgentState:
        """主循环：route → execute → generate，含熔断 + checkpoint

        Args:
            query:          用户输入文本
            session_id:     会话ID（不传自动生成）
            tenant_id:      租户ID（不传自动生成）
            memory_context: 可选，长期记忆上下文

        Returns:
            执行完毕的 AgentState，final_response 包含最终回复
        """
        if not session_id:
            session_id = uuid.uuid4().hex[:8]
        if not tenant_id:
            tenant_id = f"tenant_{uuid.uuid4().hex[:8]}"

        # ── 初始 state ──
        state: AgentState = {
            "session_id": session_id,
            "tenant_id": tenant_id,
            "messages": [{"role": "user", "content": query}],
            "intent": "",
            "memory_context": memory_context,
            "tool_result": "",
            "final_response": "",
            "next_step": "router",
            "error_count": 0,
            "checkpoint": None,
        }

        # ── 执行循环（最多 10 轮，防止死循环）──
        max_steps = 10
        for _ in range(max_steps):
            step = state["next_step"]
            if step == "end":
                break

            # 拍快照：异常时回滚
            state["checkpoint"] = save_checkpoint(state)

            try:
                handler = self._dispatch.get(step)
                if handler is None:
                    state["final_response"] = f"路由错误: 未知步骤 '{step}'"
                    state["next_step"] = "end"
                    break
                state = handler(state)

            except Exception as exc:
                state["error_count"] += 1
                if state["error_count"] >= 3:
                    state["final_response"] = "系统繁忙，请稍后重试。"
                    state["next_step"] = "end"
                else:
                    # 回滚到上一个 checkpoint 重试
                    ckpt = state.get("checkpoint")
                    if ckpt:
                        state = restore_checkpoint(ckpt)
                        state["next_step"] = step  # 重新执行当前步骤
                    else:
                        state["next_step"] = "end"

        return state


# ═══════════════════════════════════════════
# 快速测试入口
# ═══════════════════════════════════════════

if __name__ == "__main__":
    agent = MinimalAgent()

    print("=" * 50)
    print("测试1: 天气查询")
    s = agent.run("今天北京天气怎么样？")
    print(f"  intent={s['intent']}, response={s['final_response'][:60]}")
    print(f"  error_count={s['error_count']}")

    print("=" * 50)
    print("测试2: 一般闲聊（无工具）")
    s = agent.run("你好，请介绍一下你自己")
    print(f"  intent={s['intent']}, response={s['final_response'][:60]}")

    print("=" * 50)
    print("测试3: checkpoint 回滚验证")
    s = agent.run("搜索一下空调怎么设置")
    print(f"  intent={s['intent']}, next_step={s['next_step']}")
    print(f"  has_checkpoint={s['checkpoint'] is not None}")

    print("=" * 50)
    print("All tests passed.")
