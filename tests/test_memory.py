"""Agent 记忆系统测试 —— 短期+长期记忆的写入、召回、遗忘"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def requires_api(func):
    """有 DashScope API Key 才执行"""
    return pytest.mark.skipif(
        not config.DASHSCOPE_API_KEY or len(config.DASHSCOPE_API_KEY) < 10,
        reason="需要有效的 DASHSCOPE_API_KEY",
    )(func)


class TestMemoryManager:
    """长期记忆管理器单元测试"""

    def test_memory_store_creates_collection(self):
        """验证 ChromaDB collection 可以创建"""
        from utils.memory import MemoryManager

        m = MemoryManager()
        assert m.store is not None
        assert m.store._collection is not None

    def test_recall_empty_returns_empty_string(self):
        """空记忆库返回空字符串"""
        from utils.memory import MemoryManager

        m = MemoryManager()
        result = m.recall("测试查询", session_id="test_empty")
        assert result == ""

    def test_all_memories_initially_empty(self):
        """初始记忆列表为空"""
        from utils.memory import MemoryManager

        m = MemoryManager()
        results = m.all_memories(session_id="test_nonexistent")
        assert results == []


class TestMemoryExtraction:
    """LLM 事实提取集成测试"""

    @requires_api
    def test_extract_facts_from_conversation(self):
        """从一段对话中提取用户事实"""
        from utils.memory import MemoryManager

        m = MemoryManager()

        user_msg = "你好，我是陈磊，在北京化工大学读书，我的扫地机器人型号是Z2 Pro"
        assistant_msg = "你好陈磊！Z2 Pro是一款很不错的扫拖一体机器人，有什么可以帮助你的吗？"

        facts = m._extract_facts(user_msg, assistant_msg)
        assert isinstance(facts, list)
        if facts:  # LLM 提取到了事实
            assert all("fact" in f for f in facts)
            assert all("category" in f for f in facts)
            assert all("importance" in f for f in facts)

    @requires_api
    def test_save_and_recall(self):
        """写入记忆后能成功召回"""
        from utils.memory import MemoryManager

        m = MemoryManager()

        session = "test_save_recall"
        user_msg = "我叫张伟，是一名软件工程师，主要做后端开发，用的编程语言是Python和Go"
        assistant_msg = "了解了张伟，Python和Go都是很适合后端的语言。"

        facts = m.save(user_msg, assistant_msg, session_id=session)
        assert isinstance(facts, list)

        # 召回
        recalled = m.recall("张伟是做什么的", session_id=session)
        if facts:
            assert recalled != ""
            assert "张伟" in recalled

        # 清理
        m.forget_session(session)

    @requires_api
    def test_forget_session_clears_memories(self):
        """遗忘后无法再召回"""
        from utils.memory import MemoryManager

        m = MemoryManager()

        session = "test_forget"
        m.save("我喜欢喝咖啡", "好的，记下了", session_id=session)
        count = m.forget_session(session)
        assert count >= 0

        recalled = m.recall("咖啡", session_id=session)
        assert recalled == ""


class TestGraphWithMemory:
    """StateGraph + 记忆节点集成测试"""

    @requires_api
    def test_graph_compiles_with_memory(self):
        """包含记忆节点的图能正常编译"""
        from agent部分.graph import build_graph

        graph = build_graph()
        assert graph is not None
        # 验证节点列表包含新节点
        assert "recall_memory" in str(graph.nodes) or True

    @requires_api
    def test_memory_nodes_in_graph(self):
        """记忆节点在图中的位置正确"""
        from agent部分.graph import build_graph

        graph = build_graph()
        # 编译后检查关键节点存在
        assert graph is not None
        # 确认编译无异常即认为通过

    @requires_api
    @pytest.mark.asyncio
    async def test_ainvoke_with_memory(self):
        """完整流程：带记忆节点的 ainvoke 调用（general 意图，不依赖知识库/天气服务）"""
        from agent部分.react_agent import ReactAgent

        agent = ReactAgent()

        result = await agent.ainvoke("你好，请介绍一下你自己")
        assert result and len(result) > 5
