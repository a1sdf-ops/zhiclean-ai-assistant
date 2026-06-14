"""StateGraph 测试 —— 单元测试 + 集成测试（需API Key）"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agent部分.graph import _parse_intent_response, build_graph

_has_api = (
    bool(config.DASHSCOPE_API_KEY)
    and "your-api-key" not in config.DASHSCOPE_API_KEY
    and "sk-" in config.DASHSCOPE_API_KEY
)
requires_api = pytest.mark.skipif(not _has_api, reason="需要有效的 DASHSCOPE_API_KEY")


class TestIntentParsing:
    def test_parse_weather_intent(self):
        result = _parse_intent_response('{"intent": "weather", "tool_args": {"city": "北京"}, "is_report": false}')
        assert result["intent"] == "weather"
        assert result["tool_args"]["city"] == "北京"

    def test_parse_general_intent(self):
        result = _parse_intent_response('{"intent": "general", "tool_args": {}, "is_report": false}')
        assert result["intent"] == "general"

    def test_parse_invalid_json_fallback(self):
        result = _parse_intent_response("not json at all")
        assert result["intent"] == "general"

    def test_parse_markdown_wrapped_json(self):
        raw = '```json\n{"intent": "knowledge_search", "tool_args": {"query": "滤网怎么换"}, "is_report": false}\n```'
        result = _parse_intent_response(raw)
        assert result["intent"] == "knowledge_search"
        assert result["tool_args"]["query"] == "滤网怎么换"


class TestGraphCompilation:
    def test_graph_compiles(self):
        graph = build_graph()
        assert graph is not None
        nodes = list(graph.nodes.keys())
        assert "classify_intent" in nodes
        assert "generate_final_answer" in nodes
        assert "log_tool_call" in nodes

    @requires_api
    def test_graph_stream_basic(self):
        graph = build_graph()
        chunks = list(
            graph.stream(
                {"messages": [{"role": "user", "content": "你好"}]},
                stream_mode="values",
            )
        )
        assert len(chunks) > 0
        final = chunks[-1]
        assert "messages" in final


class TestMCPIntegration:
    def test_mcp_manager_registry(self):
        from agent部分.mcp_client import get_mcp_manager

        m = get_mcp_manager()
        assert "knowledge" in m._connections
        # weather server registers if binary exists
        assert len(m._connections) >= 1


@pytest.mark.asyncio
@requires_api
async def test_ainvoke_general():
    from agent部分.react_agent import ReactAgent

    agent = ReactAgent()
    result = await agent.ainvoke("你好，请简单介绍你自己")
    assert isinstance(result, str)
    assert len(result) > 0
