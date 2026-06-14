# Agent 部分 —— LangGraph 工作流编排

基于自定义 LangGraph StateGraph 的 Agent 引擎，手动编排 10 节点工作流，通过 MCP 协议调度多个工具源。

## 架构

```
用户输入 → classify_intent（LLM意图分类）
    → 条件边路由（weather/user_report/knowledge_search/...）
    → 工具节点（MCP调用 / 直接调用）
    → log_tool_call（监控记录）
    → generate_final（动态prompt切换）→ SSE流式输出
```

## 项目结构

```
agent部分/
├── state.py                # AgentState 类型定义
├── graph.py                # 12节点 StateGraph 编排（核心）
├── react_agent.py          # stream() + ainvoke() 双模式
├── agent_tools.py          # 7个知识库 LangChain tools
├── mcp_client.py           # 多MCP Server管理器（knowledge + weather）
├── tools/
│   └── external_tools.py   # 天气/用户数据/报告工具
├── agent_demo.py           # CLI交互入口
├── app_qa.py               # Streamlit 问答UI
└── app_upload.py           # Streamlit 上传UI
```

## 工具集

| 工具 | 来源 | 通信方式 |
|------|------|---------|
| search/upload/list/update/delete | RAG部分 MCP Server | MCP JSON-RPC |
| get_weather | Go Weather Server | MCP JSON-RPC |
| user_data/report | Python 直接调用 | 进程内 |

## 运行

```bash
cd agent部分
python agent_demo.py
```

或使用 FastAPI `/api/v1/agent/stream` 端点。
