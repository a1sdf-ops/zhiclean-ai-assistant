# Agent 部分 —— LangGraph 工作流编排

基于自定义 LangGraph StateGraph 的 Agent 引擎，手动编排 **12 节点**工作流，集成**三层记忆系统**、**Token 成本追踪**、**多租户隔离**、**MCP 跨语言工具调度**，支持流式（SSE）和异步全量双模式。

## 完整架构

```
用户输入 (HTTP/SSE)
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│                      LangGraph StateGraph                        │
│                                                                  │
│  ┌──────────────┐     ┌──────────────────┐                       │
│  │ recall_memory│────▶│ classify_intent  │                       │
│  │ (三层记忆召回)│     │ (LLM 意图分类)    │                       │
│  └──────────────┘     └───────┬──────────┘                       │
│                               │                                   │
│                    ┌──────────┼──────────┐                        │
│                    ▼          ▼          ▼                        │
│              ┌─────────┐ ┌────────┐ ┌──────────┐                 │
│              │ weather │ │ report │ │knowledge_│  ...(7条路由)    │
│              │ (天气)   │ │ (报告)  │ │ search   │                 │
│              └────┬────┘ └───┬────┘ └─────┬────┘                 │
│                   │          │            │                        │
│                   └──────────┼────────────┘                        │
│                              ▼                                     │
│                      ┌───────────────┐                            │
│                      │ log_tool_call │ (结构化日志埋点)            │
│                      └───────┬───────┘                            │
│                              ▼                                     │
│                      ┌──────────────────┐                         │
│                      │ generate_final   │ (动态Prompt + 记忆注入)  │
│                      └───────┬──────────┘                         │
│                              ▼                                     │
│                      ┌──────────────┐                             │
│                      │ save_memory  │ (三层持久化写入)             │
│                      └──────┬───────┘                             │
│                              │                                     │
│                              ▼                                     │
│                             END                                    │
└──────────────────────────────────────────────────────────────────┘
    │                    │                    │
    ▼                    ▼                    ▼
┌────────────┐   ┌──────────────┐   ┌────────────────┐
│ Redis      │   │ ChromaDB     │   │ MCP 工具层      │
│ Hash(短期) │   │ 向量语义记忆  │   │ Go Weather     │
│ Sorted Set │   │              │   │ Python Knowledge│
│ (长期偏好)  │   │              │   │ External Tools │
└────────────┘   └──────────────┘   └────────────────┘
```

**关键设计决策**：
- `recall_memory` 和 `save_memory` 分别位于图的**入口和出口**，确保每轮对话都经过记忆闭环
- 所有工具节点 → `log_tool_call` → `generate_final` → `save_memory` 是**统一后链路**，无论走哪个意图分支，最终都汇集到同一出口
- `classify_intent` 后的条件边支持 7 条路由，新增工具只需在 `route_by_intent` 追加映射即可

---

## State 状态机设计（`state.py`）

AgentState 采用 **TypedDict 强类型定义**，字段按功能分为五层：

```
AgentState
├── [身份层] session_id / tenant_id / trace_id
│   ├── session_id:   会话唯一标识，关联短期记忆 key
│   ├── tenant_id:    租户/用户隔离标识，多租户场景下数据绝不串扰
│   └── trace_id:     12位十六进制，一次 invoke 一个，全链路日志追踪
│
├── [对话层] messages: Annotated[list, add_messages]
│   └── LangGraph 自动合并消息列表，支持 HumanMessage/AIMessage
│
├── [路由层] intent: str
│   └── 7 个合法值 → VALID_INTENTS 白名单校验
│
├── [工具层] tool_name / tool_args / tool_result
│   ├── tool_name:   当前调用的工具名（generate_final 用它选 Prompt 模板）
│   ├── tool_args:   传给工具的参数字典
│   └── tool_result: 工具返回的原始结果字符串
│
├── [记忆层] memory_context / user_query
│   ├── memory_context: recall_memory 节点召回的记忆文本（注入后续 Prompt）
│   └── user_query:     用户原始输入（save_memory 用它关联事实）
│
└── [生成层] is_report: bool
    └── true → 使用 REPORT_PROMPT；false → FINAL_ANSWER_PROMPT
```

**设计约束**：
- `intent` 必须是 `VALID_INTENTS` 白名单成员，否则 fallback 到 `"general"`
- `trace_id` 在 `_build_initial_state()` 中自动生成，一次对话全程不变
- 所有字段都有默认值（空字符串/空字典/False），不会出现 `None` 异常

---

## Token 成本追踪（`token_tracker.py`）

分模块统计每次 LLM 调用的 **输入/输出 Token 数** 和 **延迟（ms）**，输出占比报告。

### 核心设计

```
TokenTracker (全局单例)
├── record(module, input_tokens, output_tokens, latency_ms)
│   └── 按模块名聚合累积 token 和调用次数
├── report() → dict
│   └── 返回 {module: {total_tokens, input_tokens, output_tokens,
│                      call_count, avg_latency_ms, pct}, ..., __total__}
├── reset()
│   └── 清零所有统计数据
│
├── estimate_tokens(text) → int
│   └── 粗略估算（len(str(text)) // 3），精确值应从 API response.usage 获取
│
└── @track("module_name") 装饰器
    └── 无侵入自动统计被装饰函数的输入/输出 token 和耗时
```

### 埋点位置（graph.py）

| 模块 | 埋点位置 | 统计内容 |
|------|---------|---------|
| `llm_intent_classifier` | `classify_intent()` | 意图分类 LLM 调用的 token 消耗 |
| `llm_generation` | `generate_final_answer()` | 最终回答生成 LLM 调用的 token 消耗 |

### 输出示例

```
Token报告: 总计=3870 tokens | {
  'llm_intent_classifier': 850,
  'llm_generation': 3020
} (trace=a1b2c3d4e5f6)
```

**面试价值**：字节火山引擎商业化考核核心——"你的 Agent 一次对话烧了多少 Token？" 每个模块独立可量化。

---

## 多租户隔离

所有记忆存储的 key 都按 `{tenant_id}:{session_id}` 命名空间隔离：

```
Redis Key 格式:
  mem:st:{tenant_id}:{session_id}    ← 短期记忆 (Hash)
  mem:lt:{tenant_id}:{session_id}    ← 长期记忆 (Sorted Set)

ChromaDB 元数据:
  {session_id: "xxx"}               ← 语义记忆过滤条件

State 字段:
  tenant_id                          ← 请求传入，贯穿整个图流转
  session_id                         ← 不传则自动生成
```

**隔离保证**：
- 租户 A 的短期/长期记忆 key 永远不会与租户 B 冲突
- ChromaDB 查询时按 `session_id` 过滤，不会漏出其他会话的数据
- `agent_demo.py` / API 入口均支持 `--tenant` / `tenant_id` 参数

---

## 三层记忆系统（`memory_store.py` + `utils/memory.py`）

### 第一层：短期上下文 —— Redis Hash

| 属性 | 说明 |
|------|------|
| 存储引擎 | Redis Hash |
| Key | `mem:st:{tenant_id}:{session_id}` |
| TTL | 72 小时（`config.SHORT_MEM_TTL_HOURS`），到期自动清理 |
| 内容 | `recent_msgs`（最近 20 轮对话 JSON）、`last_intent`、`last_tool`、`conversation_turn` |
| 操作 | `save()` 滑动窗口写入、`load()` 读取、`clear()` 手动清除 |
| 用途 | 多轮对话快速获取"刚才聊了什么"，避免重复推理 |

### 第二层：长期语义记忆 —— ChromaDB

| 属性 | 说明 |
|------|------|
| 存储引擎 | ChromaDB（向量数据库） |
| 提取方式 | LLM 从对话中提取事实 → `text-embedding-v4` 生成向量 → 存储 |
| 召回方式 | 语义相似度检索（与当前 query 做 embedding 匹配） |
| 实现文件 | `utils/memory.py` MemoryManager |

### 第三层：长期偏好记忆 —— Redis Sorted Set

| 属性 | 说明 |
|------|------|
| 存储引擎 | Redis Sorted Set |
| Key | `mem:lt:{tenant_id}:{session_id}` |
| 数据结构 | `ZSET {fact: weight}`，weight 越高越重要 |
| 衰减机制 | 每天 `weight *= DECAY_FACTOR`（默认 0.95），低于 `MIN_WEIGHT`（默认 0.1）自动淘汰 |
| 召回方式 | `ZREVRANGE` 取 TopK（默认 K=5），与语义通道去重合并 |

### 双通道召回策略（`graph.py` `recall_memory` 节点）

```
recall_memory 节点
├── 1. Redis Hash → 短期上下文（最近3轮对话 + 上次意图）
├── 2. ChromaDB → 长期语义召回（与当前 query 向量相似）
├── 3. Redis Sorted Set → 长期偏好 TopK（按权重降序）
│   └── 去重: 已在语义通道命中的事实不再重复注入
└── 合并为 memory_context 字符串 → 注入后续节点的 Prompt
```

**核心设计理念**：
- **语义通道（ChromaDB）** → 保证"贴合此刻话题"
- **重要性通道（Sorted Set）** → 保证"永远重要的偏好不被遗漏"
- 两条通道互补，去重合并，避免信息冗余

---

## MCP 多服务器管理（`mcp_client.py`）

基于 **JSON-RPC 2.0 over stdio** 的跨语言工具调度层。

```
MCPClientManager (应用级单例)
├── knowledge → MCPServerConnection
│   ├── 命令: python rag/mcp_server.py
│   └── 工具: search/upload/list/update/delete × 7
│
└── weather → MCPServerConnection
    ├── 命令: go-weather-server/weather-mcp-server (.exe)
    ├── fallback: go run main.go（二进制未编译时）
    └── 工具: get_weather × 1

通信流程:
  1. subprocess.Popen 启动子进程
  2. initialize 握手（protocolVersion: 2024-11-05）
  3. notifications/initialized 通知
  4. tools/call JSON-RPC 请求 → 解析响应
  5. 线程安全: 每个连接有 _lock，支持并发调用
```

**设计亮点**：
- Go 和 Python 两种语言的 MCP Server 被统一管理，对上层调用者透明
- 自动检测二进制是否存在，不存在时 fallback 到 `go run`
- stderr 后台线程消费，避免管道阻塞

---

## 工具集（共 13 个）

### 知识库工具（7 个，`agent_tools.py`）

| 工具 | 触发意图 | 说明 |
|------|---------|------|
| `search_knowledge` | `knowledge_search` | 语义搜索知识库 |
| `upload_knowledge` | `knowledge_upload` | 上传文本到知识库 |
| `upload_knowledge_file` | `knowledge_upload` | 上传本地文件到知识库 |
| `list_knowledge` | `knowledge_list` | 分页列出文档列表 |
| `update_knowledge` | `knowledge_upload` | 更新文档内容 |
| `update_knowledge_file` | `knowledge_upload` | 用文件更新文档 |
| `delete_knowledge` | `knowledge_delete` | 按名称删除文档 |

### 外部服务工具（6 个，`tools/external_tools.py`）

| 工具 | 来源 | 说明 |
|------|------|------|
| `get_weather` | Go MCP Server / Python fallback | 实时城市天气 |
| `get_user_id` | `config.OPERATOR_NAME` | 当前操作者标识 |
| `get_user_location` | 环境变量 `DEFAULT_CITY` | 默认城市 |
| `get_current_month` | `datetime.now()` | 当前月份 (YYYY-MM) |
| `fetch_external_data` | `data/user_behavior.csv` | 用户月度使用记录 |
| `fill_context_for_report` | 本地 | 激活报告生成模式 |

---

## MinimalAgent —— 零框架手写状态机（`minimal_agent.py`）

专门为面试场景设计：不依赖 LangGraph，用纯 Python 实现 Agent 状态流转。

```
核心知识点:
├── TypedDict 强类型 → 拒绝随意增删字段
├── 路由表 dispatch → 显式声明状态流转路径（无隐式跳转）
├── 手动 checkpoint → JSON 快照 + 异常回滚
└── error_count 熔断 → 3次异常强制终止

流转:
  router → execute_tool → generate → end
            ↑ 失败回滚  │
            └─ checkpoint ─┘
```

**与 LangGraph 版的对应关系**：

| 概念 | LangGraph 版 | MinimalAgent 版 |
|------|-------------|-----------------|
| 意图分类 | `classify_intent` (LLM) | `route()` (关键词匹配) |
| 工具执行 | 7个独立节点 | `execute_tool()` 模拟 |
| 生成回答 | `generate_final_answer()` (LLM) | `generate()` (拼接) |
| 记忆召回 | `recall_memory` 节点 | `memory_context` 参数传入 |
| 异常处理 | LangGraph 内部 | 手动 checkpoint + 3次熔断 |

---

## 双模式运行（`react_agent.py`）

| 模式 | 方法 | 返回方式 | 使用场景 |
|------|------|---------|---------|
| 流式（生产） | `execute_stream(query)` | `Iterator[str]` 逐 token yield | SSE 实时推送给前端 |
| 异步全量（测试） | `ainvoke(query)` | `str` 完整回答 | 批处理/集成测试，附带 Token 报告 |

```python
# 生产路径
agent = ReactAgent()
for token in agent.execute_stream("Z2 Pro怎么保养？"):
    print(token, end="", flush=True)

# 测试路径
answer = await agent.ainvoke("Z2 Pro怎么保养？")
# 后台自动输出 Token 成本报告
```

---

## 项目结构

```
agent/
├── graph.py              # ★ 核心：12节点 LangGraph StateGraph 编排
│                         #   节点: recall_memory → classify_intent → (条件边)
│                         #   → handle_* (7个工具节点) → log_tool_call
│                         #   → generate_final_answer → save_memory → END
│
├── state.py              # AgentState TypedDict 强类型定义
│                         #   五层字段: 身份层/对话层/路由层/工具层/记忆层/生成层
│                         #   trace_id 生成、VALID_INTENTS 白名单
│
├── react_agent.py        # Graph 运行器: execute_stream() + ainvoke() 双模式
│                         #   初始 state 构造（session/tenant/trace 注入）
│
├── token_tracker.py      # Token 成本埋点: TokenTracker 分模块统计
│                         #   record()/report()/reset() + @track 装饰器
│                         #   estimate_tokens() 粗略估算
│
├── memory_store.py       # 三层记忆中的 Redis 两层
│                         #   ShortTermMemory: Redis Hash, 72h TTL, 滑动窗口
│                         #   LongTermMemory: Redis Sorted Set, 权重衰减, TopK
│
├── mcp_client.py         # MCP 多服务器客户端管理器
│                         #   MCPServerConnection: JSON-RPC over stdio
│                         #   MCPClientManager: 应用级单例，管理 knowledge+weather
│
├── agent_tools.py        # 7 个 LangChain @tool 知识库工具
│                         #   search/upload/upload_file/list/update/update_file/delete
│
├── minimal_agent.py      # 零框架手写 Agent 状态机（面试参考实现）
│                         #   路由表 dispatch + checkpoint 回滚 + error_count 熔断
│
├── tools/
│   └── external_tools.py # 6 个外部工具: 天气/用户ID/位置/月份/使用数据/报告标记
│
├── agent_demo.py         # CLI 交互式对话入口（开发调试用）
├── app_qa.py             # Streamlit 问答 UI
├── app_upload.py         # Streamlit 文档上传 UI
└── __init__.py
```

---

## 运行

### CLI 交互式

```bash
cd agent
python agent_demo.py --tenant my_user
```

### FastAPI 端点

```bash
# 流式对话
curl -X POST http://localhost:8000/api/v1/agent/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Z2 Pro滤网怎么保养？", "session_id": "s1", "tenant_id": "u1"}'

# 非流式
curl -X POST http://localhost:8000/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "你好", "session_id": "s1", "tenant_id": "u1"}'
```

### Streamlit UI（开发调试）

```bash
streamlit run agent/app_qa.py      # 对话界面
streamlit run agent/app_upload.py  # 上传界面
```
