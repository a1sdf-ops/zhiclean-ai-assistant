# ZhiClean AI After-Sales Assistant

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.3+-ff6f00?logo=langchain)](https://langchain-ai.github.io/langgraph/)
[![Go](https://img.shields.io/badge/Go-1.21+-00ADD8?logo=go)](https://go.dev/)
[![MCP](https://img.shields.io/badge/MCP-1.0+-6366f1)](https://modelcontextprotocol.io/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://github.com/A1SDF-OPS/zhiclean-ai-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/A1SDF-OPS/zhiclean-ai-assistant/actions/workflows/tests.yml)

An intelligent after-sales service platform for the smart home brand "ZhiClean", built to demonstrate **production-grade LLM application engineering** — from custom LangGraph orchestration to BM25+vector hybrid retrieval, structured user profiling, and cross-language MCP protocol integration.

## Highlights

- **12-node LangGraph StateGraph** — intent classification → 7-way routing → tool execution → answer generation → profile persistence
- **User profile system** — 8-field structured JSON profile per tenant (devices with issues/usage/consumables, location tracking, purchase intent, service history, question history); deterministic code-based merge, no LLM scoring drift
- **2-layer agent memory** — ChromaDB semantic memory (session summaries, vector recall) + JSON user profile (O(1) full load); session-end summary via dedicated LLM prompt
- **BM25 + Vector hybrid RAG** — hand-written BM25 sparse retrieval, RRF fusion, BGE-Reranker re-ranking; 17-question evaluation: 0% hallucination, 100% honest on out-of-domain queries
- **Token-level SSE streaming** — `astream_events` captures real LLM tokens (verified: 155 tokens, TTFT 4.9s); default to invoke due to pre-processing latency bottleneck
- **Latency optimized** — merged dual-LLM bottleneck in RAG path, average response 22s → 12s (↓46%)
- **Full observability** — trace_id per request, per-module token & latency tracking, structured logging (1.7MB → 3.4KB noise reduction)
- **Cross-language MCP** — Go weather server + Python knowledge server, JSON-RPC over stdio
- **Multi-tenant isolation** — profile and memory keyed by `tenant_id` (MD5-derived from session_id for stability)

## Architecture

```
                         HTTP / SSE
User ─────────────────────────────────────────► FastAPI
                                                    │
                                                    ▼
┌───────────────────────────────────────────────────────┐
│                  LangGraph Agent                       │
│                                                        │
│  recall_memory ──► classify_intent ──► route ──┬── handle_weather        │
│  (profile load     (LLM: qwen-plus            ├── handle_user_report     │
│   + ChromaDB        ~970ms avg)                ├── handle_knowledge_*    │
│   semantic recall)                             └── handle_general        │
│                                                     │                    │
│                                               log_tool_call              │
│                                                     │                    │
│                                               generate_final_answer      │
│                                               (LLM: ~6.6s avg)           │
│                                                     │                    │
│                                               save_memory ──► END        │
│                                               (LLM: profile_update       │
│                                                only, no per-round facts) │
└───────────────────┬───────────────────────────────────┘
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
┌──────────────┐ ┌──────────┐ ┌────────────────┐
│ RAG Engine   │ │ Go MCP   │ │ External Tools  │
│ (Python)     │ │ Server   │ │ (Python)        │
│              │ │          │ │                 │
│ BM25 +       │ │ Weather  │ │ User behavior   │
│ Chroma +     │ │ API      │ │ Report gen      │
│ RRF + Rerank │ │          │ │                 │
└──────┬───────┘ └────┬─────┘ └────────┬───────┘
       │              │               │
       ▼              ▼               ▼
┌──────────────┐ ┌──────────┐ ┌────────────────┐
│ ChromaDB     │ │ Go MCP   │ │ JSON Profile   │
│ (knowledge)  │ │ (stdio)  │ │ (per tenant)   │
└──────────────┘ └──────────┘ └────────────────┘
```

### Intent Routing (7 handlers)

| Intent | Handler | Description |
|--------|---------|-------------|
| `weather` | `handle_weather` | MCP Go weather server → QWeather API; auto-resolves "当前城市" from profile |
| `knowledge_search` | `handle_knowledge_search` | Hybrid RAG retrieval (no LLM in handler, only in generate_final_answer) |
| `knowledge_upload` | `handle_knowledge_upload` | Upload text/file to ChromaDB knowledge base (MD5 dedup) |
| `knowledge_list` | `handle_knowledge_list` | Paginated document listing |
| `knowledge_delete` | `handle_knowledge_delete` | Delete document by source name |
| `user_report` | `handle_user_report` | Generate monthly usage report (is_report mode) |
| `general` | `handle_general` | Chitchat, greetings, agent self-introduction |

## User Profile System

Each tenant gets a structured JSON profile (`data/profiles/{tenant_id}.json`) that persists across sessions:

```json
{
  "devices": [{
    "model": "Z3 Ultra",
    "issues": [{"problem": "吸力变小", "status": "未解决", "attempted_solutions": ["上门检测，配件缺货待补发"]}],
    "consumables": [{"name": "hepa_filter", "last_replaced": "2026-07", "cycle_days": 90}],
    "usage": {"frequency": "每天一次", "primary_area": "客厅", "floor_type": "木地板", "has_pets": false}
  }],
  "current_location": "深圳",
  "purchase_intent": [{"product": "Z3 Ultra", "level": "高", "last_asked": "2026-07-30"}],
  "question_history": [
    {"category": "故障排查", "device": "Z3 Ultra", "problem": "吸力变小", "query_summary": "吸力变小怎么处理", "resolved": false, "date": "2026-07-30"}
  ],
  "service_history": [{
    "service_type": "上门维修", "target_device": "Z3 Ultra", "service_time": "周三",
    "result": "确认配件缺货，需下周补发", "resolved": false
  }],
  "preferences": {"receive_maintain_remind": true, "remind_time": "每周一上午9点"},
  "updated_at": "2026-07-30 10:00:00",
  "created_at": "2026-07-30 08:00:00"
}
```

**Key design decisions:**
- **LLM active inference** — extracts structured data from natural language (e.g. "搬到深圳" → `current_location` overwrite, "我的Z3 Ultra" → device ownership)
- **Code-only merge** — deterministic, no LLM scoring drift; issues merge by problem name, consumables merged by name, usage dict-update
- **question_history** — FIFO queue (100 item limit), LLM extracts one entry per round, code appends; injected into system prompt as summary stats + last 5 items
- **Session summaries** — on session close, dedicated LLM prompt extracts troubleshooting fragments (repair details, trigger conditions, unresolved issues, promises) and writes to ChromaDB for cross-session recall
- **tenant_id** = `MD5(session_id)[:12]` — stable within session, no random drift
- **No per-round facts in ChromaDB** — structured facts live in JSON profile (`question_history`), ChromaDB only stores session-end summaries to avoid redundancy

## Quick Start

### Prerequisites

- Python 3.11+
- Go 1.21+ (for weather MCP server)
- [DashScope API Key](https://bailian.console.aliyun.com/) (free tier available)

### Setup

```bash
# 1. Clone
git clone https://github.com/A1SDF-OPS/zhiclean-ai-assistant.git
cd zhiclean-ai-assistant

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Configure API key
cp .env.example .env
# Edit .env → set DASHSCOPE_API_KEY=sk-xxx

# 4. Build Go weather server
cd go-weather-server
go build -o weather-mcp-server .
cd ..

# 5. Upload knowledge documents (optional, demo data)
python rag/demo.py

# 6. Start the service
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/health` — you should see `{"status": "ok"}`.

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/agent/stream` | Streaming agent dialogue (SSE, token-level) |
| `POST` | `/api/v1/agent/invoke` | Full async response (testing) |
| `POST` | `/api/v1/agent/chat` | Agent chat (non-streaming) |
| `POST` | `/api/v1/agent/session/close` | Close session (LLM summary → ChromaDB) |
| `POST` | `/api/v1/rag/stream` | RAG search streaming (SSE) |
| `POST` | `/api/v1/rag/query` | RAG search non-streaming |
| `POST` | `/api/v1/knowledge/upload` | Upload text to knowledge base |
| `POST` | `/api/v1/knowledge/upload-file` | Upload file to knowledge base |
| `GET` | `/api/v1/knowledge/list` | List documents (paginated) |
| `PUT` | `/api/v1/knowledge/{name}` | Update a document |
| `DELETE` | `/api/v1/knowledge/{name}` | Delete a document |
| `GET` | `/token-stats` | Token cost dashboard (internal) |
| `GET` | `/health` | Health check |

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **Hand-written BM25**, not Elasticsearch | Demonstrate algorithm implementation; jieba tokenizer, inverted index, IDF smoothing |
| **Custom StateGraph**, not `create_agent` | Intent-based routing beats auto-ReAct for bounded-domain after-sales; 7 discrete handlers, single-intent per round |
| **JSON file profile per tenant**, not Redis/MySQL | ~hundreds of bytes per tenant; O(1) read/write; zero external dependency; physical isolation prevents tenant_id forgery |
| **MD5(session_id) for tenant_id** | Stable identity within session without login; avoids the random UUID drift problem |
| **Code-only profile merge**, no LLM scoring | Deterministic behavior; no memory drift; issues merge by problem name, not vector similarity |
| **RAG handler: retrieve only, no LLM** | Removed one LLM call from RAG path; generate_final_answer does a single pass over retrieved documents; saved 10-16s |
| **Go MCP Server** for weather | Demonstrate cross-language MCP protocol integration (JSON-RPC over stdio) |
| **BGE-Reranker** for second-stage ranking | Cross-Encoder re-ranking precision far exceeds Bi-Encoder cosine similarity |
| **Token-level SSE** (`astream_events`) | Captures real LLM token chunks, not node-level state updates; filters non-generate events by `langgraph_node` metadata |

## Latency Optimization

The original RAG path had **two LLM calls doing the same thing**: the handler LLM generated an answer from retrieved docs, then `generate_final_answer` LLM paraphrased it. Merged into one.

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| `knowledge_search` average (11 questions) | 25.1s | 13.1s | ↓48% |
| All 17 questions average | 22.1s | 11.9s | ↓46% |
| Weather/general/CRUD (5 questions) | 11.4s | 5.6s | ↓51% |
| Hallucination rate | — | 0/17 | 0% |

**Latency breakdown (current):**
- Intent classification (qwen-plus): ~970ms
- Handler retrieval (no LLM): <1s
- Answer generation (qwen-plus): ~6.6s (now the primary bottleneck)
- Profile extraction (qwen-plus): ~1.2s (post-response, not user-facing)

## Retrieval Evaluation

Quantitative comparison of 4 retrieval strategies on 25 annotated queries across 3 difficulty levels:

| Strategy | MRR | Recall@5 | Precision@5 |
|----------|-----|----------|-------------|
| BM25 (sparse only) | **1.000** | 0.980 | 0.441 |
| Vector (dense only) | 0.933 | **1.000** | 0.416 |
| Hybrid (BM25+Vec+RRF) | **1.000** | **1.000** | 0.424 |
| Hybrid + Reranker | **1.000** | **1.000** | **0.500** |

> The knowledge base is intentionally small (11 chunks) — these numbers demonstrate the *methodology* (measuring rather than assuming retrieval quality). At 1000+ chunks the metrics would spread out. The Reranker delivers a real +18% Precision@5 lift, which is the one component whose value the data proves.

## Agent Evaluation (17 Questions)

A comprehensive test suite covering all 7 intent handlers + 2 adversarial tests:

| Category | Count | Examples |
|----------|-------|---------|
| Knowledge search | 9 (T1-T7, T10, T15) | Fault diagnosis, error codes, maintenance, product comparison |
| Weather | 1 (T8) | Real-time city weather via MCP |
| General | 1 (T9) | Chitchat / self-introduction |
| Knowledge CRUD | 3 (T11-T13) | List, upload, delete documents |
| User report | 1 (T14) | Monthly usage report generation |
| Adversarial | 2 (F1-F2) | Non-existent product (Z3 Pro), out-of-domain query (dishwasher) |

**Results:** 17/17 accurate, 0 hallucinations, 2/2 adversarial honesty.

## Project Structure

```
.
├── .github/workflows/
│   └── tests.yml              # CI/CD: Python tests + Go build + Docker build
├── api/                       # FastAPI layer (routers, schemas, dependencies)
├── agent/                     # Agent core
│   ├── graph.py               # 12-node LangGraph StateGraph + all node implementations
│   ├── state.py               # AgentState TypedDict (session/tenant/trace ids, 7 intents)
│   ├── react_agent.py         # Graph runner: stream() + aexecute_stream() + ainvoke() triple mode
│   ├── token_tracker.py       # Per-module token & latency accounting (SQLite)
│   ├── agent_tools.py         # 7 knowledge-base @tool definitions
│   ├── mcp_client.py          # Multi-server MCP client manager
│   ├── agent_demo.py          # CLI interactive demo
│   ├── app_qa.py              # Streamlit chat UI (dev)
│   ├── app_upload.py          # Streamlit upload UI (dev)
│   └── tools/
│       └── external_tools.py  # Weather / user data / report tools (×6)
├── rag/                       # Knowledge retrieval engine
│   ├── rag.py                 # LCEL RAG chain (retrieve-only or LLM modes)
│   ├── bm25.py                # BM25 sparse retrieval (hand-written)
│   ├── hybrid_retriever.py    # BM25 + Vector + RRF hybrid retriever
│   ├── vector_stores.py       # ChromaDB vector store adapter
│   ├── rerank.py              # BGE-Reranker v2-m3 (CrossEncoder)
│   ├── knowledge_base.py      # Knowledge CRUD + MD5 dedup
│   ├── mcp_server.py          # MCP Server (7 tools, JSON-RPC over stdio)
│   ├── file_history_store.py  # Conversation history persistence
│   └── demo.py                # CLI knowledge upload demo
├── go-weather-server/         # Go MCP weather server
│   ├── main.go
│   └── internal/
│       ├── mcp/               # JSON-RPC protocol & stdio server
│       └── weather/           # QWeather API client
├── model/
│   └── factory.py             # LLM / Embedding factory (DashScope + OpenAI)
├── utils/                     # Shared utilities
│   ├── profile.py             # JSON user profile manager (load/merge/save, per tenant)
│   ├── memory.py              # ChromaDB semantic memory (LLM profile extraction + session summary)
│   ├── logger_handler.py      # Structured logging (whitelist + explicit suppression, daily rotation)
│   └── weather_service.py     # QWeather API (Python fallback)
├── scripts/
│   ├── export_trace.py         # Log analysis: extract trace by ID, latency breakdown, daily summary
│   ├── test_memory_e2e.py      # E2E memory test (6 rounds: profile + cross-session recall)
│   └── verify_stream_and_profile.py  # SSE streaming + profile joint verification
├── data/                      # Runtime data (gitignored)
│   ├── profiles/              # JSON user profiles ({tenant_id}.json)
│   ├── chroma_db/             # ChromaDB vectors (knowledge base + agent memories)
│   ├── token_usage.db         # Token cost tracking (SQLite)
│   └── logs/                  # Daily rotating logs (365-day retention)
├── docs/                      # Knowledge base source documents (×5)
├── tests/
│   ├── test_graph.py          # Agent graph tests (intent routing, MCP, streaming)
│   ├── test_memory.py         # Memory system tests (extraction, recall, profile)
│   ├── test_retrieval_eval.py # Retrieval evaluation (MRR, Recall@K, 4-way comparison)
│   ├── eval_queries.json      # 25 annotated evaluation queries (3 difficulty levels)
│   └── conftest.py            # Shared test configuration
├── config.py                  # Single configuration entry point
├── pyproject.toml
├── Dockerfile                 # Multi-stage build (Go + Python)
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── LICENSE
```

## Tech Stack

| Layer | Choice | Notes |
|-------|--------|-------|
| LLM | DashScope qwen-plus / OpenAI GPT | Model factory with dual provider support |
| Embedding | text-embedding-v4 (1024d) | Via DashScope API |
| Agent Framework | LangGraph StateGraph | Custom 12-node DAG, compile-time routing |
| RAG Chain | LangChain LCEL | Streaming + non-streaming dual chain |
| Vector DB | ChromaDB | Local persistence, zero infra dependency |
| Memory / Profile | JSON file + ChromaDB | Profile: per-tenant JSON (8-field structured); Semantic: ChromaDB session summaries |
| Sparse Retrieval | BM25 (hand-written) | jieba tokenizer, inverted index, IDF smoothing |
| Fusion | RRF (Reciprocal Rank Fusion) | k=60, rank-based, score-distribution-agnostic |
| Re-ranker | BGE-Reranker v2-m3 | Cross-Encoder, CPU inference |
| MCP Protocol | JSON-RPC over stdio | Go + Python cross-language |
| Backend | FastAPI | SSE streaming, lifespan preload |
| Cost Tracking | Custom TokenTracker | Per-module token & latency, SQLite persistence |
| Logging | TimedRotatingFileHandler | Daily rotation, 365-day retention, whitelist suppression |
| Demo UI | Streamlit | Dev/debug only |
| Deployment | Docker + docker-compose | Multi-stage build |

## Tools Catalog (12 total)

### Knowledge Base (7 tools)
| Tool | Trigger Intent | Description |
|------|---------------|-------------|
| `search_knowledge` | `knowledge_search` | Hybrid BM25+vector semantic search (retrieve-only, no LLM) |
| `upload_knowledge` | `knowledge_upload` | Upload text to knowledge base (MD5 dedup) |
| `upload_knowledge_file` | `knowledge_upload` | Upload file to knowledge base |
| `list_knowledge` | `knowledge_list` | Paginated list of all documents |
| `update_knowledge` | `knowledge_upload` | Replace document content |
| `update_knowledge_file` | `knowledge_upload` | Replace document from file |
| `delete_knowledge` | `knowledge_delete` | Delete document by source name |

### External Services (5 tools)
| Tool | Source | Description |
|------|--------|-------------|
| `get_weather` | Go MCP Server / Python fallback | Real-time city weather (auto-resolves "当前城市" from profile) |
| `get_user_id` | config.py | Current operator identifier |
| `get_current_month` | datetime | Current month (YYYY-MM) |
| `fetch_external_data` | `data/user_behavior.csv` | User monthly usage record |
| `fill_context_for_report` | local | Switch to report-generation mode |

## Knowledge Base Content

| Document | Content |
|----------|---------|
| Z2 Pro Product Manual | Specs, features, app guide, safety notes |
| Common Troubleshooting | 10 FAQ (power, cleaning, water, network, noise, etc.) |
| Maintenance Guide | Daily/weekly/monthly care, consumable replacement cycles |
| Z3 Ultra Launch | New features, Z2 Pro comparison, FAQ |
| After-Sales Policy | Warranty, returns, extended warranty, recycling |

## Example Conversation

```
User: 我在北京，Z3 Ultra 如何保养，每天扫一次客厅木地板

Agent: 结合您在北京、每天清扫客厅木地板的使用场景，推荐以下保养方案：
✅ 每日保养：清空尘盒、清洗拖布、检查边刷
📅 每周保养：主刷深度清洁、传感器擦拭、滤网清洁
📆 每月重点：更换滤网（北京PM2.5易堵塞）、驱动轮检查
💡 特别提示：Z3 Ultra识别木地板后会自动降低水量+适配吸力

User: 我搬到了深圳，最近吸力变小了怎么处理

Agent: [检测到搬家：北京→深圳；记录问题：吸力变小]
优先排查：清理主刷边刷缠绕物 → 检查HEPA滤网是否堵塞 → 确认尘盒安装到位
深圳环境提示：高湿度易致滤网潮黏，建议清洗后彻底阴干

User: 当前城市的天气是什么样的

Agent: [自动从画像解析"当前城市"=深圳]
深圳实时天气：大雨，26°C，湿度89%，东风4级
```

## License

MIT © [Chen Lei](https://github.com/A1SDF-OPS)
