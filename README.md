# ZhiClean AI After-Sales Assistant

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.3+-ff6f00?logo=langchain)](https://langchain-ai.github.io/langgraph/)
[![Go](https://img.shields.io/badge/Go-1.21+-00ADD8?logo=go)](https://go.dev/)
[![MCP](https://img.shields.io/badge/MCP-1.0+-6366f1)](https://modelcontextprotocol.io/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://github.com/A1SDF-OPS/zhiclean-ai-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/A1SDF-OPS/zhiclean-ai-assistant/actions/workflows/tests.yml)

An intelligent after-sales service platform for the fictional smart home brand "ZhiClean", built to demonstrate **production-grade LLM application engineering** capabilities — from custom LangGraph orchestration to BM25+vector hybrid retrieval, and cross-language MCP protocol integration.

## Why This Project Exists

This project is designed to answer one question on a resume: **"Can this candidate build a real LLM application, not just call APIs?"**

It demonstrates:

- **Algorithm depth**: Hand-written BM25 sparse retrieval (not Elasticsearch), RRF fusion, BGE-Reranker re-ranking
- **System design**: 12-node custom StateGraph (not `create_agent`), MCP protocol for tool-service decoupling
- **Agent memory**: Short-term conversation buffer + long-term semantic memory with LLM fact extraction
- **Cross-language engineering**: Go MCP weather server + Python MCP knowledge server
- **Production awareness**: Streaming SSE, structured logging, dual-mode invocation, Docker deployment

## Architecture

```
                         HTTP / SSE
User ─────────────────────────────────────────► FastAPI
                                                    │
                                                    ▼
┌───────────────────────────────────────────────────────┐
│                  LangGraph Agent                       │
│                                                        │
│  recall_memory → classify_intent ──► route ──┬── handle_weather        │
│                      (7 intents) ├── handle_user_report │
│                               ├── handle_knowledge_*   │
│                               └── handle_general        │
│                                    │                    │
│                               log_tool_call             │
│                                    │                    │
│                               generate_final            │
│                                    │                    │
│                               save_memory ──→ END       │
└───────────────────┬────────────────────────────────────┘
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
┌──────────────┐ ┌──────────┐ ┌────────────────┐
│ RAG Engine   │ │ Go MCP   │ │ External Tools  │
│ (Python)     │ │ Server   │ │ (Python, local) │
│              │ │          │ │                 │
│ BM25 +       │ │ Weather  │ │ User behavior   │
│ Chroma +     │ │ API      │ │ Report gen      │
│ RRF + Rerank │ │          │ │                 │
└──────────────┘ └──────────┘ └────────────────┘
      MCP             MCP
  (JSON-RPC)      (JSON-RPC)
```

## Quick Start

### Prerequisites

- Python 3.11+
- Go 1.21+ (for weather server)
- [DashScope API Key](https://bailian.console.aliyun.com/) (free)

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

# 5. Upload knowledge documents
python RAG部分/demo.py
# Upload docs/*.txt files through the CLI

# 6. Start the service
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/health` — you should see `{"status": "ok", "version": "2.0.0"}`.

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/agent/stream` | Streaming agent dialogue (SSE) |
| `POST` | `/api/v1/agent/invoke` | Full async response (testing) |
| `POST` | `/api/v1/agent/chat` | Agent chat (non-streaming) |
| `POST` | `/api/v1/rag/stream` | RAG search streaming (SSE) |
| `POST` | `/api/v1/rag/query` | RAG search non-streaming |
| `POST` | `/api/v1/knowledge/upload` | Upload text to knowledge base |
| `POST` | `/api/v1/knowledge/upload-file` | Upload file to knowledge base |
| `GET` | `/api/v1/knowledge/list` | List documents (paginated) |
| `PUT` | `/api/v1/knowledge/{name}` | Update a document |
| `DELETE` | `/api/v1/knowledge/{name}` | Delete a document |
| `GET` | `/health` | Health check |

### Run Tests

```bash
pytest tests/ -v
```

### Retrieval Evaluation

The project includes a quantitative retrieval evaluation framework that compares four retrieval strategies on 25 annotated queries across 3 difficulty levels.

```bash
# Run evaluation only
pytest tests/test_retrieval_eval.py -v -s

# Run comparison table only
pytest tests/test_retrieval_eval.py -v -s -k "comparison_table"
```

**Evaluation results** (11 chunks, 25 annotated queries):

| Strategy | MRR | Recall@5 | Precision@5 |
|----------|-----|----------|-------------|
| BM25 (sparse only) | 0.240 | 0.320 | 0.168 |
| Vector (dense only) | **0.933** | 1.000 | 0.416 |
| Hybrid (BM25+Vec+RRF) | 0.873 | 1.000 | 0.408 |
| Hybrid + Reranker | 0.873 | 1.000 | **0.510** |

Key takeaways:
- **Vector retrieval** dominates small, semantically distinct document sets (best MRR)
- **Reranker** provides a +10% precision boost over raw hybrid retrieval (0.51 vs 0.41)
- **BM25 alone** underperforms on Chinese semantic queries (lexical mismatch), but is valuable as a complementary signal at scale
- Difficulty-stratified MRR: easy 0.79 / medium 1.00 / hard 0.83

> Note: With only 11 document chunks, the hybrid's BM25 signal introduces noise. At larger scale (1000+ chunks), BM25's lexical precision increasingly complements vector recall. This is benchmarked and documented, not assumed.

## Key Technical Decisions (and Why)

| Decision | Rationale |
|----------|-----------|
| **Hand-written BM25**, not Elasticsearch | Demonstrate algorithm implementation ability; BM25 formula and inverted index from scratch |
| **Custom StateGraph**, not `create_agent` | Show workflow orchestration skill; specific/intent-based routing beats auto/ReAct for this use case |
| **Go MCP Server** for weather | Demonstrate cross-language MCP protocol integration |
| **Python MCP Server** for knowledge | Knowledge tools use direct function calls (no MCP overhead needed); MCP server still provided for optional external access |
| **RRF fusion**, not score normalization | RRF is score-distribution-agnostic; zero-parameter; academically validated |
| **BGE-Reranker** for second-stage ranking | Cross-Encoder re-ranking precision far exceeds Bi-Encoder cosine similarity |
| **Quantitative evaluation** (25 queries, 4 strategies) | Prove retrieval quality with numbers (MRR, Recall@K, Precision@K), not assumptions |

## Project Structure

```
.
├── .github/workflows/
│   └── tests.yml              # CI/CD: Python tests + Go build + Docker build
├── api/                       # FastAPI layer (routers, schemas, dependencies)
├── agent部分/                  # Agent core
│   ├── graph.py               # 12-node LangGraph StateGraph
│   ├── state.py               # AgentState TypedDict
│   ├── react_agent.py         # stream() + ainvoke() dual mode
│   ├── agent_tools.py         # 7 knowledge-base tools
│   ├── mcp_client.py          # Multi-server MCP client manager
│   ├── agent_demo.py          # CLI interactive demo
│   ├── app_qa.py              # Streamlit chat UI
│   ├── app_upload.py          # Streamlit upload UI
│   └── tools/
│       └── external_tools.py  # Weather / user data / report tools (×6)
├── RAG部分/                    # Knowledge retrieval engine
│   ├── rag.py                 # LCEL RAG chain (hybrid/vector switchable)
│   ├── bm25.py                # BM25 sparse retrieval (hand-written)
│   ├── hybrid_retriever.py    # BM25 + Vector + RRF hybrid retriever
│   ├── vector_stores.py       # ChromaDB vector store
│   ├── rerank.py              # BGE-Reranker v2-m3 (CrossEncoder)
│   ├── knowledge_base.py      # Knowledge CRUD + MD5 dedup
│   ├── mcp_server.py          # MCP Server (7 tools, JSON-RPC over stdio)
│   ├── file_history_store.py  # Conversation history persistence
│   └── demo.py                # CLI knowledge upload demo
├── go-weather-server/         # Go MCP weather server
│   ├── main.go
│   └── internal/
│       ├── mcp/               # JSON-RPC protocol & stdio server
│       └── weather/            # QWeather API client
├── model/
│   └── factory.py             # LLM / Embedding factory (DashScope + OpenAI)
├── utils/
│   ├── logger_handler.py      # Structured logging (console + file rotation)
│   ├── weather_service.py     # QWeather API (Python fallback)
│   └── memory.py              # Agent long-term memory manager
├── docs/                      # Knowledge base source documents (×5)
├── tests/
│   ├── test_graph.py          # Agent graph tests (intent, compile, MCP, ainvoke)
│   ├── test_memory.py         # Memory system tests (CRUD, extraction, recall)
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
| Vector DB | ChromaDB | Local persistence, no infra dependency |
| Sparse Retrieval | BM25 (hand-written) | jieba tokenizer, inverted index, IDF smoothing |
| Fusion | RRF (Reciprocal Rank Fusion) | k=60, rank-based, score-distribution-agnostic |
| Re-ranker | BGE-Reranker v2-m3 | Cross-Encoder, CPU inference |
| Agent Framework | LangGraph StateGraph | Custom 12-node workflow |
| RAG Chain | LangChain LCEL | Streaming + non-streaming dual chain |
| MCP Protocol | JSON-RPC over stdio | Python + Go cross-language |
| Backend | FastAPI | SSE streaming, lifespan preload |
| Demo UI | Streamlit | Dev/debug only |
| Deployment | Docker + docker-compose | Multi-stage build |

## Tools Catalog (13 in total)

### Knowledge Base (7 tools, via LangChain `@tool`)
| Tool | Trigger Intent | Description |
|------|---------------|-------------|
| `search_knowledge` | `knowledge_search` | Semantic search on product knowledge base |
| `upload_knowledge` | `knowledge_upload` | Upload text to knowledge base |
| `upload_knowledge_file` | `knowledge_upload` | Upload file to knowledge base |
| `list_knowledge` | `knowledge_list` | Paginated list of all documents |
| `update_knowledge` | `knowledge_upload` | Replace document content |
| `update_knowledge_file` | `knowledge_upload` | Replace document from file |
| `delete_knowledge` | `knowledge_delete` | Delete document by name |

### External Services (6 tools, direct function call)
| Tool | Source | Description |
|------|--------|-------------|
| `get_weather` | Go MCP Server / Python fallback | Real-time city weather |
| `get_user_id` | config.py | Current operator identifier |
| `get_user_location` | env `DEFAULT_CITY` | Default city for weather |
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

## Demo

### Streamlit UI (dev)

```bash
# Chat interface
cd agent部分
streamlit run app_qa.py

# Document upload interface
streamlit run app_upload.py
```

### Example Conversations

```
User: Z2 Pro滤网要怎么保养？

Agent: 根据保养维护指南，滤网保养建议如下：
- 每周轻敲去除浮灰
- 如颜色发灰/发黄，清水冲洗后彻底晾干（至少24小时）
- HEPA滤网最多水洗3次，超过后过滤效率明显下降
- 建议1-2个月更换一次（价格29元）
- 如果出风口有肉眼可见灰尘喷出，说明滤网已失效需立刻更换

User: 我叫小明，我的扫地机器人是Z2 Pro

Agent: 了解了小明！Z2 Pro是扫拖一体机器人...
(下一轮对话中 Agent 能通过长期记忆召回用户叫"小明"、机型是Z2 Pro)
```

## License

MIT © [Chen Lei](https://github.com/A1SDF-OPS)
