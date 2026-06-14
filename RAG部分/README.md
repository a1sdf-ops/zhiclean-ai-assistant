# RAG Engine — Hybrid Retrieval

BM25 sparse retrieval + ChromaDB dense retrieval + RRF fusion + BGE-Reranker re-ranking — a production-grade retrieval pipeline for the knowledge base.

## Pipeline

```
User Query
    │
    ├──→ BM25 Sparse Retrieval (jieba tokenizer → inverted index → BM25 scoring)
    │
    ├──→ ChromaDB Dense Retrieval (text-embedding-v4 → cosine similarity)
    │
    └──→ RRF Fusion (rank-based, k=60)
              │
              └──→ BGE-Reranker v2-m3 (CrossEncoder re-score)
                        │
                        └──→ Top-K Documents → LLM Generation
```

## Project Structure

```
RAG部分/
├── rag.py                  # LCEL RAG chain (hybrid/vector switchable via config)
├── bm25.py                 # BM25 sparse retrieval (hand-written, jieba tokenizer)
├── hybrid_retriever.py     # BM25 + Vector + RRF hybrid retriever
├── knowledge_base.py       # Knowledge CRUD + MD5 dedup
├── vector_stores.py        # ChromaDB vector store wrapper
├── rerank.py               # BGE-Reranker v2-m3 (CrossEncoder, CPU inference)
├── mcp_server.py           # MCP Server (7 tools, JSON-RPC over stdio)
├── file_history_store.py   # Conversation history JSON persistence
└── demo.py                 # CLI upload/demo entry
```

## Configuration

All configuration centralized in project root `config.py`. Key retrieval settings:

```python
RETRIEVER_K = 8         # Final documents returned
RERANK_TOP_K = 4        # After re-ranking
ENABLE_HYBRID = True    # Toggle BM25+Vector hybrid search
BM25_K1 = 1.5           # BM25 term frequency saturation
BM25_B = 0.75           # BM25 document length normalization
RRF_K = 60              # RRF rank damping coefficient
```

## Usage

```bash
cd RAG部分
python demo.py
```

Upload the knowledge documents from `docs/` before first use.
