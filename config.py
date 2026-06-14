"""统一配置文件 —— 项目唯一配置入口，所有模块从此导入"""

import os

from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# override=True: .env 文件值优先于系统环境变量
# 所以如果 .env 里是占位符，会覆盖系统真实Key
# 改为 override=False：系统环境变量优先，.env 只是 fallback
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)

# ============ API 密钥 ============
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")


def _is_placeholder(key: str) -> bool:
    return not key or "your-api-key" in key.lower() or key.startswith("$")


# 清理 .env 值中可能误带的前后空格和引号
DASHSCOPE_API_KEY = DASHSCOPE_API_KEY.strip().strip('"').strip("'")

if _is_placeholder(DASHSCOPE_API_KEY):
    DASHSCOPE_API_KEY = ""  # 置空，让后续检查统一报错

if not DASHSCOPE_API_KEY:
    raise ValueError(
        "未检测到有效的 DASHSCOPE_API_KEY。请任选一种方式配置：\n"
        "  1) 系统环境变量: export/set DASHSCOPE_API_KEY=sk-xxx\n"
        "  2) 项目 .env 文件: 编辑项目根目录 .env，写入 DASHSCOPE_API_KEY=sk-xxx\n"
        "  免费申请: https://bailian.console.aliyun.com/"
    )

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# ============ 模型配置 ============
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL", "qwen-plus")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ============ 路径 ============
RAG_DIR = os.path.join(PROJECT_ROOT, "RAG部分")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOG_DIR = os.path.join(DATA_DIR, "logs")
CHROMA_PERSIST_DIR = os.path.join(DATA_DIR, "chroma_db")
CHAT_HISTORY_DIR = os.path.join(DATA_DIR, "chat_history")
MD5_PATH = os.path.join(DATA_DIR, "md5.text")
MCP_SERVER_PATH = os.path.join(RAG_DIR, "mcp_server.py")

# ============ 日志 ============
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB 单文件上限
LOG_BACKUP_COUNT = 5  # 保留 5 个历史文件

# ============ Chroma ============
COLLECTION_NAME = "rag_project_v2"

# ============ 文本分割 ============
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
SEPARATORS = ["\n\n", "\n", ".", "!", "?", "。", "！", " ", "，"]
MAX_SPLIT_CHAR_NUMBER = 1000

# ============ 检索 ============
RETRIEVER_K = 8
RERANK_TOP_K = 4
ENABLE_RERANK = True
ENABLE_HYBRID = True  # 启用 BM25 + 向量混合检索（RRF 融合）
BM25_K1 = 1.5  # BM25 词频饱和因子
BM25_B = 0.75  # BM25 文档长度归一化因子
RRF_K = 60  # RRF 融合排名阻尼系数

# ============ Agent ============
OPERATOR_NAME = os.getenv("OPERATOR_NAME", "admin")

# ============ Memory ============
ENABLE_MEMORY = True  # 启用 Agent 长期记忆
MEMORY_COLLECTION = "agent_memories"  # ChromaDB collection 名称
MEMORY_TOP_K = 4  # 每次召回记忆条数

# ============ FastAPI ============
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
