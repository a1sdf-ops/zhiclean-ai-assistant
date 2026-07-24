"""统一日志配置 —— 控制台 + 文件滚动日志 + trace_id 注入"""

import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.makedirs(config.LOG_DIR, exist_ok=True)

# ── trace_id 线程局部上下文 ──
_trace_ctx = threading.local()
_trace_ctx.current_id = "-"


def set_trace_id(trace_id: str) -> None:
    _trace_ctx.current_id = trace_id


def get_trace_id() -> str:
    return getattr(_trace_ctx, "current_id", "-")


class TraceFilter(logging.Filter):
    def filter(self, record):
        record.trace_id = getattr(_trace_ctx, "current_id", "-")
        return True


# 控制台输出（简洁格式）
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.DEBUG),
    format="%(asctime)s | %(levelname)-7s | %(trace_id)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

# 文件输出（含文件名+行号+trace_id，方便定位单次请求全链路）
root_logger = logging.getLogger()
for h in root_logger.handlers:
    h.addFilter(TraceFilter())

file_handler = RotatingFileHandler(
    os.path.join(config.LOG_DIR, "app.log"),
    maxBytes=config.LOG_MAX_BYTES,
    backupCount=config.LOG_BACKUP_COUNT,
    encoding="utf-8",
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(trace_id)s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
file_handler.addFilter(TraceFilter())
root_logger.addHandler(file_handler)

# 禁用 LangChain / httpx 等三方库的 DEBUG 日志噪音
for noisy in ["langchain", "langchain_core", "httpx", "httpcore", "openai", "chromadb"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("agent-app")
