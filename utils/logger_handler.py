"""统一日志配置 —— 控制台 + 文件滚动日志"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.makedirs(config.LOG_DIR, exist_ok=True)

# 控制台输出（简洁格式）
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.DEBUG),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

# 文件输出（含文件名+行号，方便定位）
root_logger = logging.getLogger()
file_handler = RotatingFileHandler(
    os.path.join(config.LOG_DIR, "app.log"),
    maxBytes=config.LOG_MAX_BYTES,
    backupCount=config.LOG_BACKUP_COUNT,
    encoding="utf-8",
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
root_logger.addHandler(file_handler)

# 禁用 LangChain / httpx 等三方库的 DEBUG 日志噪音
for noisy in ["langchain", "langchain_core", "httpx", "httpcore", "openai", "chromadb"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("agent-app")
