"""统一日志配置 —— 控制台 + 文件滚动日志 + trace_id 注入"""

import logging
import os
import sys
import threading
from logging.handlers import TimedRotatingFileHandler

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

# 每日轮转文件日志（保留 365 天，永久可追溯）
# 文件名格式: app.log.2026-07-28, app.log.2026-07-27, ...
daily_handler = TimedRotatingFileHandler(
    os.path.join(config.LOG_DIR, "app.log"),
    when="midnight",
    interval=1,
    backupCount=getattr(config, "LOG_BACKUP_COUNT", 365),
    encoding="utf-8",
)
daily_handler.suffix = "%Y-%m-%d"  # 轮转后缀用日期，不用默认的 .YYYY-MM-DD
daily_handler.setLevel(logging.DEBUG)
daily_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(trace_id)s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
daily_handler.addFilter(TraceFilter())
root_logger.addHandler(daily_handler)

# 白名单模式: 只让项目自身的 logger 输出 DEBUG，其余全部 WARNING
# 避免未来新增任何第三方库的 DEBUG 噪音（embedding 向量、HTTP 响应体等）
root_logger.setLevel(logging.WARNING)
logging.getLogger("agent-app").setLevel(logging.DEBUG)
logging.getLogger("agent").setLevel(logging.DEBUG)
logging.getLogger("api").setLevel(logging.DEBUG)
logging.getLogger("rag").setLevel(logging.DEBUG)
logging.getLogger("utils").setLevel(logging.DEBUG)
logging.getLogger("model").setLevel(logging.DEBUG)

# 显式压制：这些 SDK 内部自己 setLevel(DEBUG)，root=WARNING 挡不住
logging.getLogger("dashscope").setLevel(logging.WARNING)

logger = logging.getLogger("agent-app")
