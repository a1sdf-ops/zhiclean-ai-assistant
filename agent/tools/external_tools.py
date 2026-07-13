"""外部工具 —— 天气、用户数据、报告上下文（从项目一迁移）"""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from utils.logger_handler import logger

# ---------- 外部用户数据（从 CSV 加载） ----------

_external_data: dict = {}
_external_data_loaded: bool = False


def _load_external_data() -> None:
    global _external_data, _external_data_loaded
    if _external_data_loaded:
        return
    _external_data_loaded = True
    csv_path = os.path.join(config.PROJECT_ROOT, "data", "user_behavior.csv")
    if not os.path.exists(csv_path):
        logger.warning("外部数据文件不存在: %s", csv_path)
        return
    with open(csv_path, encoding="utf-8") as f:
        for line in f.readlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("用户ID"):
                continue
            arr = stripped.split(",")
            if len(arr) < 6:
                continue
            uid = arr[0].replace('"', "")
            _external_data.setdefault(uid, {})[arr[5].replace('"', "")] = {
                "特征": arr[1].replace('"', ""),
                "效率": arr[2].replace('"', ""),
                "耗材": arr[3].replace('"', ""),
                "对比": arr[4].replace('"', ""),
            }


# ---------- 工具函数 ----------


def get_weather(city: str) -> str:
    """获取指定城市的实时天气"""
    try:
        from utils.weather_service import get_weather_data

        return get_weather_data(city)
    except ImportError:
        return f"[天气服务未就绪] 无法获取 {city} 的天气"


def get_user_id() -> str:
    """返回当前用户ID"""
    return config.OPERATOR_NAME


def get_user_location() -> str:
    """返回用户所在城市"""
    return os.getenv("DEFAULT_CITY", "北京")


def get_current_month() -> str:
    """返回当前月份 YYYY-MM"""
    return datetime.now().strftime("%Y-%m")


def fetch_external_data(user_id: str, month: str) -> str:
    """获取用户月度使用记录"""
    _load_external_data()
    try:
        return json.dumps(_external_data[user_id][month], ensure_ascii=False, indent=2)
    except KeyError:
        logger.warning("未找到用户 %s 在 %s 的使用记录", user_id, month)
        return "{}"


def fill_context_for_report() -> str:
    """标记为报告生成场景，触发报告模式prompt"""
    return "[报告模式已激活]"
