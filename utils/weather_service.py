import os

import requests

# 尝试自动加载 .env 文件（如果 python-dotenv 已安装）
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# 和风天气 API 地址
_API_HOST = os.environ.get("QWEATHER_API_HOST", "devapi.qweather.com")

WEATHER_NOW_URL = f"https://{_API_HOST}/v7/weather/now"

# 常用城市 → 和风天气城市ID 映射（覆盖全国主要城市）
# 完整列表可参考：https://github.com/qwd/LocationList/blob/master/China-City-List-latest.csv
CITY_ID_MAP = {
    # 直辖市
    "北京": "101010100",
    "上海": "101020100",
    "天津": "101030100",
    "重庆": "101040100",
    # 广东
    "广州": "101280101",
    "深圳": "101280601",
    "东莞": "101281601",
    "佛山": "101280301",
    "珠海": "101280701",
    "惠州": "101280301",
    "中山": "101281701",
    "汕头": "101280501",
    # 浙江
    "杭州": "101210101",
    "宁波": "101210401",
    "温州": "101210701",
    "嘉兴": "101210301",
    "绍兴": "101210501",
    "金华": "101210901",
    "台州": "101210601",
    # 江苏
    "南京": "101190101",
    "苏州": "101190401",
    "无锡": "101190201",
    "常州": "101191101",
    "南通": "101190501",
    "徐州": "101190801",
    # 四川
    "成都": "101270101",
    "绵阳": "101270401",
    "德阳": "101272001",
    # 湖北
    "武汉": "101200101",
    "宜昌": "101200901",
    "襄阳": "101200201",
    # 湖南
    "长沙": "101250101",
    "株洲": "101250301",
    "湘潭": "101250201",
    # 安徽
    "合肥": "101220101",
    "芜湖": "101220301",
    "蚌埠": "101220201",
    "马鞍山": "101220501",
    "安庆": "101220601",
    # 福建
    "福州": "101230101",
    "厦门": "101230201",
    "泉州": "101230501",
    # 山东
    "济南": "101120101",
    "青岛": "101120201",
    "烟台": "101120501",
    "潍坊": "101120601",
    "临沂": "101120901",
    # 河南
    "郑州": "101180101",
    "洛阳": "101180901",
    "开封": "101180801",
    # 河北
    "石家庄": "101090101",
    "唐山": "101090501",
    "保定": "101090201",
    # 陕西
    "西安": "101110101",
    "咸阳": "101110200",
    # 辽宁
    "沈阳": "101070101",
    "大连": "101070201",
    # 其他省会
    "哈尔滨": "101050101",
    "长春": "101060101",
    "太原": "101100101",
    "呼和浩特": "101080101",
    "南昌": "101240101",
    "南宁": "101300101",
    "海口": "101310101",
    "贵阳": "101260101",
    "昆明": "101290101",
    "拉萨": "101140101",
    "兰州": "101160101",
    "西宁": "101150101",
    "银川": "101170101",
    "乌鲁木齐": "101130101",
    # 港澳台
    "香港": "101320101",
    "澳门": "101330101",
    "台北": "101340101",
}
# 拼音别名（方便测试）
CITY_ID_MAP.update(
    {
        "beijing": "101010100",
        "shanghai": "101020100",
        "tianjin": "101030100",
        "chongqing": "101040100",
        "guangzhou": "101280101",
        "shenzhen": "101280601",
        "hangzhou": "101210101",
        "chengdu": "101270101",
        "wuhan": "101200101",
        "nanjing": "101190101",
        "suzhou": "101190401",
        "xian": "101110101",
    }
)

# 城市名称 → 城市ID 缓存
_city_cache = {}


def _get_api_key() -> str:
    return os.environ.get("QWEATHER_API_KEY", "")


def _lookup_city_id(city_name: str) -> str | None:
    """将中文城市名转为和风天气城市ID，优先查内置映射表"""
    if city_name in _city_cache:
        return _city_cache[city_name]

    # 先从内置映射表查
    if city_name in CITY_ID_MAP:
        _city_cache[city_name] = CITY_ID_MAP[city_name]
        return CITY_ID_MAP[city_name]

    # 没找到则尝试调用 geo API（兼容不支持自定义host的geo接口）
    api_key = _get_api_key()
    if api_key:
        try:
            resp = requests.get(
                "https://geoapi.qweather.com/v2/city/lookup",
                params={"location": city_name, "key": api_key},
                timeout=5,
            )
            data = resp.json()
            if data.get("code") == "200" and data.get("location"):
                city_id = data["location"][0]["id"]
                _city_cache[city_name] = city_id
                return city_id
        except Exception:
            pass

    return None


def get_weather_data(city: str) -> str:
    """
    获取指定城市的实时天气数据
    返回简短的天气文本
    """
    api_key = _get_api_key()
    if not api_key:
        return "天气服务未配置"

    city_id = _lookup_city_id(city)
    if not city_id:
        return f"未找到城市「{city}」的天气信息"

    try:
        resp = requests.get(
            WEATHER_NOW_URL,
            params={"location": city_id, "key": api_key},
            timeout=5,
        )
        now_data = resp.json().get("now") if resp.json().get("code") == "200" else None
    except Exception:
        now_data = None

    if not now_data:
        return f"暂时无法获取城市「{city}」的天气数据"

    return (
        f"【{city}实时天气】"
        f"{now_data.get('text', '未知')}，"
        f"{now_data.get('temp', 'N/A')}°C，"
        f"湿度{now_data.get('humidity', 'N/A')}%，"
        f"{now_data.get('windDir', 'N/A')}{now_data.get('windScale', 'N/A')}级"
    )
