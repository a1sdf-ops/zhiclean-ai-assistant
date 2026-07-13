"""token_tracker.py —— Token 成本埋点：分模块统计 + 装饰器无侵入注入

面试价值：字节火山引擎的商业化考核核心——"你的Agent一次对话烧了多少Token？"
         分模块统计（LLM生成 / RAG检索 / Reranker / 记忆提取），每项独立可量化。

函数总结：
  TokenTracker.record      —— 记录一次调用（模块名 + 输入/输出Token + 延迟）
  TokenTracker.report      —— 输出成本报告（分模块 + 总计）
  TokenTracker.reset       —— 重置统计
  track(module_name)       —— 装饰器，自动统计被装饰函数的Token和延迟
  estimate_tokens(text)    —— 从文本长度粗略估算Token数（中文约1.5token/字）
  get_tracker / get_report —— 全局单例访问入口
"""

import functools
import re
import time
from collections import defaultdict
from typing import Optional

from utils.logger_handler import logger


# ── Token 估算：中英文分开计算 ──
# Qwen/DashScope 实测: 中文约1.5~2字符/token，英文约3.5~4字符/token
# 本项目以中文为主，采用 中文//1.5 + 英文//4 的混合算法


def estimate_tokens(text) -> int:
    """从文本粗略估算 Token 数（精确值应从 API response.usage 获取）"""
    if not text:
        return 0
    if hasattr(text, "content"):
        text = text.content
    if isinstance(text, list):
        total = 0
        for msg in text:
            total += estimate_tokens(msg.content if hasattr(msg, "content") else str(msg))
        return total

    s = str(text)
    # 中文字符（含中文标点）
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', s))
    non_chinese = len(s) - chinese_chars
    # 中文 ~1.5 chars/token，非中文（英文/数字/标点）~4 chars/token
    tokens = int(chinese_chars / 1.5) + int(non_chinese / 4)
    return max(1, tokens)


# ═══════════════════════════════════════════
# TokenTracker
# ═══════════════════════════════════════════

class TokenTracker:
    """分模块 Token 消耗 & 延迟统计"""

    def __init__(self):
        self.stats = defaultdict(lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "call_count": 0,
            "total_latency_ms": 0.0,
        })

    def record(self, module: str, input_tokens: int,
               output_tokens: int, latency_ms: float):
        """记录一次调用"""
        s = self.stats[module]
        s["input_tokens"] += input_tokens
        s["output_tokens"] += output_tokens
        s["call_count"] += 1
        s["total_latency_ms"] += latency_ms

    def report(self) -> dict:
        """输出成本报告"""
        result = {}
        grand_input = 0
        grand_output = 0

        for module, s in sorted(self.stats.items()):
            total = s["input_tokens"] + s["output_tokens"]
            avg_lat = (s["total_latency_ms"] / s["call_count"]
                       if s["call_count"] > 0 else 0)
            result[module] = {
                "total_tokens": total,
                "input_tokens": s["input_tokens"],
                "output_tokens": s["output_tokens"],
                "call_count": s["call_count"],
                "avg_latency_ms": round(avg_lat, 2),
                "pct": 0.0,  # 后面填充
            }
            grand_input += s["input_tokens"]
            grand_output += s["output_tokens"]

        # 计算各模块占比
        grand_total = grand_input + grand_output
        for module in result:
            if grand_total > 0:
                result[module]["pct"] = round(
                    (result[module]["total_tokens"] / grand_total) * 100, 1
                )

        result["__total__"] = {
            "total_input_tokens": grand_input,
            "total_output_tokens": grand_output,
            "total_tokens": grand_total,
        }
        return result

    def reset(self):
        self.stats.clear()


# ── 全局单例 ──

_tracker: Optional[TokenTracker] = None


def get_tracker() -> TokenTracker:
    global _tracker
    if _tracker is None:
        _tracker = TokenTracker()
    return _tracker


def get_report() -> dict:
    return get_tracker().report()


def reset_tracker():
    get_tracker().reset()


# ── 装饰器 ──

def track(module_name: str):
    """装饰器：自动统计被装饰函数的 Token 消耗和耗时

    用法:
        @track("llm_intent_classifier")
        def classify(prompt): ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            result = func(*args, **kwargs)
            latency = (time.time() - t0) * 1000

            # 从参数估算输入 Token
            input_text = str(args[0]) if args else ""
            it = estimate_tokens(input_text)

            # 从返回值估算输出 Token
            output_text = result.content if hasattr(result, "content") else str(result)
            ot = estimate_tokens(output_text)

            get_tracker().record(module_name, it, ot, latency)
            logger.debug("TokenTracker [%s]: in=%d out=%d lat=%.1fms",
                         module_name, it, ot, latency)
            return result
        return wrapper
    return decorator


# ═══════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════

if __name__ == "__main__":
    tracker = get_tracker()

    # 模拟调用
    tracker.record("llm_intent_classifier",  input_tokens=800,  output_tokens=50,  latency_ms=320)
    tracker.record("llm_generation",        input_tokens=2500, output_tokens=200, latency_ms=850)
    tracker.record("llm_memory_extraction", input_tokens=600,  output_tokens=120, latency_ms=280)
    tracker.record("rag_retrieval",         input_tokens=1500, output_tokens=0,   latency_ms=45)
    tracker.record("reranker",              input_tokens=800,  output_tokens=0,   latency_ms=30)

    report = tracker.report()
    print("Token 成本报告:")
    for module, stats in report.items():
        if module == "__total__":
            print(f"  总计: {stats['total_tokens']} tokens")
        else:
            print(f"  [{module}] {stats['total_tokens']}t ({stats['pct']}%) "
                  f"in={stats['input_tokens']} out={stats['output_tokens']} "
                  f"calls={stats['call_count']} avg_lat={stats['avg_latency_ms']}ms")

    total = report.get("__total__", {})
    assert total["total_tokens"] == 6570, f"FAIL: total={total['total_tokens']}"
    # 2700/6570 ≈ 41.1%
    assert abs(report["llm_generation"]["pct"] - 41.1) < 0.5, f"FAIL: pct={report['llm_generation']['pct']}"
    print("\nAll tests passed.")
