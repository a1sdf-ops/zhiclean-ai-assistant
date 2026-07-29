"""按 trace_id 从日志中提取延迟数据，供面试比对使用。

用法：
  python scripts/export_trace.py <trace_id>              # 查询单条 trace
  python scripts/export_trace.py --today                 # 列出今天所有 trace
  python scripts/export_trace.py --today --summary       # 今天的延迟汇总
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# 日志格式: 2026-07-28 14:20:14 | INFO    | trace_id | logger | file:line | message
LOG_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| (\w+)\s*\| (\S+) \| (\S+) \| (\S+):(\d+) \| (.*)")

# 关键延迟行匹配
LATENCY_PATTERNS = {
    "意图分类": re.compile(r"意图分类: (\S+) \|.*latency=(\d+)ms"),
    "最终回答生成": re.compile(r"最终回答生成完成:.*latency=(\d+)ms"),
    "记忆提取LLM": re.compile(r"记忆提取LLM: (\d+) 条事实 \| latency=(\d+)ms"),
    "记忆存储": re.compile(r"记忆已存储: (\d+) 条"),
    "记忆召回": re.compile(r"记忆召回: (\d+) 条"),
    "总请求耗时": re.compile(r"(POST|GET) (\S+) -> (\d+) \((\d+)ms\)"),
}


def parse_log_file(filepath):
    """解析单个日志文件，返回按 trace_id 分组的事件列表"""
    traces = defaultdict(list)
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            m = LOG_PATTERN.match(line.strip())
            if not m:
                continue
            ts, level, trace_id, logger, filename, lineno, msg = m.groups()
            traces[trace_id].append(
                {
                    "ts": ts,
                    "level": level,
                    "logger": logger,
                    "file": filename,
                    "line": int(lineno),
                    "msg": msg,
                }
            )
    return traces


def extract_latency(events):
    """从事件列表中提取延迟数据"""
    result = {
        "trace_id": None,
        "start": None,
        "end": None,
        "intent": None,
        "intent_latency": None,
        "generate_latency": None,
        "memory_extract_facts": None,
        "memory_extract_latency": None,
        "memory_stored": None,
        "memory_recalled": None,
        "total_ms": None,
        "endpoint": None,
    }
    for e in events:
        if result["trace_id"] is None and e["logger"] == "agent-app":
            result["trace_id"] = e.get("trace_id") or result["trace_id"]

        # 解析延迟
        for pattern_name, pattern in LATENCY_PATTERNS.items():
            m = pattern.search(e["msg"])
            if m:
                if pattern_name == "意图分类":
                    result["intent"] = m.group(1)
                    result["intent_latency"] = int(m.group(2))
                elif pattern_name == "最终回答生成":
                    result["generate_latency"] = int(m.group(1))
                elif pattern_name == "记忆提取LLM":
                    result["memory_extract_facts"] = int(m.group(1))
                    result["memory_extract_latency"] = int(m.group(2))
                elif pattern_name == "记忆存储":
                    result["memory_stored"] = int(m.group(1))
                elif pattern_name == "记忆召回":
                    result["memory_recalled"] = int(m.group(1))
                elif pattern_name == "总请求耗时":
                    result["endpoint"] = m.group(2)
                    result["total_ms"] = int(m.group(4))

        if e["level"] == "INFO" and "开始" in e.get("msg", ""):
            result["start"] = e["ts"]
        if result["total_ms"] and result["start"]:
            result["end"] = e["ts"]

    # 如果没有匹配到总请求，用第一个和最后一个事件的时间戳估算
    if result["total_ms"] is None and len(events) >= 2:
        try:
            t0 = datetime.strptime(events[0]["ts"], "%Y-%m-%d %H:%M:%S")
            t1 = datetime.strptime(events[-1]["ts"], "%Y-%m-%d %H:%M:%S")
            result["total_ms"] = int((t1 - t0).total_seconds() * 1000)
        except Exception:
            pass

    return result


def find_log_files(date=None):
    """查找日志文件"""
    pattern = os.path.join(config.LOG_DIR, "app.log*")
    if date:
        pattern = os.path.join(config.LOG_DIR, f"app.log.{date}")
    return sorted(glob(pattern), reverse=True)


def cmd_show(trace_id):
    """查询单条 trace"""
    log_files = find_log_files()
    if not log_files:
        print("未找到日志文件")
        return

    for lf in log_files:
        traces = parse_log_file(lf)
        if trace_id in traces:
            events = traces[trace_id]
            latency = extract_latency(events)
            print(f"\n{'=' * 60}")
            print(f"Trace:  {trace_id}")
            print(f"文件:  {lf}")
            print(f"{'=' * 60}")
            print(f"端点:    {latency['endpoint'] or 'N/A'}")
            print(
                f"总耗时:  {latency['total_ms']}ms ({latency['total_ms'] / 1000:.1f}s)"
                if latency["total_ms"]
                else "总耗时:  N/A"
            )
            print("--- LLM 调用拆解 ---")
            print(
                f"意图分类:   {latency['intent'] or 'N/A'} | {latency['intent_latency']}ms"
                if latency["intent_latency"]
                else "意图分类:   N/A"
            )
            print(f"回答生成:   {latency['generate_latency']}ms" if latency["generate_latency"] else "回答生成:   N/A")
            mem_lat = latency["memory_extract_latency"]
            mem_n = latency["memory_extract_facts"]
            print(f"记忆提取:   {mem_n}条事实 | {mem_lat}ms" if mem_lat else "记忆提取:   N/A")
            print("--- 完整事件时间线 ---")
            for e in events:
                print(f"  {e['ts']} | {e['level']:<7} | {e['file']}:{e['line']} | {e['msg'][:120]}")
            return

    print(f"未找到 trace_id={trace_id}，可能已被轮转清除或不存在")


def cmd_today_summary():
    """汇总今天所有 trace 的延迟"""
    today = datetime.now().strftime("%Y-%m-%d")
    log_files = find_log_files()
    if not log_files:
        print("未找到日志文件")
        return

    # 找今天的日志：先读当前 app.log，再找 app.log.YYYY-MM-DD
    today_files = []
    for lf in log_files:
        if lf.endswith(".log") or today in lf:
            today_files.append(lf)

    all_traces = {}
    for lf in today_files:
        traces = parse_log_file(lf)
        all_traces.update(traces)

    # 过滤出有实际请求的 trace（有 Agent 开始事件）
    request_traces = {}
    for tid, events in all_traces.items():
        for e in events:
            if "Agent 开始" in e["msg"]:
                request_traces[tid] = events
                break

    print(f"\n今天 ({today}) 共 {len(request_traces)} 次请求:\n")
    print(f"{'Trace ID':<16} {'意图':<20} {'意图耗时':>8} {'生成耗时':>8} {'记忆耗时':>8} {'总耗时':>10}")
    print("-" * 80)

    for tid, events in sorted(request_traces.items(), key=lambda x: x[1][0]["ts"]):
        lat = extract_latency(events)
        intent_str = lat["intent"] or "N/A"
        print(
            f"{tid:<16} {intent_str:<20} "
            f"{lat['intent_latency'] or '-':>6}ms "
            f"{lat['generate_latency'] or '-':>6}ms "
            f"{lat['memory_extract_latency'] or '-':>6}ms "
            f"{lat['total_ms'] or '-':>8}ms"
        )

    # 统计
    intents = [lat["intent_latency"] for lat in map(extract_latency, request_traces.values()) if lat["intent_latency"]]
    gens = [lat["generate_latency"] for lat in map(extract_latency, request_traces.values()) if lat["generate_latency"]]
    totals = [lat["total_ms"] for lat in map(extract_latency, request_traces.values()) if lat["total_ms"]]

    if totals:
        print("\n--- 统计 ---")
        print(f"平均总耗时: {sum(totals) / len(totals):.0f}ms ({sum(totals) / len(totals) / 1000:.1f}s)")
        if intents:
            print(f"平均意图分类: {sum(intents) / len(intents):.0f}ms")
        if gens:
            print(f"平均回答生成: {sum(gens) / len(gens):.0f}ms")


def main():
    parser = argparse.ArgumentParser(description="Agent 日志延迟分析工具")
    parser.add_argument("trace_id", nargs="?", help="查询指定 trace_id 的详细日志")
    parser.add_argument("--today", action="store_true", help="汇总今天的所有 trace")
    parser.add_argument("--summary", action="store_true", help="与 --today 配合，显示汇总表格")
    parser.add_argument("--date", help="查询指定日期的日志 (YYYY-MM-DD)")

    args = parser.parse_args()

    if args.today or args.summary:
        cmd_today_summary()
    elif args.trace_id:
        cmd_show(args.trace_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
