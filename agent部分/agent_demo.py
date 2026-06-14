"""
Agent 交互式入口 — ReAct Agent + MCP + RAG 完整链路演示

启动:
  cd agent部分
  python agent_demo.py

前提: 已在项目根目录 .env 配置 DASHSCOPE_API_KEY，RAG部分 已有上传的知识文档
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent部分.react_agent import ReactAgent


def main():
    print()
    print("=" * 55)
    print("  智净 AI 售后助手 — Agent 智能服务")
    print("  Agent → 意图分类 → 工具调度 → MCP/RAG")
    print("=" * 55)
    print()

    print("  正在初始化 Agent（加载 StateGraph + 模型）...")
    agent = ReactAgent()
    print("  初始化完成！")
    print()
    print("-" * 55)
    print("  输入问题（输入 q / quit / exit 退出）")
    print("  示例：")
    print("    · Z2 Pro滤网怎么保养？")
    print("    · 北京今天什么天气，适不适合开窗通风？")
    print("    · 帮我生成这个月的使用报告")
    print("    · Z3 Ultra和Z2 Pro有什么区别？")
    print("-" * 55)

    while True:
        try:
            query = input("\n  你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  再见！")
            break

        if not query:
            continue
        if query.lower() in ("q", "quit", "exit"):
            print("  再见！")
            break

        print("\n  智净 > ", end="", flush=True)
        try:
            for token in agent.execute_stream(query):
                try:
                    token.encode("utf-8")
                except UnicodeEncodeError:
                    token = token.encode("utf-8", errors="replace").decode("utf-8")
                print(token, end="", flush=True)
        except Exception as e:
            print(f"\n  [错误] {e}")
        print()


if __name__ == "__main__":
    main()
