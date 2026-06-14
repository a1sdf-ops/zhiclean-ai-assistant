"""
知识库 RAG 问答系统 - 命令行交互式入口

运行方式：在项目根目录下执行
    python RAG部分/demo.py

前提：已配置 .env 中的 DASHSCOPE_API_KEY，并已上传 docs/ 下的知识文档
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RAG部分.knowledge_base import KnowledgeBaseService
from RAG部分.rag import RagService


def main():
    kb = KnowledgeBaseService()
    result = kb.list_knowledge(page=1, page_size=10)

    print()
    print("=" * 55)
    print("  智净 AI 售后助手 — 知识库 RAG 问答")
    print("=" * 55)

    if "error" in result:
        print(f"\n  [错误] 读取知识库失败: {result['error']}")
    elif result.get("data"):
        print(f"\n  已加载 {result['total']} 篇产品知识文档：")
        for item in result["data"]:
            print(f"    · {item['source']}")
    else:
        print("\n  [提示] 知识库为空，请先上传 docs/ 目录下的文档")
    print()

    rag = RagService()

    print("  输入问题（示例：Z2 Pro的滤网怎么保养？）")
    print("  输入 q / quit / exit 退出")
    print("-" * 55)
    session_id = "demo_session"

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
            for token in rag.ask_stream(query, session_id=session_id):
                print(token, end="", flush=True)
        except Exception as e:
            print(f"\n  [错误] {e}")
        print()


if __name__ == "__main__":
    main()
