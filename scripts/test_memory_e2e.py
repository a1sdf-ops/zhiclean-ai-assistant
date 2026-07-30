"""端到端记忆测试脚本 —— 跑 4 轮对话 + close_session + 跨会话召回"""

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

BASE = "http://localhost:8000/api/v1/agent"

SESSION = "e2e_test_session"
TENANT = f"tenant_{hashlib.md5(SESSION.encode()).hexdigest()[:12]}"  # 固定 tenant，跨 session 共用

messages = []  # 收集完整对话，给 close_session 用


def chat(query: str, session_id: str = None):
    resp = requests.post(
        f"{BASE}/chat",
        json={
            "query": query,
            "session_id": session_id or SESSION,
            "tenant_id": TENANT,
        },
    )
    data = resp.json()
    answer = data.get("answer", "")
    messages.append(f"用户: {query}")
    messages.append(f"助手: {answer}")
    return answer


def close():
    resp = requests.post(
        f"{BASE}/session/close",
        json={
            "session_id": SESSION,
            "tenant_id": TENANT,
            "messages": messages,
        },
    )
    return resp.json()


# ── Round 1: 新用户 ──
print("=" * 50)
print("Round 1: 新用户首次对话")
print("=" * 50)
a1 = chat("你好，我在北京，刚买了Z3 Ultra，每天扫一次客厅木地板，怎么保养？")
print(f"  → {a1[:100]}...")

input("检查 profiles/ 是否生成了新 JSON，按回车继续...")

# ── Round 2: 画像召回 ──
print("\n" + "=" * 50)
print("Round 2: 产品对比（验证画像召回）")
print("=" * 50)
a2 = chat("Z3 Ultra 和 Z2 Pro 有什么区别，值得升级吗？")
print(f"  → {a2[:100]}...")

input("检查 question_history 是否有 2 条，按回车继续...")

# ── Round 3: 搬家 + 故障 ──
print("\n" + "=" * 50)
print("Round 3: 搬家检测 + 故障提取")
print("=" * 50)
a3 = chat("我搬到了深圳，最近吸力变小了怎么处理，上次维修师傅周三下午来过，说配件缺货下周补发")
print(f"  → {a3[:100]}...")

input("检查 current_location 是否为'深圳'、issues 是否有'吸力变小'、service_history 是否有记录，按回车继续...")

# ── Round 4: 画像上下文验证 ──
print("\n" + "=" * 50)
print("Round 4: 画像上下文能否被 LLM 使用")
print("=" * 50)
a4 = chat("根据我现在所在的城市和机器的使用情况，推荐一下保养建议")
print(f"  → {a4[:100]}...")

input("检查回答中是否提到'深圳'和'Z3 Ultra'（不需要用户重复），按回车继续...")

# ── 关闭会话 ──
print("\n" + "=" * 50)
print("关闭会话，生成 ChromaDB 摘要")
print("=" * 50)
result = close()
print(f"  success={result.get('success')}, summary_length={result.get('summary_length')}")
if result.get("summary"):
    print(f"  摘要: {result['summary']}")

input("按回车开始跨会话召回测试...")

# ── Round 5: 跨会话召回 ──
SESSION_CROSS = "e2e_cross_session"
print("\n" + "=" * 50)
print("Round 5: 新 session 跨会话召回")
print("=" * 50)

resp = requests.post(
    f"{BASE}/chat",
    json={
        "query": "上次说的那个配件补发到了吗，吸力问题还在",
        "session_id": SESSION_CROSS,
        "tenant_id": TENANT,  # 同一个 tenant，跨 session 复用画像 + ChromaDB 摘要
    },
)
print(f"  → {resp.json().get('answer', '')[:200]}")

print("\n" + "=" * 50)
print("测试完成。检查:")
print("  1. data/profiles/ 下的 JSON 结构是否完整")
print("  2. 日志中 Round 5 的 '记忆召回' 是否 semantic=有")
print("  3. Agent 回答是否能接上'配件缺货下周补发'的上下文")
print("=" * 50)
