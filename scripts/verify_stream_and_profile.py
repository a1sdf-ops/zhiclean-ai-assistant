"""流式输出 + 画像系统 联合验证脚本

三阶段:
  阶段一: SSE Token 级流式验证 (httpx 消费 POST /stream)
  阶段二: 画像系统 4 轮对话 (POST /chat)
  阶段三: 跨会话召回 (POST /session/close + 新 session)
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import httpx

AUTO = "--auto" in sys.argv
BASE = "http://localhost:8000/api/v1/agent"
SESSION = "verify_stream_session"
TENANT = f"tenant_{hashlib.md5(SESSION.encode()).hexdigest()[:12]}"

ROOT = Path(__file__).resolve().parent.parent
PROFILE_DIR = ROOT / "data" / "profiles"
PROFILE_FILE = PROFILE_DIR / f"{TENANT}.json"

PASS = 0
FAIL = 0


def pause(prompt_text: str):
    """自动模式下跳过 input，手动模式下暂停等待确认"""
    if AUTO:
        print(f"  [AUTO] 跳过检查点: {prompt_text[:60]}...")
        time.sleep(0.5)
    else:
        input(prompt_text)


def check(label: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}  -- {detail}")


# ============================================================
# 阶段一: SSE Token 级流式验证
# ============================================================
print("=" * 60)
print("阶段一: SSE Token 级流式验证")
print("=" * 60)

tokens = []
token_times = []
first_token_time = None

with httpx.Client(timeout=httpx.Timeout(120.0)) as client:
    t0 = time.time()
    with client.stream(
        "POST",
        f"{BASE}/stream",
        json={
            "query": "请详细介绍一下扫地机器人的日常保养方法，包括滤网清洗、主刷清理、传感器擦拭和废水处理四个部分",
            "session_id": SESSION,
            "tenant_id": TENANT,
        },
    ) as resp:
        print(f"  HTTP 状态: {resp.status_code}")
        for line in resp.iter_lines():
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    tok = data.get("token", "")
                    now = time.time()
                    if first_token_time is None:
                        first_token_time = now
                        ttft = now - t0
                    tokens.append(tok)
                    token_times.append(now)
                except json.JSONDecodeError:
                    pass

    total = time.time() - t0
    answer = "".join(tokens)

print(f"\n  首 token 延迟 (TTFT): {ttft:.2f}s")
print(f"  Token 总数: {len(tokens)}")
print(f"  总耗时: {total:.2f}s")
print(f"  回答前 150 字:\n  {answer[:150]}...\n")

# 验证项
check("HTTP 200", resp.status_code == 200, f"status={resp.status_code}")
check("收到 token 数 > 5", len(tokens) > 5, f"tokens={len(tokens)}")
check("首 token < 5s", ttft < 5.0, f"TTFT={ttft:.1f}s")

# 真流式判定: 至少 80% 的 token 间隔 < 500ms
if len(token_times) > 1:
    intervals = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
    fast_ratio = sum(1 for dt in intervals if dt < 0.5) / len(intervals)
    check(
        f"真流式 (间隔<500ms 占比 {fast_ratio:.0%})",
        fast_ratio > 0.8,
        f"fast_interval_ratio={fast_ratio:.1%}",
    )
    print(
        f"  Token 间隔统计: min={min(intervals) * 1000:.0f}ms max={max(intervals) * 1000:.0f}ms mean={sum(intervals) / len(intervals) * 1000:.0f}ms"
    )

if first_token_time:
    print(f"  [INFO] 首 token 在 {ttft:.1f}s 出现，之后 {total - ttft:.1f}s 内流式输出剩余 {len(tokens) - 1} tokens")
    if len(tokens) > 1:
        streaming_rate = (len(tokens) - 1) / max(total - ttft, 0.001)
        print(f"  [INFO] 流式速率: {streaming_rate:.0f} tokens/s")

pause("\n检查上面的 Token 间隔统计数据，确认真流式，按回车继续...")

# ============================================================
# 阶段二: 画像系统 4 轮验证
# ============================================================
print("\n" + "=" * 60)
print("阶段二: 画像系统 4 轮验证")
print("=" * 60)

messages = []  # 收集对话，给 close_session 用


def chat(query: str, session: str = SESSION) -> str:
    resp = httpx.post(
        f"{BASE}/chat",
        json={"query": query, "session_id": session, "tenant_id": TENANT},
        timeout=120.0,
    )
    data = resp.json()
    answer = data.get("answer", "")
    messages.append(f"用户: {query}")
    messages.append(f"助手: {answer}")
    return answer


# Round 1: 新用户
print("\n--- Round 1: 新用户首次对话 ---")
a1 = chat("你好，我在广州，刚买了Z3 Ultra，每天扫一次卧室瓷砖，怎么保养？")
print(f"  回答: {a1[:120]}...")
pause("检查 profiles/ 是否生成了新 JSON (model=Z3 Ultra, location=广州)，按回车继续...")

# Round 2: 画像增量
print("\n--- Round 2: 产品对比 ---")
a2 = chat("Z3 Ultra 和 Z2 Pro 有什么区别，值得升级吗？")
print(f"  回答: {a2[:120]}...")
pause("检查 question_history 是否增长、purchase_intent 是否有记录，按回车继续...")

# Round 3: 搬家 + 故障
print("\n--- Round 3: 搬家检测 + 故障提取 ---")
a3 = chat("我搬到了上海，最近吸力越来越小了，上次师傅周四来看过说滤网堵塞严重需要更换")
print(f"  回答: {a3[:120]}...")
pause("检查 current_location 是否为'上海'、issues 是否有'吸力'、service_history 是否有周四维修记录，按回车继续...")

# Round 4: 画像上下文验证
print("\n--- Round 4: 画像上下文是否被 LLM 使用 ---")
a4 = chat("根据我现在所在的城市和机器的使用情况，推荐一下保养建议")
print(f"  回答: {a4[:120]}...")

# 检查回答是否引用了画像上下文
combined_lower = a4.lower()
check("回答提到'上海'", "上海" in a4, "回答应引用画像中的城市")
check(
    "回答提到'Z3 Ultra'",
    "z3" in combined_lower and ("ultra" in combined_lower or "ultra" in combined_lower),
    "回答应引用画像中的设备",
)
check("回答提到'瓷砖'或'卧室'", "瓷砖" in a4 or "卧室" in a4 or "地板" in a4, "回答应引用使用场景")

# 验证画像文件
if PROFILE_FILE.exists():
    profile = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
    check(
        "current_location = 上海",
        profile.get("current_location") == "上海",
        f"actual={profile.get('current_location')}",
    )
    devices = profile.get("devices", [])
    check("有设备记录", len(devices) > 0, f"devices={len(devices)}")
    if devices:
        d = devices[0]
        issues = d.get("issues", [])
        has_suction = any("吸力" in (i.get("problem", "")) for i in issues)
        check("issues 包含吸力问题", has_suction)
    qh = profile.get("question_history", [])
    check("question_history >= 4 条", len(qh) >= 4, f"actual={len(qh)}")
    sh = profile.get("service_history", [])
    check("service_history 有记录", len(sh) > 0, f"actual={len(sh)}")
    if sh:
        check("service_history 提到周四", any("周四" in str(s) for s in sh))

pause("\n检查画像文件内容，确认各字段正确，按回车继续...")

# ============================================================
# 阶段三: 跨会话召回
# ============================================================
print("\n" + "=" * 60)
print("阶段三: 跨会话召回")
print("=" * 60)

# 先关闭会话，生成 ChromaDB 摘要
print("\n--- 关闭会话 ---")
resp = httpx.post(
    f"{BASE}/session/close",
    json={"session_id": SESSION, "tenant_id": TENANT, "messages": messages},
    timeout=120.0,
)
close_data = resp.json()
print(f"  success={close_data.get('success')}, summary_length={close_data.get('summary_length')}")
if close_data.get("summary"):
    print(f"  摘要: {close_data['summary'][:200]}...")
check("session/close 成功", close_data.get("success") is True)

# 新 session，同一 tenant
SESSION_CROSS = "verify_cross_session"
print(f"\n--- 新 Session ({SESSION_CROSS}) 跨会话召回 ---")

resp = httpx.post(
    f"{BASE}/chat",
    json={
        "query": "上次说的那个滤网堵塞的问题，更换滤网后还需要做什么吗，吸力问题还在吗",
        "session_id": SESSION_CROSS,
        "tenant_id": TENANT,
    },
    timeout=120.0,
)
cross_answer = resp.json().get("answer", "")
print(f"  回答: {cross_answer[:200]}...")

# 检查是否能跨会话接上上下文
combined = cross_answer.lower()
check(
    "跨会话: 回答提到'滤网'或'吸力'",
    "滤网" in cross_answer or "吸力" in cross_answer or "滤" in cross_answer,
    f"answer preview: {cross_answer[:80]}",
)

# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 60)
print(f"验证完成: {PASS} PASS / {FAIL} FAIL / {PASS + FAIL} TOTAL")
print("=" * 60)

if FAIL > 0:
    print("\n请检查上面 [FAIL] 项，排查原因。")
    sys.exit(1)
else:
    print("全部通过!")
