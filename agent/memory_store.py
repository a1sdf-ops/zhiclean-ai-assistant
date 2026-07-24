"""memory_store.py —— 生产级记忆存储：Redis Hash（短期）+ Redis Sorted Set（长期）

替代原有的纯 ChromaDB / JSON 文件方案，补齐三项工业级能力：
  1. 会话隔离 —— 每个 tenant_id + session_id 独立的 Redis key，天然多租户
  2. 自动过期 —— 短期记忆 72h TTL，到期自动清理，符合数据合规要求
  3. 权重衰减 —— 长期记忆按 DECAY_FACTOR 逐日衰减，低于 MIN_WEIGHT 淘汰

数据结构选型理由：
  Redis Hash   → 短期会话上下文（HGET O(1), HSET 原子写入, EXPIRE 原生 TTL）
  Redis Sorted Set → 长期偏好记忆（ZADD 按分数排序, ZREVRANGE 取 TopK, 衰减=乘以因子）

函数总结（按代码顺序）：
  get_redis_client      —— 创建 Redis 连接（单例复用）
  ShortTermMemory.save  —— 保存会话上下文（滑动窗口 + TTL）
  ShortTermMemory.load  —— 读取会话上下文
  ShortTermMemory.clear —— 手动清除会话
  LongTermMemory.save   —— 写入/更新一条长期记忆（ZADD，带重要性分数）
  LongTermMemory.topk   —— 取权重最高的 K 条记忆
  LongTermMemory.apply_decay —— 所有权重乘以衰减因子
  LongTermMemory.cleanup —— 删除低于阈值的记忆
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import TYPE_CHECKING, Optional

# 确保项目根目录在 sys.path 中（自测和外部 import 均可用）
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config  # noqa: E402  # 必须在 sys.path 设置之后才能 import

if TYPE_CHECKING:
    import redis


def get_redis_client():
    """懒加载 Redis 连接（延迟 import，避免未安装 redis 包时崩溃）"""
    import redis  # noqa: E402  # 延迟导入，graph.py 的 try/except 可以兜底

    return redis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        db=config.REDIS_DB,
        decode_responses=True,
    )


# ═══════════════════════════════════════════
# ShortTermMemory: Redis Hash
# ═══════════════════════════════════════════


class ShortTermMemory:
    """短期记忆 —— 会话级上下文缓存

    Key 格式: mem:st:{tenant_id}:{session_id}
    存储字段: last_intent, last_device, conversation_turn, recent_msgs(JSON), updated_at
    用途: 多轮对话中快速获取"刚才聊了什么"，避免重复推理
    """

    def __init__(self, r: redis.Redis | None = None, ttl_hours: int = None):
        self.r = r or get_redis_client()
        self.ttl = (ttl_hours or config.SHORT_MEM_TTL_HOURS) * 3600

    def _key(self, tenant_id: str, session_id: str) -> str:
        return f"mem:st:{tenant_id}:{session_id}"

    def save(
        self,
        tenant_id: str,
        session_id: str,
        messages: list,
        intent: str = "",
        tool_name: str = "",
        max_window: int = 20,
    ):
        """保存会话上下文到 Redis Hash（滑动窗口 + TTL）

        Args:
            tenant_id:   租户ID
            session_id:  会话ID
            messages:    最近 N 轮对话（LangChain Message 对象列表）
            intent:      当前意图
            tool_name:   最后调用的工具名
            max_window:  滑动窗口大小，只保留最近 N 轮
        """
        key = self._key(tenant_id, session_id)
        trimmed = messages[-max_window:]

        # 序列化 messages → JSON 字符串
        serialized = []
        for m in trimmed:
            if hasattr(m, "content"):
                serialized.append({"role": m.__class__.__name__, "content": m.content})
            elif isinstance(m, dict):
                serialized.append(m)

        pipe = self.r.pipeline()
        pipe.delete(key)
        pipe.hset(
            key,
            mapping={
                "recent_msgs": json.dumps(serialized, ensure_ascii=False),
                "conversation_turn": str(len(serialized)),
                "last_intent": intent,
                "last_tool": tool_name,
                "updated_at": str(time.time()),
            },
        )
        pipe.expire(key, self.ttl)
        pipe.execute()

    def load(self, tenant_id: str, session_id: str) -> dict:
        """读取会话上下文，返回字典（不存在则返回空 dict）"""
        key = self._key(tenant_id, session_id)
        data = self.r.hgetall(key)
        if not data:
            return {}

        # 反序列化 messages
        try:
            data["recent_msgs"] = json.loads(data.get("recent_msgs", "[]"))
        except json.JSONDecodeError:
            data["recent_msgs"] = []

        data["conversation_turn"] = int(data.get("conversation_turn", "0"))
        return data

    def clear(self, tenant_id: str, session_id: str):
        """手动清除会话（如用户请求删除数据）"""
        self.r.delete(self._key(tenant_id, session_id))


# ═══════════════════════════════════════════
# LongTermMemory: Redis Sorted Set
# ═══════════════════════════════════════════


class LongTermMemory:
    """长期记忆 —— 带权重的用户偏好存储

    Key 格式: mem:lt:{tenant_id}:{session_id}
    每条记忆: member=事实文本, score=权重(float)
    ZADD 新增/更新, ZREVRANGE 取 TopK, ZREMRANGEBYSCORE 清理低分

    衰减机制: 每天对所有记忆 score *= DECAY_FACTOR
              score < MIN_WEIGHT → 自动淘汰
              解决"老旧记忆永远占据排名"的记忆漂移问题
    """

    def __init__(self, r: redis.Redis | None = None):
        self.r = r or get_redis_client()
        self.decay_factor = config.DECAY_FACTOR  # 0.95
        self.min_weight = config.MIN_WEIGHT  # 0.1

    def _key(self, tenant_id: str, session_id: str) -> str:
        return f"mem:lt:{tenant_id}:{session_id}"

    def save(self, tenant_id: str, session_id: str, fact: str, importance: float = 3.0):
        """写入或更新一条长期记忆

        Args:
            fact:       记忆文本（如 "用户偏好客厅亮度50%"）
            importance: 初始权重 1.0~5.0，越高越重要

        时效性偏置：权重末尾附加一个微小的时间戳增量（约 0.00017/天），
        使得同重要度的新旧事实在 topk 排序中，更新的事实排在前面。
        这个增量足够小，不会让 importance=3 的新事实超过 importance=5 的旧事实，
        但足够大来打破「9条旧北京 vs 1条新深圳」的同权僵局。
        """
        key = self._key(tenant_id, session_id)
        # 归一化 importance 到 0.0~1.0，末尾附加时间戳微调打破平局
        base = max(0.0, min(1.0, importance / 5.0))
        bias = time.time() / 1e13  # 约 0.00017，肉眼不可见
        self.r.zadd(key, {fact: base + bias})

    def topk(self, tenant_id: str, session_id: str, k: int = 5) -> list[tuple[str, float]]:
        """取权重最高的 K 条记忆，返回 [(text, score), ...]"""
        key = self._key(tenant_id, session_id)
        items = self.r.zrevrange(key, 0, k - 1, withscores=True)
        return [(text, round(score, 4)) for text, score in items]

    def apply_decay(self, tenant_id: str, session_id: str):
        """所有权重乘以衰减因子（一天执行一次即可）"""
        key = self._key(tenant_id, session_id)
        items = self.r.zrange(key, 0, -1, withscores=True)
        if not items:
            return
        pipe = self.r.pipeline()
        for text, score in items:
            new_score = score * self.decay_factor
            if new_score >= self.min_weight:
                pipe.zadd(key, {text: new_score})
            else:
                pipe.zrem(key, text)
        pipe.execute()

    def cleanup(self, tenant_id: str, session_id: str):
        """删除低于 MIN_WEIGHT 的记忆（apply_decay 已含此逻辑，这里作独立入口）"""
        key = self._key(tenant_id, session_id)
        self.r.zremrangebyscore(key, 0, self.min_weight - 0.001)

    def all(self, tenant_id: str, session_id: str) -> list[tuple[str, float]]:
        """列出全部记忆（调试用），按权重降序"""
        key = self._key(tenant_id, session_id)
        items = self.r.zrevrange(key, 0, -1, withscores=True)
        return [(text, round(score, 4)) for text, score in items]

    def clear(self, tenant_id: str, session_id: str):
        """清除全部长期记忆"""
        self.r.delete(self._key(tenant_id, session_id))


# ═══════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════

if __name__ == "__main__":
    r = get_redis_client()
    print(f"Redis: {r.ping()}, keys={r.dbsize()}")

    tid, sid = "test_tenant", "test_session"

    # ── 短期记忆 ──
    stm = ShortTermMemory(r)
    stm.save(
        tid,
        sid,
        [
            {"role": "user", "content": "客厅灯开了吗"},
            {"role": "assistant", "content": "客厅灯已关闭"},
        ],
        intent="device_control",
        tool_name="light_check",
    )
    loaded = stm.load(tid, sid)
    print(f"短期记忆: turn={loaded.get('conversation_turn')}, intent={loaded.get('last_intent')}")
    assert loaded["conversation_turn"] == 2, "FAIL: short term turn count"
    print("  ShortTermMemory PASS")

    # ── 长期记忆 ──
    ltm = LongTermMemory(r)
    ltm.save(tid, sid, "用户偏好客厅亮度50%", importance=4.8)
    ltm.save(tid, sid, "用户习惯晚上8点后开灯", importance=4.0)
    ltm.save(tid, sid, "用户3个月前问过扫地机器人", importance=1.5)
    top = ltm.topk(tid, sid, k=3)
    print(f"长期记忆 TopK: {top}")
    assert len(top) == 3 and top[0][0] == "用户偏好客厅亮度50%", "FAIL: topk order"
    print("  LongTermMemory topk PASS")

    # ── 衰减 ──
    ltm.apply_decay(tid, sid)
    after = ltm.all(tid, sid)
    print(f"衰减后: {after}")
    # 1.5*0.95=1.425 → 仍然在，但最低的应该还在
    for text, score in after:
        assert score >= ltm.min_weight, f"FAIL: {text} below threshold"
    print("  LongTermMemory decay PASS")

    # ── 清理 ──
    stm.clear(tid, sid)
    ltm.clear(tid, sid)
    assert r.dbsize() == 0, "FAIL: cleanup"
    print("Cleanup PASS")
    print("\nAll tests passed.")
