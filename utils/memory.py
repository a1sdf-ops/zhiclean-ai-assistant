"""Agent 记忆管理器 —— 结构化画像 + 轻量向量兜底

画像（权威事实源）：LLM 每轮提取 profile_update → 增量合并到 JSON 文件
向量（兜底长尾）：会话结束时异步写入摘要到 ChromaDB，跨会话排障历史召回

架构决策：
- 旧 Redis 三层记忆（LLM 打分排序）已完全移除——消除记忆漂移、降低延迟
- Per-round facts 不再写入 ChromaDB——画像的 question_history 已覆盖逐轮记录
- ChromaDB 仅存 session summary，用于召回非结构化排障过程碎片
"""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_chroma import Chroma

import config
from model.factory import create_chat_model, create_embedding_model
from utils.logger_handler import logger

MEMORY_EXTRACTION_PROMPT = """分析以下对话，提取用户画像更新（profile_update）。

**主动推断规则**：
- 用户说"我的XX产品"、"XX怎么保养/使用/设置" → 推断拥有该设备，填写 devices
- 用户说"搬到了/搬到/移居/搬家到/现在在 某城市" → 提取 current_location
- 用户问产品对比、新品咨询 → 可能有意向，提取 purchase_intent
- 用户提到维修/报修/换货/售后经历 → 提取 service_history
- 用户问"最近XX怎么不行了"等持续性问题 → 推断问题未解决

**可提取字段**（只填写有实际依据的字段，不编造）：
{{
  "devices": [{{
    "model": "产品型号",
    "purchased": "购买日期(可选)",
    "usage": {{"frequency": "每天一次", "primary_area": "客厅", "floor_type": "木地板", "has_pets": false}},
    "issues": [{{"problem": "问题描述", "status": "未解决/已解决", "attempted_solutions": ["已尝试方案"]}}],
    "consumables": [{{"name": "hepa_filter", "last_replaced": "2026-07", "cycle_days": 90}}]
  }}],
  "current_location": "当前居住城市",
  "preferences": {{"receive_maintain_remind": true/false, "remind_time": "每周一/每月1号/..."}},
  "purchase_intent": [{{"product": "产品名", "level": "高/中/低"}}],
  "service_history": [{{"service_type": "上门维修/咨询/退换", "target_device": "设备型号", "service_time": "日期", "result": "服务结果", "resolved": true/false}}],
  "question_entry": {{"category": "故障排查/保养维护/功能咨询/售后政策/耗材/产品对比/天气/闲聊", "device": "设备型号(可选)", "problem": "标准化问题描述(可选)", "query_summary": "一句话概括用户问题", "resolved": true/false}}
}}
如果本轮对话没有新的画像信息可提取，返回 {{"profile_update": null}}。
question_entry 每轮对话必须提取，和其他字段独立。

返回 JSON：{{"profile_update": {{...}} 或 null}}
仅输出 JSON，不要其他内容。

对话：
用户: {user_msg}
助手: {assistant_msg}
"""


class MemoryManager:
    """长期记忆管理器：基于 ChromaDB 的语义记忆存储与召回"""

    def __init__(self):
        self._store = None
        self._enabled = getattr(config, "ENABLE_MEMORY", True)

    @property
    def store(self):
        if self._store is None and self._enabled:
            import os

            persist_dir = os.path.join(config.DATA_DIR, "memory_db")
            self._store = Chroma(
                collection_name=getattr(config, "MEMORY_COLLECTION", "agent_memories"),
                embedding_function=create_embedding_model(),
                persist_directory=persist_dir,
            )
        return self._store

    # ---------- 写入 ----------

    def save(
        self, user_msg: str, assistant_msg: str, session_id: str = "default", tenant_id: str = "default"
    ) -> dict | None:
        """从一轮对话中提取画像更新并返回，不再写入 ChromaDB per-round facts"""
        if not self._enabled:
            return None

        profile_update = self._extract_profile(user_msg, assistant_msg)
        return profile_update

    def _extract_profile(self, user_msg: str, assistant_msg: str) -> dict | None:
        """用 LLM 从对话中提取画像更新（不再提取 facts）"""
        raw = ""
        try:
            import time as _time

            from agent.token_tracker import estimate_tokens, get_tracker

            model = create_chat_model(temperature=0.0)
            prompt = MEMORY_EXTRACTION_PROMPT.replace("{user_msg}", user_msg).replace("{assistant_msg}", assistant_msg)
            t0 = _time.time()
            response = model.invoke(prompt)
            latency = (_time.time() - t0) * 1000
            get_tracker().record(
                "llm_memory_extraction",
                input_tokens=estimate_tokens(prompt),
                output_tokens=estimate_tokens(response),
                latency_ms=latency,
            )

            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                if raw.endswith("```"):
                    raw = raw[:-3]
            result = json.loads(raw)

            profile_update = result.get("profile_update") if isinstance(result, dict) else None

            logger.info(
                "画像提取: profile=%s | latency=%.0fms",
                "有" if profile_update else "无",
                latency,
            )
            return profile_update
        except Exception as e:
            logger.warning("画像提取失败: %s | raw=%s", e, raw[:200] if raw else "(无输出)")
            return None

    def save_session_summary(self, tenant_id: str, session_id: str,
                             summary: str, metadata: dict = None) -> bool:
        """会话结束时异步写入摘要到 ChromaDB，兜底跨会话排障历史碎片"""
        if not self._enabled or self.store is None:
            return False
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            doc_id = f"summary_{tenant_id}_{session_id}_{timestamp.replace(':', '').replace(' ', '_')}"
            self.store.add_texts(
                [summary],
                metadatas=[{
                    "tenant_id": tenant_id,
                    "session_id": session_id,
                    "type": "session_summary",
                    "timestamp": timestamp,
                    **(metadata or {}),
                }],
                ids=[doc_id],
            )
            logger.info("会话摘要已存储: tenant=%s session=%s", tenant_id, session_id)
            return True
        except Exception as e:
            logger.warning("会话摘要存储失败: %s", e)
            return False

    def close_session(self, messages: list[str], tenant_id: str, session_id: str) -> str | None:
        """用 LLM 生成会话摘要并写入 ChromaDB

        messages: 用户和助手的交替对话文本列表，如 ["用户: 你好", "助手: 你好！...", ...]
        返回: 生成的摘要文本，失败返回 None
        """
        if not self._enabled or self.store is None or not messages:
            return None

        conversation = "\n".join(messages)
        summary_prompt = """分析以下售后对话，提取一份排障过程摘要，聚焦 **JSON 用户画像 schema 无法覆盖的碎片化细节**：

- 维修/服务过程细节（师傅什么时候来、配件是否缺货、补发时间）
- 故障的触发条件（"高温才复现""下雨天才出现"）
- 用户提到的临时信息（"下个月要搬家""最近在装修"）
- 未解决的遗留问题及原因
- 客服给过的临时方案或承诺

不要重复用户画像已有的结构化信息（设备型号、故障类型、城市等），只提取 schema 以外的排障过程碎片。
用中文简明扼要，控制在 200 字以内。只输出摘要本身，不要加前缀。

对话：
""" + conversation

        try:
            model = create_chat_model(temperature=0.0)
            response = model.invoke(summary_prompt)
            summary = response.content.strip()
            if summary:
                self.save_session_summary(tenant_id, session_id, summary)
                logger.info("会话摘要生成成功: %d 字 (session=%s)", len(summary), session_id)
                return summary
            return None
        except Exception as e:
            logger.warning("会话摘要生成失败: %s", e)
            return None

    # ---------- 召回 ----------

    def recall(self, query: str, session_id: str = None, tenant_id: str = None, top_k: int = None) -> str:
        """语义搜索相关长期记忆，返回拼接后的上下文字符串"""
        if not self._enabled or self.store is None:
            return ""

        if top_k is None:
            top_k = getattr(config, "MEMORY_TOP_K", 4)

        try:
            # 检查是否有数据
            count = self.store._collection.count()
            if count == 0:
                return ""

            # ChromaDB where 多条件需用 $and 包裹
            conditions = []
            if tenant_id:
                conditions.append({"tenant_id": tenant_id})
            if session_id:
                conditions.append({"session_id": session_id})
            if len(conditions) == 1:
                filter_dict = conditions[0]
            elif len(conditions) > 1:
                filter_dict = {"$and": conditions}
            else:
                filter_dict = None

            docs = self.store.similarity_search(query, k=min(top_k, count), filter=filter_dict)
            if not docs:
                return ""

            lines = []
            for doc in docs:
                ts = doc.metadata.get("timestamp", "")
                doc_type = doc.metadata.get("type", "")
                prefix = "[会话摘要]" if doc_type == "session_summary" else ""
                lines.append(f"{prefix} [{ts}] {doc.page_content}")

            logger.info("记忆召回: %d 条 (query=%s)", len(docs), query[:40])
            return "\n".join(lines)
        except Exception as e:
            logger.warning("记忆召回失败: %s", e)
            return ""

    # ---------- 管理 ----------

    def forget_session(self, session_id: str) -> int:
        """删除指定 session 的所有记忆，返回删除条数"""
        if not self._enabled or self.store is None:
            return 0
        try:
            results = self.store.get(where={"session_id": session_id})
            count = len(results.get("ids", []))
            if count > 0:
                self.store.delete(where={"session_id": session_id})
                logger.info("已清除 session=%s 的记忆 (%d 条)", session_id, count)
            return count
        except Exception as e:
            logger.warning("记忆清除失败: %s", e)
            return 0

    def all_memories(self, session_id: str = None) -> list[dict]:
        """列出所有记忆（调试用）"""
        if not self._enabled or self.store is None:
            return []
        try:
            where = {"session_id": session_id} if session_id else None
            data = self.store.get(include=["metadatas", "documents"], where=where)
            results = []
            for i, (text, meta) in enumerate(zip(data.get("documents", []), data.get("metadatas", []))):
                results.append({"id": data["ids"][i], "content": text, **meta})
            return sorted(results, key=lambda x: x.get("timestamp", ""), reverse=True)
        except Exception as e:
            logger.warning("列出记忆失败: %s", e)
            return []
