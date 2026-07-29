"""Agent 记忆管理器 —— 短期对话历史 + 长期语义记忆

短期记忆：FileChatMessageHistory（已有，按 session 存对话 JSON）
长期记忆：LLM 提取关键事实 → Embedding → ChromaDB 持久化 → 语义召回
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

MEMORY_EXTRACTION_PROMPT = """分析以下对话，提取两部分信息：

1. "facts": 对后续交互有价值的离散事实数组，每条包含：
   - "fact": 事实描述（简明扼要，一句以内）
   - "category": 类别（preference / identity / context / knowledge）
   - "importance": 重要性 1-5（5=必须记住，1=顺便提及）
   只提取用户相关的信息，不提取助手回答中的通用知识。没有则返回 []。

2. "profile_update": 用户画像更新。
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
       "consumables": {{"hepa_filter": {{"last_replaced": "2026-07"}}, "side_brush": {{...}}, "main_brush": {{...}}, "mop_pad": {{...}}, "dust_bag": {{...}}}}
     }}],
     "current_location": "当前居住城市",
     "locations": ["城市名(历史记录)"],
     "preferences": {{"tech_level": "入门/进阶/专家", "reply_style": "简要/详细步骤/表格对比"}},
     "purchase_intent": [{{"product": "产品名", "level": "高/中/低"}}],
     "service_history": [{{"date": "日期", "type": "维修/咨询/退换", "device": "设备型号", "description": "简述"}}],
     "question_entry": {{"category": "故障排查/保养维护/功能咨询/售后政策/耗材/产品对比/天气/闲聊", "device": "设备型号(可选)", "problem": "标准化问题描述(可选)", "query_summary": "一句话概括用户问题", "resolved": true/false}}
   }}
   如果本轮对话没有新的画像信息可提取，profile_update 为 null。
   question_entry 每轮对话必须提取，和其他字段独立。

返回 JSON：{{"facts": [...], "profile_update": {{...}} 或 null}}
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

    def save(self, user_msg: str, assistant_msg: str, session_id: str = "default", tenant_id: str = "default") -> tuple[list[dict], dict | None]:
        """从一轮对话中提取事实并持久化，返回 (事实列表, 画像更新)"""
        if not self._enabled or self.store is None:
            return [], None

        facts, profile_update = self._extract_facts(user_msg, assistant_msg)
        if not facts and not profile_update:
            return [], None

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if facts:
            texts, metadatas, ids = [], [], []
            for i, f in enumerate(facts):
                texts.append(f["fact"])
                metadatas.append(
                    {
                        "tenant_id": tenant_id,
                        "session_id": session_id,
                        "category": f.get("category", "context"),
                        "importance": f.get("importance", 3),
                        "timestamp": timestamp,
                        "source": "agent_memory",
                    }
                )
                ids.append(f"mem_{tenant_id}_{session_id}_{timestamp.replace(':', '').replace(' ', '_')}_{i}")

            try:
                self.store.add_texts(texts, metadatas=metadatas, ids=ids)
                logger.info("记忆已存储: %d 条 (tenant=%s session=%s)", len(facts), tenant_id, session_id)
            except Exception as e:
                logger.warning("记忆存储失败: %s", e)

        return facts, profile_update

    def _extract_facts(self, user_msg: str, assistant_msg: str) -> tuple[list[dict], dict | None]:
        """用 LLM 从对话中提取关键事实 + 画像更新"""
        raw = ""
        try:
            import time as _time

            from agent.token_tracker import estimate_tokens, get_tracker

            model = create_chat_model(temperature=0.0)
            # 用 replace 而非 .format()：助手的回答中可能包含 JSON
            # （如 {"model": "Z2 Pro"}），.format() 会把花括号当占位符
            prompt = MEMORY_EXTRACTION_PROMPT.replace("{user_msg}", user_msg).replace("{assistant_msg}", assistant_msg)
            t0 = _time.time()
            response = model.invoke(prompt)
            latency = (_time.time() - t0) * 1000
            # Token 埋点: 记忆提取
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

            facts = result.get("facts", []) if isinstance(result, dict) else []
            profile_update = result.get("profile_update") if isinstance(result, dict) else None

            logger.info("记忆提取LLM: %d 条事实, profile=%s | latency=%.0fms",
                        len(facts) if isinstance(facts, list) else 0,
                        "有" if profile_update else "无",
                        latency)
            return (facts if isinstance(facts, list) else []), profile_update
        except Exception as e:
            logger.warning("记忆提取失败: %s | raw=%s", e, raw[:200] if raw else "(无输出)")
            return [], None

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
                importance = doc.metadata.get("importance", 3)
                timestamp = doc.metadata.get("timestamp", "")
                cat = doc.metadata.get("category", "")
                lines.append(f"[重要性:{importance}] [{cat}] {doc.page_content}")

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
                results.append({"id": data["ids"][i], "fact": text, **meta})
            return sorted(results, key=lambda x: x.get("importance", 0), reverse=True)
        except Exception as e:
            logger.warning("列出记忆失败: %s", e)
            return []
