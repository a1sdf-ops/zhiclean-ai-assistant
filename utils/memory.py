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

MEMORY_EXTRACTION_PROMPT = """分析以下对话，提取对后续交互有价值的用户信息。以 JSON 数组返回，每条包含：

- "fact": 事实描述（简明扼要，一句以内）
- "category": 类别（preference / identity / context / knowledge）
- "importance": 重要性 1-5（5=必须记住，1=顺便提及）

只提取用户相关的信息，不提取助手回答中的通用知识。如果没有值得长期记忆的信息，返回空数组 []。

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

    def save(self, user_msg: str, assistant_msg: str, session_id: str = "default") -> list[dict]:
        """从一轮对话中提取事实并持久化，返回提取到的事实列表"""
        if not self._enabled or self.store is None:
            return []

        facts = self._extract_facts(user_msg, assistant_msg)
        if not facts:
            return []

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        texts, metadatas, ids = [], [], []

        for i, f in enumerate(facts):
            texts.append(f["fact"])
            metadatas.append(
                {
                    "session_id": session_id,
                    "category": f.get("category", "context"),
                    "importance": f.get("importance", 3),
                    "timestamp": timestamp,
                    "source": "agent_memory",
                }
            )
            ids.append(f"mem_{session_id}_{timestamp.replace(':', '').replace(' ', '_')}_{i}")

        try:
            self.store.add_texts(texts, metadatas=metadatas, ids=ids)
            logger.info("记忆已存储: %d 条 (session=%s)", len(facts), session_id)
        except Exception as e:
            logger.warning("记忆存储失败: %s", e)

        return facts

    def _extract_facts(self, user_msg: str, assistant_msg: str) -> list[dict]:
        """用 LLM 从对话中提取关键事实"""
        try:
            model = create_chat_model(temperature=0.0)
            response = model.invoke(
                MEMORY_EXTRACTION_PROMPT.format(
                    user_msg=user_msg,
                    assistant_msg=assistant_msg,
                )
            )
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                if raw.endswith("```"):
                    raw = raw[:-3]
            facts = json.loads(raw)
            return facts if isinstance(facts, list) else []
        except Exception as e:
            logger.warning("记忆提取失败: %s", e)
            return []

    # ---------- 召回 ----------

    def recall(self, query: str, session_id: str = None, top_k: int = None) -> str:
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

            filter_dict = None
            if session_id:
                filter_dict = {"session_id": session_id}

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
