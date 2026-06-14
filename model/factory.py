"""模型工厂 —— 统一创建 LLM 和 Embedding 实例，支持多 provider 切换"""

import os
import sys
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import config


@lru_cache(maxsize=2)
def create_chat_model(model_name: str | None = None, temperature: float = 0.0) -> ChatOpenAI:
    model = model_name or config.CHAT_MODEL_NAME
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return ChatOpenAI(
            model=model,
            api_key=config.OPENAI_API_KEY or "sk-placeholder",
            base_url=config.OPENAI_BASE_URL,
            temperature=temperature,
        )
    return ChatOpenAI(
        model=model,
        api_key=config.DASHSCOPE_API_KEY,
        base_url=config.DASHSCOPE_BASE_URL,
        temperature=temperature,
    )


@lru_cache(maxsize=2)
def create_embedding_model(model_name: str | None = None):
    model = model_name or config.EMBEDDING_MODEL_NAME
    if model.startswith("text-embedding-3"):
        return OpenAIEmbeddings(
            model=model,
            api_key=config.OPENAI_API_KEY or "sk-placeholder",
            base_url=config.OPENAI_BASE_URL,
        )
    return DashScopeEmbeddings(
        model=model,
        dashscope_api_key=config.DASHSCOPE_API_KEY,
    )
