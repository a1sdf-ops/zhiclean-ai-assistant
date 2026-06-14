import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_chroma import Chroma

import config


# 向量的存储服务
class VectorStoreService:
    def __init__(self, embedding):
        # param embedding:嵌入模型导入
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=self.embedding,
            persist_directory=config.CHROMA_PERSIST_DIR,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": config.RETRIEVER_K})
