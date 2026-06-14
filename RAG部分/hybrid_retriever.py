"""BM25 + 向量检索 + RRF 融合 —— 混合检索器

RRF (Reciprocal Rank Fusion):
  score(d) = Σ 1/(k + rank_i(d))
  其中 k=60（标准值），rank_i 是文档在第 i 个检索结果中的排名

融合后可选过 BGE-Reranker 做第二stage精排。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document

import config
from RAG部分.bm25 import BM25Retriever
from utils.logger_handler import logger


class HybridRetriever:
    """混合检索器：BM25 稀疏检索 + Chroma 向量检索，RRF 融合"""

    def __init__(self, vector_store, k1=None, b=None, rrf_k=60):
        self.vector_store = vector_store
        self.bm25 = BM25Retriever(
            k1=k1 if k1 is not None else config.BM25_K1,
            b=b if b is not None else config.BM25_B,
        )
        self.rrf_k = rrf_k
        self._docs: list[Document] = []
        self._doc_count = 0
        self._indexed = False

    def _ensure_index(self):
        """从 ChromaDB 拉取全部文档构建 BM25 倒排索引（文档数变化时自动重建）"""
        all_data = self.vector_store.get(include=["documents", "metadatas"])
        texts = all_data.get("documents", [])
        current_count = len(texts)

        if self._indexed and current_count == self._doc_count:
            return

        if current_count == 0:
            self._indexed = True
            self._doc_count = 0
            return

        metadatas = all_data.get("metadatas", [])
        ids = all_data.get("ids", [])
        self._docs = [Document(page_content=t, metadata=m or {}, id=i) for t, m, i in zip(texts, metadatas, ids)]
        self.bm25.index(texts)
        self._doc_count = current_count
        self._indexed = True
        logger.info("BM25 索引已更新: %d 篇文档", current_count)

    def invalidate(self):
        """强制下次检索时重建 BM25 索引"""
        self._indexed = False

    def search(self, query: str, top_k: int = None) -> list[Document]:
        """混合检索，返回 top_k 个 Document"""
        if top_k is None:
            top_k = config.RETRIEVER_K
        self._ensure_index()

        if self._doc_count == 0:
            return []

        k_candidates = max(top_k * 2, 16)

        # BM25 稀疏检索
        bm25_results = self.bm25.search(query, top_k=k_candidates)

        # 向量检索
        vector_results = self.vector_store.similarity_search_with_score(query, k=k_candidates)

        # RRF 融合
        rrf_scores: dict[int, float] = {}
        doc_by_idx: dict[int, Document] = {}

        for rank, (doc_id, _) in enumerate(bm25_results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (self.rrf_k + rank)
            doc_by_idx[doc_id] = self._docs[doc_id]

        for rank, (doc, _) in enumerate(vector_results, start=1):
            # 用 page_content 在 _docs 中反查 idx
            idx = self._find_doc_index(doc.page_content)
            if idx == -1:
                continue
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (self.rrf_k + rank)
            doc_by_idx[idx] = doc

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_by_idx[idx] for idx, _ in ranked[:top_k]]

    def _find_doc_index(self, text: str) -> int:
        """在 _docs 中查找 text 对应的下标，-1 表示未找到"""
        for i, doc in enumerate(self._docs):
            if doc.page_content == text:
                return i
        return -1

    def invoke(self, query: str, **kwargs) -> list[Document]:
        """兼容 LangChain retriever 接口"""
        return self.search(query)
