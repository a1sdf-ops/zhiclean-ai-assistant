"""BM25 稀疏检索 —— 手写实现（基于TF-IDF改进）

BM25 核心公式：
  Score(D, Q) = Σ IDF(qi) × (f(qi,D) × (k1 + 1)) / (f(qi,D) + k1 × (1 − b + b × |D|/avgdl))

参数：
  k1: 词频饱和因子（默认1.5），限制单次词频增长的贡献
  b : 文档长度归一化因子（默认0.75），b=1时完全归一化，b=0时忽略长度
"""

import math
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger_handler import logger

# 中文分词（优先用 jieba）
try:
    import jieba

    def _tokenize(text: str) -> list[str]:
        return [t.strip() for t in jieba.cut(text) if len(t.strip()) > 1]
except ImportError:
    _CHINESE_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
    _WORD_RE = re.compile(r"[a-zA-Z0-9]+")

    def _tokenize(text: str) -> list[str]:
        tokens = []
        for m in _CHINESE_RE.finditer(text):
            tokens.append(m.group())
        for m in _WORD_RE.finditer(text):
            tokens.append(m.group().lower())
        return tokens


class BM25Retriever:
    """BM25 检索器 —— 基于倒排索引的稀疏检索"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: list[str] = []
        self.doc_tokens: list[list[str]] = []  # 每篇文档的分词结果
        self.doc_len: list[int] = []  # 每篇文档的长度
        self.avgdl: float = 0.0  # 平均文档长度
        self.idf: dict[str, float] = {}  # 词 → IDF值
        self.inverted_index: dict[str, list[tuple[int, int]]] = defaultdict(list)  # 词 → [(doc_id, tf), ...]
        self._built = False

    def index(self, documents: list[str]) -> None:
        """构建倒排索引"""
        self.documents = documents
        self.doc_tokens = [_tokenize(doc) for doc in documents]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        N = len(documents)

        # 词频统计（每篇文档内）
        doc_term_freqs: list[dict[str, int]] = []
        for tokens in self.doc_tokens:
            tf_map: dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1
            doc_term_freqs.append(tf_map)

        # 倒排索引 + IDF 计算
        self.inverted_index.clear()
        df: dict[str, int] = defaultdict(int)  # 文档频率

        for doc_id, tf_map in enumerate(doc_term_freqs):
            for term, freq in tf_map.items():
                df[term] += 1
                self.inverted_index[term].append((doc_id, freq))

        # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
        self.idf = {}
        for term, doc_freq in df.items():
            self.idf[term] = math.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

        self._built = True
        logger.info("BM25索引构建完成: %d 篇文档, %d 个词条", N, len(self.inverted_index))

    def search(self, query: str, top_k: int = 8) -> list[tuple[int, float]]:
        """检索，返回 [(doc_id, score), ...] 按分数降序"""
        if not self._built or not self.documents:
            return []

        query_tokens = _tokenize(query)
        scores: dict[int, float] = defaultdict(float)

        for token in query_tokens:
            idf = self.idf.get(token, 0.0)
            if idf == 0.0:
                continue
            for doc_id, tf in self.inverted_index.get(token, []):
                doc_len = self.doc_len[doc_id]
                # BM25 核心公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                scores[doc_id] += idf * numerator / denominator

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
