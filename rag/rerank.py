import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from sentence_transformers import CrossEncoder

import config

logger = logging.getLogger(__name__)


class Rerank:
    def __init__(self):
        self._model = None
        self._load_attempted = False

    @property
    def model(self):
        if not self._load_attempted:
            self._load_attempted = True
            self._load_model()
        return self._model

    def _load_model(self):
        try:
            model_path = os.path.expanduser("~/.cache/modelscope/hub/models/BAAI/bge-reranker-v2-m3")
            if not os.path.exists(model_path):
                model_path = "BAAI/bge-reranker-v2-m3"

            self._model = CrossEncoder(model_path, device="cpu")
            logger.info("Rerank 模型加载成功")
        except Exception as e:
            logger.warning("Rerank 模型加载失败，已降级: %s", e)

    def rerank_documents(self, query: str, documents: list):
        if not config.ENABLE_RERANK:
            return documents[: config.RERANK_TOP_K]

        if self._model is None:
            return documents[: config.RERANK_TOP_K]

        if len(documents) < 2:
            return documents

        pairs = [[query, doc.page_content] for doc in documents]

        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            logger.warning("Rerank 预测失败: %s", e)
            return documents[: config.RERANK_TOP_K]

        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in scored[: config.RERANK_TOP_K]]
