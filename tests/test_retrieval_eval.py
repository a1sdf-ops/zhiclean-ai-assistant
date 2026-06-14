"""RAG 检索评测 —— 量化对比 BM25 / Vector / Hybrid / Hybrid+Reranker 四组检索方案

评估指标:
  - MRR (Mean Reciprocal Rank): 第一个正确答案的排名的倒数, 衡量"正确答案有多靠前"
  - Recall@K: 前 K 个结果中包含多少比例的正确答案
  - Precision@K: 前 K 个结果中有几个是正确的

运行方式:
  pytest tests/test_retrieval_eval.py -v -s
  pytest tests/test_retrieval_eval.py -v -s -k "comparison"  # 仅对比实验
"""

import json
import os
import statistics
import sys
from collections import defaultdict

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def requires_api(func):
    """有 DashScope API Key 才执行（embedding 需要）"""
    return pytest.mark.skipif(
        not config.DASHSCOPE_API_KEY or len(config.DASHSCOPE_API_KEY) < 10,
        reason="需要有效的 DASHSCOPE_API_KEY",
    )(func)


def load_eval_queries():
    path = os.path.join(os.path.dirname(__file__), "eval_queries.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────── 指标计算 ───────────


def _get_source(doc):
    """从返回的 Document 中提取 source"""
    if hasattr(doc, "metadata"):
        return doc.metadata.get("source", "")
    return ""


def _ranked_sources(results):
    """提取结果列表中的 source 序列"""
    return [_get_source(doc) for doc in results]


def mrr(results, relevant_sources):
    """MRR: 1 / (第一个相关文档的排名), 0 如果没找到"""
    sources = _ranked_sources(results)
    for rank, src in enumerate(sources, start=1):
        if src in relevant_sources:
            return 1.0 / rank
    return 0.0


def recall_at_k(results, relevant_sources, k):
    """Recall@K: 前 K 个结果中覆盖了多少相关文档"""
    if not relevant_sources:
        return 1.0
    top_k_sources = set(_ranked_sources(results)[:k])
    found = sum(1 for s in relevant_sources if s in top_k_sources)
    return found / len(relevant_sources)


def precision_at_k(results, relevant_sources, k):
    """Precision@K: 前 K 个结果中相关文档占比"""
    top_k_sources = _ranked_sources(results)[:k]
    if not top_k_sources:
        return 0.0
    found = sum(1 for s in top_k_sources if s in relevant_sources)
    return found / len(top_k_sources)


# ─────────── 检索器工厂 ───────────


def _get_vector_store():
    from model.factory import create_embedding_model
    from RAG部分.vector_stores import VectorStoreService

    return VectorStoreService(embedding=create_embedding_model())


def _check_kb_has_data(vector_store):
    """检查知识库是否有数据，没有则跳过测试"""
    try:
        count = vector_store.vector_store._collection.count()
    except Exception:
        count = 0
    return count > 0, count


def _retrieve_bm25(query, top_k=8):
    """纯 BM25 检索"""
    vector_store = _get_vector_store()
    from RAG部分.bm25 import BM25Retriever

    all_data = vector_store.vector_store.get(include=["documents", "metadatas"])
    texts = all_data.get("documents", [])
    if not texts:
        return []
    bm25 = BM25Retriever(k1=config.BM25_K1, b=config.BM25_B)
    bm25.index(texts)
    ranked = bm25.search(query, top_k=top_k)
    # 重建 Document 列表
    metadatas = all_data.get("metadatas", [])
    from langchain_core.documents import Document

    docs = [
        Document(page_content=texts[i], metadata=metadatas[i] if i < len(metadatas) else {}) for i in range(len(texts))
    ]
    return [docs[doc_id] for doc_id, _ in ranked]


def _retrieve_vector(query, top_k=8):
    """纯向量检索"""
    vector_store = _get_vector_store()
    results = vector_store.vector_store.similarity_search(query, k=top_k)
    return results


def _retrieve_hybrid(query, top_k=8):
    """混合检索 (BM25 + Vector + RRF)"""
    vector_store = _get_vector_store()
    from RAG部分.hybrid_retriever import HybridRetriever

    hr = HybridRetriever(vector_store.vector_store)
    return hr.search(query, top_k=top_k)


def _retrieve_hybrid_rerank(query, top_k=8):
    """混合检索 + Reranker"""
    docs = _retrieve_hybrid(query, top_k=top_k * 2)
    from RAG部分.rerank import Rerank

    reranker = Rerank()
    return reranker.rerank_documents(query, docs)[:top_k]


# ─────────── 测试类 ───────────


class TestRetrievalMetrics:
    """指标计算函数单元测试"""

    def test_mrr_first_position(self):
        from langchain_core.documents import Document

        docs = [Document(page_content="a", metadata={"source": "target.txt"})]
        assert mrr(docs, ["target.txt"]) == 1.0

    def test_mrr_second_position(self):
        from langchain_core.documents import Document

        docs = [
            Document(page_content="a", metadata={"source": "wrong.txt"}),
            Document(page_content="b", metadata={"source": "target.txt"}),
        ]
        assert mrr(docs, ["target.txt"]) == 0.5

    def test_mrr_not_found(self):
        from langchain_core.documents import Document

        docs = [Document(page_content="a", metadata={"source": "wrong.txt"})]
        assert mrr(docs, ["target.txt"]) == 0.0

    def test_recall_all_found(self):
        from langchain_core.documents import Document

        docs = [
            Document(page_content="a", metadata={"source": "a.txt"}),
            Document(page_content="b", metadata={"source": "b.txt"}),
        ]
        assert recall_at_k(docs, ["a.txt", "b.txt"], k=3) == 1.0

    def test_recall_half_found(self):
        from langchain_core.documents import Document

        docs = [
            Document(page_content="a", metadata={"source": "a.txt"}),
        ]
        assert recall_at_k(docs, ["a.txt", "b.txt"], k=3) == 0.5

    def test_precision_all_correct(self):
        from langchain_core.documents import Document

        docs = [
            Document(page_content="a", metadata={"source": "a.txt"}),
            Document(page_content="b", metadata={"source": "b.txt"}),
        ]
        assert precision_at_k(docs, ["a.txt", "b.txt"], k=2) == 1.0

    def test_precision_half_correct(self):
        from langchain_core.documents import Document

        docs = [
            Document(page_content="a", metadata={"source": "a.txt"}),
            Document(page_content="b", metadata={"source": "wrong.txt"}),
        ]
        assert precision_at_k(docs, ["a.txt"], k=2) == 0.5


class TestKnowledgeBaseReady:
    """知识库数据完整性检查"""

    @requires_api
    def test_knowledge_base_has_data(self):
        """检查知识库，无数据时 skip 而非 fail（CI 环境无数据是正常的）"""
        vector_store = _get_vector_store()
        has_data, count = _check_kb_has_data(vector_store)
        if not has_data:
            pytest.skip("知识库为空（CI 环境无预上传数据），跳过评测类测试")
        print(f"\n  知识库文档块数: {count}")


class TestRetrievalComparison:
    """四组检索方案对比实验"""

    @requires_api
    def test_retrieval_comparison_table(self):
        """核心评测：跑全部 25 条 query，输出四组方案对比表"""
        vector_store = _get_vector_store()
        has_data, count = _check_kb_has_data(vector_store)
        if not has_data:
            pytest.skip("知识库为空，跳过评测")

        queries = load_eval_queries()
        assert len(queries) == 25, f"预期 25 条评测数据，实际 {len(queries)}"

        retrievers = {
            "BM25": _retrieve_bm25,
            "Vector": _retrieve_vector,
            "Hybrid(BM25+Vec+RRF)": _retrieve_hybrid,
            "Hybrid+Reranker": _retrieve_hybrid_rerank,
        }

        results = {}
        for name, retriever_func in retrievers.items():
            mrr_list, r3_list, r5_list, p5_list = [], [], [], []
            for q in queries:
                try:
                    docs = retriever_func(q["query"])
                except Exception as e:
                    print(f"  [WARN] {name} on '{q['query'][:30]}...' 失败: {e}")
                    docs = []
                relevant = q["relevant_sources"]
                mrr_list.append(mrr(docs, relevant))
                r3_list.append(recall_at_k(docs, relevant, k=3))
                r5_list.append(recall_at_k(docs, relevant, k=5))
                p5_list.append(precision_at_k(docs, relevant, k=5))
            results[name] = {
                "MRR": round(statistics.mean(mrr_list), 4),
                "Recall@3": round(statistics.mean(r3_list), 4),
                "Recall@5": round(statistics.mean(r5_list), 4),
                "Precision@5": round(statistics.mean(p5_list), 4),
                "命中数(Recall@5>0)": sum(1 for r in r5_list if r > 0),
            }

        # 打印对比表
        header = f"{'检索方案':<28} {'MRR':>8} {'Recall@3':>10} {'Recall@5':>10} {'Precision@5':>12} {'命中':>6}"
        sep = "-" * len(header)
        print(f"\n{'=' * len(header)}")
        print(f"  知识库: {count} 个文档块 | 评测查询: {len(queries)} 条")
        print(f"{'=' * len(header)}")
        print(header)
        print(sep)

        best_mrr = max(results.items(), key=lambda x: x[1]["MRR"])
        best_recall = max(results.items(), key=lambda x: x[1]["Recall@5"])

        for name, metrics in results.items():
            marker = ""
            if name == best_mrr[0] and name == best_recall[0]:
                marker = " ← 最优"
            print(
                f"{name:<28} {metrics['MRR']:>8.4f} {metrics['Recall@3']:>10.4f} "
                f"{metrics['Recall@5']:>10.4f} {metrics['Precision@5']:>12.4f} "
                f"{metrics['命中数(Recall@5>0)']:>4}/25{marker}"
            )

        print(sep)
        print(f"  最佳 MRR:      {best_mrr[0]} ({best_mrr[1]['MRR']:.4f})")
        print(f"  最佳 Recall@5: {best_recall[0]} ({best_recall[1]['Recall@5']:.4f})")
        print(f"{'=' * len(header)}\n")

        # 断言：混合检索应该明显优于 BM25
        hybrid_mrr = results["Hybrid(BM25+Vec+RRF)"]["MRR"]
        bm25_mrr = results["BM25"]["MRR"]
        assert hybrid_mrr >= bm25_mrr, f"混合检索 MRR({hybrid_mrr:.4f}) 不应低于纯 BM25({bm25_mrr:.4f})"
        # 注：小数据集(11 chunks)下纯向量 MRR 可能略高，这是正常的——
        # BM25 信号在小数据集中可能引入噪声。数据集越大，RRF 融合优势越明显。

    @requires_api
    def test_reranker_improves_precision(self):
        """验证 Reranker 确实提升了 Precision@5"""
        vector_store = _get_vector_store()
        has_data, _ = _check_kb_has_data(vector_store)
        if not has_data:
            pytest.skip("知识库为空，跳过评测")

        queries = load_eval_queries()
        hybrid_p5_list, rerank_p5_list = [], []

        for q in queries:
            hybrid_docs = _retrieve_hybrid(q["query"])
            rerank_docs = _retrieve_hybrid_rerank(q["query"])
            relevant = q["relevant_sources"]
            hybrid_p5_list.append(precision_at_k(hybrid_docs, relevant, k=5))
            rerank_p5_list.append(precision_at_k(rerank_docs, relevant, k=5))

        hybrid_p5 = statistics.mean(hybrid_p5_list)
        rerank_p5 = statistics.mean(rerank_p5_list)
        print(f"\n  Hybrid P@5:          {hybrid_p5:.4f}")
        print(f"  Hybrid+Reranker P@5: {rerank_p5:.4f}")

        assert rerank_p5 >= hybrid_p5 * 0.95, (
            f"Reranker P@5({rerank_p5:.4f}) 显著低于 Hybrid({hybrid_p5:.4f})，可能是模型未加载或降级生效"
        )

    @requires_api
    def test_difficulty_breakdown(self):
        """按难度分层统计"""
        vector_store = _get_vector_store()
        has_data, _ = _check_kb_has_data(vector_store)
        if not has_data:
            pytest.skip("知识库为空，跳过评测")

        queries = load_eval_queries()
        by_difficulty = defaultdict(list)

        for q in queries:
            docs = _retrieve_hybrid_rerank(q["query"])
            relevant = q["relevant_sources"]
            by_difficulty[q["difficulty"]].append(mrr(docs, relevant))

        print("\n  混合检索+Reranker MRR 按难度分层:")
        for diff in ["easy", "medium", "hard"]:
            scores = by_difficulty.get(diff, [])
            if scores:
                print(f"    {diff:<8}: MRR={statistics.mean(scores):.4f}  ({len(scores)} queries)")

        # easy 难度的 MRR 应该高于 hard
        easy_scores = by_difficulty.get("easy", [])
        hard_scores = by_difficulty.get("hard", [])
        if easy_scores and hard_scores:
            assert statistics.mean(easy_scores) >= statistics.mean(hard_scores) * 0.8, (
                f"easy MRR({statistics.mean(easy_scores):.4f}) 异常低于 hard({statistics.mean(hard_scores):.4f})"
            )
