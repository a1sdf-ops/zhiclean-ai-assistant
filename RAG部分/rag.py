import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections.abc import Iterator

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableWithMessageHistory

import config
from model.factory import create_chat_model, create_embedding_model
from RAG部分.file_history_store import get_history
from RAG部分.hybrid_retriever import HybridRetriever
from RAG部分.rerank import Rerank
from RAG部分.vector_stores import VectorStoreService


class RagService:
    def __init__(self):
        self.vector_service = VectorStoreService(embedding=create_embedding_model())

        self.reranker = Rerank()

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "严格遵循自己提供的资料，并列出所有可能得选择，注意！不要给我没有给的内容，不要瞎编，一切回答按照我给的资料来回答"
                    "简洁和专业的回答用户的问题，回答时给出我提供的参考资料。参考资料：{context}。",
                ),
                ("system", "并且我提供用户的历史对话记录，记录如下："),
                MessagesPlaceholder("history"),
                ("user", "请回答用户的提问：{input}"),
            ]
        )

        self.chat_model = create_chat_model()

        self.chain = self._get_chain()

    def _get_chain(self):
        if config.ENABLE_HYBRID:
            retriever = HybridRetriever(self.vector_service.vector_store)
        else:
            retriever = self.vector_service.get_retriever()

        def format_document(docs: list[Document]):
            if not docs:
                return "无相关参考资料"
            formated_str = ""
            for doc in docs:
                formated_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"
            return formated_str

        def format_rerank_retriever(input_dict):
            query = input_dict["input"]
            docs = retriever.invoke(query)
            docs = self.reranker.rerank_documents(query, docs)
            return format_document(docs)

        def format_for_prompt_template(value):
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        chain = (
            {
                "context": RunnableLambda(format_rerank_retriever),
                "input": RunnablePassthrough(),
            }
            | RunnableLambda(format_for_prompt_template)
            | self.prompt_template
            | self.chat_model
            | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain

    def _get_stream_chain(self):
        """构建流式链（无 StrOutputParser，保留 AIMessageChunk）"""
        if config.ENABLE_HYBRID:
            retriever = HybridRetriever(self.vector_service.vector_store)
        else:
            retriever = self.vector_service.get_retriever()

        def _retrieve_and_rerank(input_dict):
            query = input_dict["input"]
            docs = retriever.invoke(query)
            docs = self.reranker.rerank_documents(query, docs)
            if not docs:
                return "无相关参考资料"
            return "\n\n".join(f"文档片段：{d.page_content}\n文档元数据：{d.metadata}" for d in docs)

        def _reshape(value):
            return {
                "input": value["input"]["input"],
                "context": value["context"],
                "history": value["input"]["history"],
            }

        chain = (
            {"context": RunnableLambda(_retrieve_and_rerank), "input": RunnablePassthrough()}
            | RunnableLambda(_reshape)
            | self.prompt_template
            | self.chat_model
        )

        return RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def ask_stream(self, question: str, session_id: str = "default") -> Iterator[str]:
        """流式问答，逐 token 返回"""
        stream_chain = self._get_stream_chain()
        for chunk in stream_chain.stream(
            {"input": question},
            {"configurable": {"session_id": session_id}},
        ):
            if hasattr(chunk, "content") and chunk.content:
                text = chunk.content
                try:
                    text.encode("utf-8")
                except UnicodeEncodeError:
                    text = text.encode("utf-8", errors="replace").decode("utf-8")
                yield text
