import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hashlib
from datetime import datetime

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from model.factory import create_embedding_model


def check_md5(md5_str: str):
    if not os.path.exists(config.MD5_PATH):
        with open(config.MD5_PATH, "w", encoding="utf-8") as f:
            pass
        return False
    else:
        with open(config.MD5_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip() == md5_str:
                    return True
        return False


def save_md5(md5_str: str):
    with open(config.MD5_PATH, "a", encoding="utf-8") as f:
        f.write(md5_str + "\n")


def get_string_md5(input_str: str, encoding="utf-8"):
    """将传入的字符串转变成md5字符串，避免重复处理同一段文本"""
    str_bytes = input_str.encode(encoding=encoding)
    md5_obj = hashlib.md5()
    md5_obj.update(str_bytes)
    md5_hex = md5_obj.hexdigest()
    return md5_hex


class KnowledgeBaseService:
    def __init__(self):
        os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)

        self.chroma = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=create_embedding_model(),
            persist_directory=config.CHROMA_PERSIST_DIR,
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=config.SEPARATORS,
            length_function=len,
        )

    def upload_by_str(self, data: str, filename: str):
        """将传入字符串向量化并存入知识库"""
        md5_hex = get_string_md5(data)

        if check_md5(md5_hex):
            return "[跳过]内容已存在知识库中"

        if len(data) > config.MAX_SPLIT_CHAR_NUMBER:
            knowledge_chunks: list[str] = self.spliter.split_text(data)
        else:
            knowledge_chunks = [data]

        metadata = {
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": config.OPERATOR_NAME,
        }

        self.chroma.add_texts(knowledge_chunks, metadatas=[metadata for _ in knowledge_chunks])

        save_md5(md5_hex)
        return "[成功]内容已经成功载入向量库"

    def upload_by_file(self, file_path: str):
        """读取本地 txt 文件，自动写入知识库"""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, encoding="gbk") as f:
                content = f.read()

        filename = os.path.basename(file_path)
        return self.upload_by_str(content, filename)

    def list_knowledge(self, page: int = 1, page_size: int = 10):
        """分页列出知识库中的所有文档（按 source 去重）"""
        try:
            all_data = self.chroma.get(include=["metadatas"])
        except Exception as e:
            return {"error": f"读取知识库失败: {e}", "data": [], "total": 0}

        metadatas = all_data.get("metadatas", [])
        if not metadatas:
            return {"data": [], "total": 0, "page": page, "page_size": page_size}

        seen = set()
        unique_sources = []
        for meta in metadatas:
            source = meta.get("source", "unknown")
            if source not in seen:
                seen.add(source)
                unique_sources.append(
                    {
                        "source": source,
                        "create_time": meta.get("create_time", ""),
                        "operator": meta.get("operator", ""),
                    }
                )

        total = len(unique_sources)
        start = (page - 1) * page_size
        end = start + page_size

        return {
            "data": unique_sources[start:end],
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
        }

    def delete_knowledge(self, source_name: str):
        """按 source 名称删除知识库中的文档"""
        try:
            self.chroma.delete(where={"source": source_name})
            return f"[成功] 已删除知识: {source_name}"
        except Exception as e:
            return f"[失败] 删除失败: {e}"

    def update_knowledge(self, source_name: str, new_content: str):
        """更新知识库中的文档：先删旧数据，再上传新内容"""
        delete_result = self.delete_knowledge(source_name)
        upload_result = self.upload_by_str(new_content, source_name)
        return f"{delete_result}; {upload_result}"

    def update_knowledge_by_file(self, source_name: str, file_path: str):
        """更新知识库中的文档：先删旧数据，再从文件上传新内容"""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, encoding="gbk") as f:
                content = f.read()
        return self.update_knowledge(source_name, content)
