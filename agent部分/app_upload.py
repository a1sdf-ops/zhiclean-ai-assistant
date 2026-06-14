"""知识库文档管理 — Streamlit 文件上传/查看/删除

启动:
  streamlit run app_upload.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from RAG部分.knowledge_base import KnowledgeBaseService

st.set_page_config(page_title="智净知识库管理", page_icon="📚")
st.title("📚 智净知识库管理")
st.caption("上传产品文档到知识库（docs/ 目录下已有 5 篇）")

# 初始化服务
if "kb_service" not in st.session_state:
    st.session_state["kb_service"] = KnowledgeBaseService()

kb = st.session_state["kb_service"]

# ── 两个标签页 ──
tab_upload, tab_list = st.tabs(["📤 上传文档", "📋 已存文档"])

# ── 上传标签 ──
with tab_upload:
    upload_method = st.radio("上传方式", ["上传 TXT 文件", "直接输入文本"], horizontal=True)

    if upload_method == "上传 TXT 文件":
        uploaded_file = st.file_uploader("选择 TXT 文件", type=["txt"], accept_multiple_files=False)
        if uploaded_file is not None:
            file_name = uploaded_file.name
            file_size = uploaded_file.size / 1024
            st.write(f"文件名: **{file_name}** | 大小: {file_size:.1f} KB")

            try:
                text = uploaded_file.getvalue().decode("utf-8")
            except UnicodeDecodeError:
                text = uploaded_file.getvalue().decode("gbk")

            with st.spinner("正在将文档载入知识库..."):
                time.sleep(0.5)
                result = kb.upload_by_str(text, file_name)
            if "成功" in result:
                st.success(result)
            elif "跳过" in result:
                st.info(result)
            else:
                st.error(result)
    else:
        doc_name = st.text_input("文档名称", placeholder="例如：朱自清《春》")
        doc_content = st.text_area("文档内容", placeholder="粘贴文本内容...", height=250)
        if st.button("提交到知识库", type="primary"):
            if not doc_name.strip():
                st.warning("请输入文档名称")
            elif not doc_content.strip():
                st.warning("请输入文档内容")
            else:
                with st.spinner("载入中..."):
                    result = kb.upload_by_str(doc_content.strip(), doc_name.strip())
                if "成功" in result:
                    st.success(result)
                elif "跳过" in result:
                    st.info(result)
                else:
                    st.error(result)

# ── 已存文档标签 ──
with tab_list:
    page_size = st.selectbox("每页条数", [5, 10, 20], index=1)
    page = st.number_input("页码", min_value=1, value=1)

    if st.button("刷新列表"):
        st.rerun()

    data = kb.list_knowledge(page=page, page_size=page_size)

    if "error" in data:
        st.error(data["error"])
    elif not data.get("data"):
        st.info("知识库为空，请先上传文档")
    else:
        st.write(f"共 **{data['total']}** 篇文档，第 {data['page']}/{data['total_pages']} 页")

        for item in data["data"]:
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.write(f"📄 {item['source']}")
            with col2:
                st.write(f"📅 {item.get('create_time', '')}")
            with col3:
                if st.button("🗑 删除", key=f"del_{item['source']}"):
                    result = kb.delete_knowledge(item["source"])
                    st.toast(result)
                    time.sleep(0.5)
                    st.rerun()
