"""知识库智能问答 — Streamlit 对话界面（Agent + MCP + RAG 完整链路）

启动:
  streamlit run app_qa.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from agent部分.react_agent import ReactAgent

st.set_page_config(page_title="智净AI售后助手", page_icon="🤖")
st.title("🤖 智净 AI 售后助手")
st.caption("Agent → 意图分类 → 工具调度 → MCP/RAG 知识库 → 流式输出")

st.divider()

if "agent" not in st.session_state:
    with st.spinner("正在初始化 Agent（加载 StateGraph + 模型）..."):
        st.session_state["agent"] = ReactAgent()

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": '你好！我是智净 AI 售后助手，可以帮你：\n\n📚 **产品咨询** — "Z3 Ultra和Z2 Pro有什么区别？"\n🔧 **故障排查** — "机器人回不了充电桩怎么办？"\n🛠 **保养指导** — "滤网多久换一次，怎么清洗？"\n🌤 **环境建议** — "最近梅雨天，应该怎么设置？"\n📊 **使用报告** — "帮我生成这个月的使用报告"',
        }
    ]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

prompt = st.chat_input("输入你的问题...")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.spinner("Agent 思考中..."):
        stream = st.session_state["agent"].execute_stream(prompt)
        response = st.chat_message("assistant").write_stream(stream)

    if response:
        st.session_state["messages"].append({"role": "assistant", "content": response})
    else:
        st.session_state["messages"].append({"role": "assistant", "content": "抱歉，检索未返回结果。"})
