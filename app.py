"""Streamlit web UI for LocalMind Agent."""

from pathlib import Path

import streamlit as st

from src.agent import run_agent, synthesize_answer
from src.config import RAW_DOCS_PATH
from src.loader import build_chunks
from src.vectorstore import collection_size, index_documents

st.set_page_config(page_title="LocalMind Agent", page_icon="🧠", layout="wide")
st.title("🧠 LocalMind Agent")
st.caption("Local LLM + RAG research assistant")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("📚 Document Index")
    st.metric("Chunks stored", collection_size())

    uploaded_files = st.file_uploader(
        "Add documents",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("📥 Add and index", use_container_width=True):
        with st.spinner("Saving, splitting, embedding..."):
            Path(RAW_DOCS_PATH).mkdir(parents=True, exist_ok=True)
            for uploaded_file in uploaded_files:
                destination = Path(RAW_DOCS_PATH) / uploaded_file.name
                destination.write_bytes(uploaded_file.getvalue())
            chunks = build_chunks(RAW_DOCS_PATH)
            n = index_documents(chunks, reset=True)
        st.success(f"Indexed {n} chunks from {len(uploaded_files)} file(s).")
        st.rerun()

    if st.button("🔍 Re-index documents", use_container_width=True):
        with st.spinner("Loading, splitting, embedding..."):
            chunks = build_chunks(RAW_DOCS_PATH)
            n = index_documents(chunks, reset=True)
        if n:
            st.success(f"Indexed {n} chunks.")
        else:
            st.warning("No PDF, TXT, or Markdown documents found to index.")
        st.rerun()

    st.divider()
    mode = st.radio("Mode", ["agent", "rag"], index=0, horizontal=True)
    st.caption(
        "**agent**: full ReAct loop with tool use\n\n"
        "**rag**: direct retrieval + grounded synthesis"
    )

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- Chat state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Input ----------
if prompt := st.chat_input("Ask something about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if collection_size() == 0:
                answer = (
                    "⚠️ The vector store is empty. Upload a PDF, TXT, or Markdown "
                    "document in the sidebar, then index it."
                )
            elif mode == "rag":
                answer = synthesize_answer(prompt)
            else:
                answer = run_agent(prompt)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
