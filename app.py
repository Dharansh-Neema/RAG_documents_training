import streamlit as st
import sys
from pathlib import Path
import os

# Ensure src is in the path
sys.path.append(str(Path(__file__).parent / "src"))
from src.generator import generate_response, llm

st.set_page_config(page_title="RAG Chatbot", layout="wide")

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "history" not in st.session_state:
    st.session_state["history"] = []

# --- Sidebar ---
st.sidebar.title("RAG System Info")
model_name = getattr(llm, "model", "Unknown")
st.sidebar.markdown(f"**Model in use:** {model_name}")

# Count number of chunks or indexed documents
def get_chunk_count():
    chunks_dir = Path("chunks")
    if chunks_dir.exists() and chunks_dir.is_dir():
        return len(list(chunks_dir.glob("*.txt")))
    return 0
chunk_count = get_chunk_count()
st.sidebar.markdown(f"**Number of chunks:** {chunk_count}")

if st.sidebar.button("Clear chat / Reset"):
    st.session_state["messages"] = []
    st.session_state["history"] = []
    st.experimental_rerun()

st.title("📄🔎 RAG Chatbot")
st.markdown("Ask a question about your indexed documents.")

# --- Chat UI ---
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Show source passages"):
                for i, doc in enumerate(msg["sources"], 1):
                    st.markdown(f"**Source {i}:**\n{doc}")

user_query = st.chat_input("Type your question and press Enter...")

if user_query:
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Placeholder for streaming response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        sources_placeholder = st.empty()
        # Call backend
        response_data = generate_response(user_query)
        if isinstance(response_data, tuple):
            response, retrieved_docs = response_data
        else:
            response, retrieved_docs = response_data, {"documents": [[]]}
        # Simulate streaming by displaying response in chunks
        import time
        words = response.split()
        streamed = ""
        for w in words:
            streamed += w + " "
            response_placeholder.markdown(streamed)
            time.sleep(0.015)  # Simulate streaming
        # Show sources
        docs = retrieved_docs.get("documents", [[]])[0]
        if docs:
            with sources_placeholder.expander("Show source passages"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**Source {i}:**\n{doc}")
        # Save assistant message
        st.session_state["messages"].append({
            "role": "assistant",
            "content": response,
            "sources": docs if docs else None
        }) 