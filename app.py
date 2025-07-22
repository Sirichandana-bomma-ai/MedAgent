import os
import streamlit as st
from rag_assistant import RAGAssistant
from database import VectorDB
from memory_agent import MemoryAgent

# Load keys from Streamlit Secrets into environment variables
if "TOGETHER_API_KEY" in st.secrets:
    os.environ["TOGETHER_API_KEY"] = st.secrets["TOGETHER_API_KEY"]

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]



st.set_page_config(page_title="🩺 Medical RAG Assistant", layout="wide")
st.title("🩺 Medical RAG Assistant (with Context & Citations)")

if 'memory' not in st.session_state:
    st.session_state.memory = MemoryAgent()
    st.session_state.retriever = VectorDB()
    st.session_state.assistant = RAGAssistant(st.session_state.retriever)

query = st.text_input("Ask a medical question:")

if query:
    with st.spinner("Generating answer..."):
        answer = st.session_state.assistant.generate(query)
        st.session_state.memory.add_interaction(query, answer)

    st.subheader("✅ Answer:")
    st.write(answer)

    with st.expander("📜 Conversation History"):
        for h in st.session_state.memory.history:
            st.write(f"**Q:** {h['question']}")
            st.write(f"**A:** {h['answer']}")
