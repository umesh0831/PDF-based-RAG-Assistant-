import os
import sys
import tempfile

import chromadb
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.chunker import chunk_text
from rag.llm import generate_answer
from rag.parser import parse_pdf
from rag.reranker import rerank
from rag.store import add_chunks, get_fresh_store, retrieve

st.set_page_config(
    page_title="PDF Q&A Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif !important; }

/* Hide Streamlit chrome */
#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* App background */
.stApp { background: #0d1117; color: #e6edf3; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #21262d !important;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
[data-testid="stSidebarContent"] { padding: 1.5rem 1rem; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #21262d;
    border: 2px dashed #30363d;
    border-radius: 12px;
    padding: 0.5rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #58a6ff; }
[data-testid="stFileUploaderDropzone"] { background: transparent !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
    transition: opacity 0.2s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Chat messages */
[data-testid="stChatMessage"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    color: #e6edf3 !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.15) !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: #8b949e !important; font-size: 0.82rem !important; }

/* Success / info / error boxes */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* Divider */
hr { border-color: #21262d !important; }

/* Spinner */
[data-testid="stSpinner"] { color: #58a6ff !important; }

/* Sidebar doc card */
.doc-card {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
}
.doc-card .doc-name { font-weight: 600; color: #58a6ff; font-size: 0.9rem; word-break: break-all; }
.doc-card .doc-meta { color: #8b949e; font-size: 0.78rem; margin-top: 0.2rem; }

/* Model badge */
.model-badge {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.75rem;
    color: #8b949e;
    margin: 0.15rem;
}

/* Welcome card */
.welcome-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 3rem 2.5rem;
    text-align: center;
    max-width: 500px;
    margin: 4rem auto;
}
.welcome-card h2 { color: #e6edf3; font-size: 1.5rem; margin-bottom: 0.5rem; }
.welcome-card p { color: #8b949e; font-size: 0.95rem; line-height: 1.6; }
.step { display: flex; align-items: flex-start; gap: 0.8rem; margin: 0.8rem 0; text-align: left; }
.step-num {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 50%;
    width: 26px; height: 26px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 700; flex-shrink: 0;
}
.step-text { color: #8b949e; font-size: 0.88rem; padding-top: 3px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_chroma_client():
    return chromadb.Client()


client = get_chroma_client()

for key, default in [
    ("store", None),
    ("pdf_name", None),
    ("chunk_count", 0),
    ("messages", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding-bottom: 1rem;'>
        <div style='font-size:2.2rem;'>📄</div>
        <div style='font-size:1.1rem; font-weight:700; color:#e6edf3;'>PDF Q&A Assistant</div>
        <div style='font-size:0.78rem; color:#8b949e; margin-top:0.2rem;'>
            Fully local · Open-source
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='color:#8b949e; font-size:0.8rem; font-weight:600; margin-bottom:0.5rem;'>UPLOAD DOCUMENT</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file and uploaded_file.name != st.session_state.pdf_name:
        with st.spinner("Indexing document..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                text = parse_pdf(tmp_path)
                os.unlink(tmp_path)

                if not text.strip():
                    st.error("No text found. Scanned/image PDFs are not supported.")
                else:
                    chunks = chunk_text(text, source=uploaded_file.name)
                    store = get_fresh_store(client)
                    add_chunks(store, chunks)

                    st.session_state.store = store
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.chunk_count = len(chunks)
                    st.session_state.messages = []

            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.pdf_name:
        st.markdown(f"""
        <div class='doc-card'>
            <div class='doc-name'>📎 {st.session_state.pdf_name}</div>
            <div class='doc-meta'>{st.session_state.chunk_count} chunks indexed · Ready</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Clear & Upload New"):
            st.session_state.store = None
            st.session_state.pdf_name = None
            st.session_state.chunk_count = 0
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.markdown("<div style='color:#8b949e; font-size:0.8rem; font-weight:600; margin-bottom:0.6rem;'>MODELS</div>", unsafe_allow_html=True)
    st.markdown("""
    <div>
        <span class='model-badge'>🔍 qwen3-embedding:0.6b</span>
        <span class='model-badge'>🧠 gpt-oss:20b</span>
        <span class='model-badge'>🗄️ ChromaDB</span>
        <span class='model-badge'>⚙️ Ollama</span>
    </div>
    """, unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────
if not st.session_state.store:
    st.markdown("""
    <div class='welcome-card'>
        <h2>Ask anything about your PDF</h2>
        <p>Upload any PDF and ask questions in plain English. Fully local, fully open-source — your documents never leave your machine.</p>
        <br/>
        <div class='step'>
            <div class='step-num'>1</div>
            <div class='step-text'>Make sure Ollama is running with qwen3-embedding:0.6b and gpt-oss:20b pulled</div>
        </div>
        <div class='step'>
            <div class='step-num'>2</div>
            <div class='step-text'>Upload a PDF using the panel on the left</div>
        </div>
        <div class='step'>
            <div class='step-num'>3</div>
            <div class='step-text'>Type your question and get a grounded answer</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("context"):
                with st.expander("View source context"):
                    for i, chunk in enumerate(msg["context"], 1):
                        st.markdown(f"**Chunk {i}**")
                        st.caption(chunk[:600] + ("..." if len(chunk) > 600 else ""))

    # Chat input
    if question := st.chat_input("Ask a question about your document..."):
        # Show user message
        with st.chat_message("user", avatar="🧑"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # Generate answer
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching document and generating answer..."):
                try:
                    context = rerank(retrieve(st.session_state.store, question))
                    result = generate_answer(context)
                    answer = result.answer
                    context_chunks = result.context_chunks
                except Exception as e:
                    answer = f"Error: {e}"
                    context_chunks = []

            st.markdown(answer)
            if context_chunks:
                with st.expander("View source context"):
                    for i, chunk in enumerate(context_chunks, 1):
                        st.markdown(f"**Chunk {i}**")
                        st.caption(chunk[:600] + ("..." if len(chunk) > 600 else ""))

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "context": context_chunks,
        })
