"""
Streamlit Web UI for the LangChain RAG System.

Run with: streamlit run src/app.py
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DOCUMENTS_DIR, CHROMA_PERSIST_DIR
from src.document_loader import load_and_split
from src.vectorstore import get_or_create_vectorstore, create_vectorstore
from src.rag_chain import create_rag_chain_with_sources


# Page configuration
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #3B82F6;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E0F2FE;
    }
    .assistant-message {
        background-color: #F0FDF4;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "rag_func" not in st.session_state:
        st.session_state.rag_func = None
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []


def load_vectorstore():
    """Load or create the vector store."""
    with st.spinner("Loading documents and creating index..."):
        try:
            chunks = load_and_split()
            if not chunks:
                st.error("No documents found. Please add documents to the data/documents folder.")
                return False
            
            vectorstore = get_or_create_vectorstore(chunks)
            st.session_state.vectorstore = vectorstore
            st.session_state.rag_func = create_rag_chain_with_sources(vectorstore)
            return True
        except Exception as e:
            st.error(f"Error loading documents: {e}")
            return False


def save_uploaded_file(uploaded_file):
    """Save an uploaded file to the documents directory."""
    docs_path = Path(DOCUMENTS_DIR)
    docs_path.mkdir(parents=True, exist_ok=True)
    
    file_path = docs_path / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


def main():
    """Main application."""
    initialize_session_state()
    
    # Header
    st.markdown('<p class="main-header">📚 RAG Document Q&A</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about your documents using AI</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Document upload
        st.subheader("📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Add PDF, TXT, or MD files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = save_uploaded_file(uploaded_file)
                st.success(f"Saved: {uploaded_file.name}")
        
        st.divider()
        
        # Index management
        st.subheader("🔄 Index Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Load Index", use_container_width=True):
                if load_vectorstore():
                    st.success("Index loaded!")
        
        with col2:
            if st.button("Rebuild Index", use_container_width=True):
                # Clear existing index
                import shutil
                chroma_path = Path(CHROMA_PERSIST_DIR)
                if chroma_path.exists():
                    shutil.rmtree(chroma_path)
                
                if load_vectorstore():
                    st.success("Index rebuilt!")
        
        # Show index status
        if st.session_state.vectorstore is not None:
            count = st.session_state.vectorstore._collection.count()
            st.info(f"📊 Index contains {count} chunks")
        else:
            st.warning("⚠️ No index loaded. Click 'Load Index' to start.")
        
        st.divider()
        
        # Document list
        st.subheader("📄 Current Documents")
        docs_path = Path(DOCUMENTS_DIR)
        if docs_path.exists():
            files = list(docs_path.glob("**/*.*"))
            if files:
                for f in files:
                    st.text(f"• {f.name}")
            else:
                st.text("No documents yet")
        
        st.divider()
        
        # Clear chat
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.rerun()
    
    # Main chat interface
    if st.session_state.vectorstore is None:
        st.info("👈 Click 'Load Index' in the sidebar to start asking questions.")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📄 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.markdown(f"> {source[:300]}...")
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.rag_func(prompt)
                    answer = result["answer"]
                    sources = [doc.page_content for doc in result["sources"]]
                    
                    st.markdown(answer)
                    
                    # Show sources
                    with st.expander("📄 View Sources"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Source {i}:**")
                            st.markdown(f"> {source[:300]}...")
                            st.divider()
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    st.session_state.last_sources = sources
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")


if __name__ == "__main__":
    main()
