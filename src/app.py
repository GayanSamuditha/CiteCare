"""
CiteCare - Clinical Evidence Q&A Assistant

Features:
- Upload PDFs or paste abstracts
- RAG-powered evidence retrieval
- Confidence scoring for answers
- Page-level citations
- Markdown export
- Multiple collections
"""

import streamlit as st
from pathlib import Path
import sys
import shutil
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DOCUMENTS_DIR, COLLECTIONS_DIR
from src.document_loader import load_and_split
from src.vectorstore import (
    create_collection, load_collection, delete_collection,
    list_collections, get_collection_stats
)
from src.rag_chain import create_rag_chain_with_sources, generate_markdown_summary


# Page config
st.set_page_config(
    page_title="CiteCare - Clinical Evidence Q&A",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    
    /* Header */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0EA5E9 0%, #8B5CF6 50%, #D946EF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.15rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    
    /* Chat bubbles */
    .user-bubble {
        background: linear-gradient(135deg, #0EA5E9 0%, #6366F1 100%);
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 1.25rem 1.25rem 0.25rem 1.25rem;
        margin: 0.75rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.25);
    }
    
    .assistant-bubble {
        background: #F8FAFC;
        color: #1E293B;
        padding: 1.25rem;
        border-radius: 1.25rem 1.25rem 1.25rem 0.25rem;
        margin: 0.75rem 0;
        max-width: 90%;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Confidence badge */
    .confidence-high {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.75rem 0;
    }
    
    .confidence-moderate {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.75rem 0;
    }
    
    .confidence-low {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.75rem 0;
    }
    
    /* Source cards */
    .source-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 0.75rem;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.2s ease;
    }
    
    .source-card:hover {
        border-color: #0EA5E9;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.15);
    }
    
    .source-file {
        font-weight: 600;
        color: #1E293B;
    }
    
    .source-page {
        background: #0EA5E9;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 1rem;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Sidebar */
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1E293B;
    }
    
    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border: 1px solid #E2E8F0;
    }
    
    /* Example questions */
    .example-btn {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .example-btn:hover {
        border-color: #0EA5E9;
        background: #F0F9FF;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state."""
    defaults = {
        "messages": [],
        "current_collection": None,
        "rag_func": None,
        "last_result": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


from src.utils.validation import validate_collection_name


def save_uploaded_files(uploaded_files, collection_name: str) -> Path:
    """Save uploaded files."""
    collection_path = Path(COLLECTIONS_DIR) / collection_name
    collection_path.mkdir(parents=True, exist_ok=True)
    
    for file in uploaded_files:
        file_path = collection_path / file.name
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    
    return collection_path


def save_abstract_as_file(abstract: str, collection_name: str, title: str = "pasted_abstract") -> Path:
    """Save pasted abstract as a text file."""
    collection_path = Path(COLLECTIONS_DIR) / collection_name
    collection_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{title}_{timestamp}.txt"
    file_path = collection_path / file_name
    
    with open(file_path, "w") as f:
        f.write(abstract)
    
    return collection_path


def get_confidence_class(confidence_text: str) -> str:
    """Determine CSS class based on confidence level."""
    confidence_lower = confidence_text.lower()
    if "high" in confidence_lower:
        return "confidence-high"
    elif "moderate" in confidence_lower or "medium" in confidence_lower:
        return "confidence-moderate"
    else:
        return "confidence-low"


def render_sources(sources: list):
    """Render source citations."""
    for i, src in enumerate(sources, 1):
        file_name = src.get("file_name", "Unknown")
        page = src.get("page", "N/A")
        content = src.get("content", "")[:350]
        
        st.markdown(f"""
        <div class="source-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span class="source-file">📄 {file_name}</span>
                <span class="source-page">Page {page}</span>
            </div>
            <div style="color: #64748B; font-size: 0.9rem; line-height: 1.6;">
                {content}...
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar."""
    with st.sidebar:
        st.markdown('<p class="sidebar-title">🔬 CiteCare</p>', unsafe_allow_html=True)
        st.caption("Clinical Evidence Q&A Assistant")
        st.divider()
        
        # Create collection section
        st.markdown("### 📁 Collections")
        
        collections = list_collections()
        
        with st.expander("➕ Add New Collection", expanded=not collections):
            tab1, tab2 = st.tabs(["📄 Upload PDFs", "📝 Paste Abstract"])
            
            with tab1:
                new_name = st.text_input("Collection name", placeholder="e.g., alzheimers-2024", key="pdf_name")
                uploaded = st.file_uploader(
                    "Upload papers/guidelines",
                    type=["pdf", "txt", "md"],
                    accept_multiple_files=True,
                    key="pdf_upload"
                )
                
                if st.button("Create from PDFs", type="primary", use_container_width=True):
                    if new_name and uploaded:
                        valid, error_msg = validate_collection_name(new_name)
                        if not valid:
                            st.error(f"Invalid name: {error_msg}")
                        else:
                            too_big = [f for f in uploaded if f.size and f.size > 50 * 1024 * 1024]
                            if too_big:
                                st.error("One or more files exceed 50MB. Please upload smaller files.")
                            else:
                                with st.spinner("📚 Processing documents..."):
                                    doc_path = save_uploaded_files(uploaded, new_name)
                                    chunks = load_and_split(str(doc_path), collection_name=new_name)
                                    
                                    if chunks:
                                        create_collection(new_name, chunks)
                                        st.success(f"✓ Created '{new_name}'")
                                        st.rerun()
                                    else:
                                        st.error("Could not extract content")
                    else:
                        st.warning("Enter name and upload files")
            
            with tab2:
                abstract_name = st.text_input("Collection name", placeholder="e.g., study-abstract", key="abstract_name")
                abstract_text = st.text_area(
                    "Paste abstract or text",
                    height=200,
                    placeholder="Paste your abstract, methods section, or any research text here..."
                )
                
                if st.button("Create from Text", type="primary", use_container_width=True):
                    if abstract_name and abstract_text:
                        valid, error_msg = validate_collection_name(abstract_name)
                        if not valid:
                            st.error(f"Invalid name: {error_msg}")
                        else:
                            with st.spinner("📝 Processing text..."):
                                doc_path = save_abstract_as_file(abstract_text, abstract_name)
                                chunks = load_and_split(str(doc_path), collection_name=abstract_name)
                                
                                if chunks:
                                    create_collection(abstract_name, chunks)
                                    st.success(f"✓ Created '{abstract_name}'")
                                    st.rerun()
                    else:
                        st.warning("Enter name and paste text")
        
        st.divider()
        
        # Collection list
        if collections:
            for coll in collections:
                stats = get_collection_stats(coll)
                is_active = st.session_state.current_collection == coll
                
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    btn_label = f"{'✅' if is_active else '📁'} {coll}"
                    if st.button(btn_label, key=f"sel_{coll}", use_container_width=True,
                                type="primary" if is_active else "secondary"):
                        if not is_active:
                            st.session_state.current_collection = coll
                            vectorstore = load_collection(coll)
                            if vectorstore:
                                st.session_state.rag_func = create_rag_chain_with_sources(vectorstore)
                                st.session_state.messages = []
                            st.rerun()
                
                with col2:
                    if st.button("🗑️", key=f"del_{coll}"):
                        delete_collection(coll)
                        coll_path = Path(COLLECTIONS_DIR) / coll
                        if coll_path.exists():
                            shutil.rmtree(coll_path)
                        if st.session_state.current_collection == coll:
                            st.session_state.current_collection = None
                            st.session_state.rag_func = None
                        st.rerun()
                
                if is_active and stats:
                    st.caption(f"  📊 {stats.get('chunks', 0)} chunks • {stats.get('file_count', 0)} files")
        else:
            st.info("Create a collection to get started")
        
        st.divider()
        
        # Export button
        if st.session_state.last_result:
            if st.button("📥 Export Last Answer to Markdown", use_container_width=True):
                result = st.session_state.last_result
                md_content = generate_markdown_summary(
                    question=result.get("question", ""),
                    answer=result.get("answer", ""),
                    sources=result.get("sources", []),
                    confidence=result.get("confidence", ""),
                    collection_name=st.session_state.current_collection or ""
                )
                st.download_button(
                    label="⬇️ Download Markdown",
                    data=md_content,
                    file_name=f"evidence_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
        
        # Clear chat
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.last_result = None
                st.rerun()


def render_main():
    """Render main content area."""
    # Hero header
    st.markdown('<p class="hero-title">CiteCare</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Clinical Paper → Evidence Q&A Assistant</p>', unsafe_allow_html=True)
    
    if not st.session_state.current_collection:
        # Welcome state
        st.markdown("""
        <div class="feature-card">
            <h4>🚀 Get Started</h4>
            <p>Upload clinical papers, guidelines, or paste abstracts to create a collection. Then ask evidence-based questions!</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **📄 Upload PDFs**  
            Papers, guidelines, reviews
            """)
        with col2:
            st.markdown("""
            **🔍 Ask Questions**  
            Outcomes, cohorts, limitations
            """)
        with col3:
            st.markdown("""
            **📊 Get Evidence**  
            Citations + confidence scores
            """)
        
        st.markdown("### 💡 Example Questions")
        st.markdown("""
        - "What are the main outcomes of this study?"
        - "What patient cohorts were included?"
        - "What are the study limitations?"
        - "What statistical methods were used?"
        - "What are the key findings?"
        """)
        return
    
    # Active collection header
    stats = get_collection_stats(st.session_state.current_collection)
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown(f"**📁 Collection:** `{st.session_state.current_collection}`")
    with col2:
        st.caption(f"📄 {stats.get('file_count', 0)} files")
    with col3:
        st.caption(f"📊 {stats.get('chunks', 0)} chunks")
    
    st.divider()
    
    # Chat messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
            
            # Confidence badge
            if "confidence" in msg:
                conf_class = get_confidence_class(msg["confidence"])
                st.markdown(f'<div class="{conf_class}">{msg["confidence"]}</div>', unsafe_allow_html=True)
            
            # Sources
            if "sources" in msg and msg["sources"]:
                with st.expander(f"📚 View {len(msg['sources'])} Sources", expanded=False):
                    render_sources(msg["sources"])
    
    # Example questions (only show if no messages)
    if not st.session_state.messages:
        st.markdown("**💡 Try asking:**")
        example_cols = st.columns(3)
        examples = [
            "What are the main outcomes?",
            "What cohorts were studied?",
            "What are the limitations?"
        ]
        for i, ex in enumerate(examples):
            with example_cols[i]:
                if st.button(ex, key=f"ex_{i}", use_container_width=True):
                    st.session_state.pending_question = ex
                    st.rerun()
    
    # Chat input
    prompt = st.chat_input("Ask about outcomes, methods, cohorts, limitations...")
    
    # Check for pending question from example buttons
    if hasattr(st.session_state, 'pending_question'):
        prompt = st.session_state.pending_question
        del st.session_state.pending_question
    
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        if st.session_state.rag_func:
            with st.spinner("🔍 Analyzing evidence..."):
                try:
                    result = st.session_state.rag_func(prompt)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                        "confidence": result["confidence"]
                    })
                    
                    st.session_state.last_result = result
                    
                except Exception as e:
                    st.error(f"Error: {e}")
        
        st.rerun()


def main():
    """Main application."""
    init_session_state()
    
    Path(DOCUMENTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(COLLECTIONS_DIR).mkdir(parents=True, exist_ok=True)
    
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
