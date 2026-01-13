"""
Clinical Evidence RAG Chain with confidence scoring.

Features:
- Evidence-based answers with citations
- Confidence scoring based on source quality
- Structured responses for clinical use
"""

from typing import List, Dict, Any

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.config import CHAT_MODEL, RETRIEVAL_K
from src.vectorstore import get_retriever


# Clinical Evidence RAG Prompt
CLINICAL_RAG_PROMPT = """You are a clinical research assistant helping analyze medical and scientific literature.

INSTRUCTIONS:
1. Carefully analyze the provided research excerpts
2. Provide a clear, evidence-based answer to the question
3. Structure your response appropriately for clinical/research use
4. Be precise about what the evidence shows vs. what it doesn't
5. Note any limitations or caveats in the available evidence
6. If the context doesn't contain sufficient information, clearly state this

RESEARCH CONTEXT:
{context}

QUESTION: {question}

Provide a comprehensive, evidence-based response:"""


# Confidence Assessment Prompt
CONFIDENCE_PROMPT = """Based on the sources provided, assess the evidence confidence level.

Consider:
- How directly do the sources address the question?
- How many independent sources support the answer?
- Are there any contradictions or limitations?
- Is this primary research, review, or guidelines?

Sources summary:
{sources_summary}

Question asked: {question}

Provide a brief (1-2 sentence) confidence assessment in this format:
"Evidence Confidence: [HIGH/MODERATE/LOW] - [brief explanation]"

Only output the confidence assessment, nothing else."""


def format_docs_with_metadata(docs: List[Document]) -> str:
    """Format documents with source information."""
    formatted_parts = []
    
    for i, doc in enumerate(docs, 1):
        file_name = doc.metadata.get("file_name", "Unknown")
        page = doc.metadata.get("page", "N/A")
        
        header = f"[Source {i}: {file_name}, Page {page}]"
        content = doc.page_content.strip()
        
        formatted_parts.append(f"{header}\n{content}")
    
    return "\n\n---\n\n".join(formatted_parts)


def get_sources_summary(sources: List[Dict]) -> str:
    """Create a summary of sources for confidence assessment."""
    summary_parts = []
    for i, src in enumerate(sources, 1):
        file_name = src.get("file_name", "Unknown")
        page = src.get("page", "N/A")
        content_preview = src.get("content", "")[:150]
        summary_parts.append(f"Source {i} ({file_name}, p.{page}): {content_preview}...")
    return "\n".join(summary_parts)


def create_rag_chain(
    vectorstore: Chroma,
    model_name: str = CHAT_MODEL,
    k: int = RETRIEVAL_K
):
    """Create a basic RAG chain."""
    llm = ChatOllama(model=model_name, temperature=0.1)
    retriever = get_retriever(vectorstore, k=k)
    prompt = ChatPromptTemplate.from_template(CLINICAL_RAG_PROMPT)
    
    def rag_invoke(question: str) -> str:
        docs = retriever.invoke(question)
        context = format_docs_with_metadata(docs)
        messages = prompt.invoke({"context": context, "question": question})
        response = llm.invoke(messages)
        return response.content
    
    return rag_invoke


def create_rag_chain_with_sources(
    vectorstore: Chroma,
    model_name: str = CHAT_MODEL,
    k: int = RETRIEVAL_K
):
    """Create a clinical RAG chain with sources and confidence."""
    llm = ChatOllama(model=model_name, temperature=0.1)
    retriever = get_retriever(vectorstore, k=k)
    
    main_prompt = ChatPromptTemplate.from_template(CLINICAL_RAG_PROMPT)
    confidence_prompt = ChatPromptTemplate.from_template(CONFIDENCE_PROMPT)
    
    def rag_with_sources(question: str) -> Dict[str, Any]:
        # Retrieve documents
        docs = retriever.invoke(question)
        
        # Format context
        context = format_docs_with_metadata(docs)
        
        # Generate main answer
        main_messages = main_prompt.invoke({"context": context, "question": question})
        main_response = llm.invoke(main_messages)
        
        # Build source info
        sources = []
        for doc in docs:
            sources.append({
                "content": doc.page_content,
                "file_name": doc.metadata.get("file_name", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "source": doc.metadata.get("source", ""),
            })
        
        # Generate confidence assessment
        sources_summary = get_sources_summary(sources)
        conf_messages = confidence_prompt.invoke({
            "sources_summary": sources_summary,
            "question": question
        })
        confidence_response = llm.invoke(conf_messages)
        
        return {
            "answer": main_response.content,
            "sources": sources,
            "confidence": confidence_response.content,
            "question": question,
            "num_sources": len(sources)
        }
    
    return rag_with_sources


def generate_markdown_summary(
    question: str,
    answer: str,
    sources: List[Dict],
    confidence: str,
    collection_name: str = ""
) -> str:
    """Generate a Markdown summary for export."""
    
    md_lines = [
        "# Evidence Summary Report",
        "",
        f"**Collection:** {collection_name}" if collection_name else "",
        f"**Generated:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## Question",
        "",
        question,
        "",
        "## Answer",
        "",
        answer,
        "",
        "## Evidence Assessment",
        "",
        confidence,
        "",
        "## Sources",
        "",
    ]
    
    for i, src in enumerate(sources, 1):
        file_name = src.get("file_name", "Unknown")
        page = src.get("page", "N/A")
        content = src.get("content", "")[:500]
        
        md_lines.extend([
            f"### Source {i}: {file_name} (Page {page})",
            "",
            f"> {content}...",
            "",
        ])
    
    md_lines.extend([
        "---",
        "",
        "*Generated by CiteCare - Clinical Evidence Q&A Assistant*"
    ])
    
    return "\n".join(md_lines)
