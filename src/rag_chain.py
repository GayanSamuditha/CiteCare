"""
RAG (Retrieval-Augmented Generation) chain implementation.

This module provides the main RAG functionality that:
1. Retrieves relevant documents based on a query
2. Passes context to the LLM
3. Generates an answer grounded in the retrieved documents
"""

from typing import List, Optional

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.config import CHAT_MODEL, RETRIEVAL_K
from src.vectorstore import get_retriever


# Default RAG prompt template
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context.

Use ONLY the following context to answer the question. If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer that question."

Context:
{context}

Question: {question}

Answer:"""


def format_docs(docs: List[Document]) -> str:
    """
    Format a list of documents into a single string for the prompt.
    
    Args:
        docs: List of Document objects
        
    Returns:
        Formatted string with all document contents
    """
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(
    vectorstore: Chroma,
    model_name: str = CHAT_MODEL,
    k: int = RETRIEVAL_K,
    custom_prompt: Optional[str] = None
):
    """
    Create a RAG chain that retrieves context and generates answers.
    
    Args:
        vectorstore: The Chroma vector store to retrieve from
        model_name: Name of the Ollama model to use
        k: Number of documents to retrieve
        custom_prompt: Optional custom prompt template
        
    Returns:
        A runnable RAG chain
    """
    # Initialize components
    llm = ChatOllama(model=model_name)
    retriever = get_retriever(vectorstore, k=k)
    
    # Use custom prompt or default
    prompt_text = custom_prompt or RAG_PROMPT_TEMPLATE
    prompt = ChatPromptTemplate.from_template(prompt_text)
    
    # Build the chain using LCEL
    # The chain:
    # 1. Takes a question
    # 2. Retrieves relevant documents
    # 3. Formats them as context
    # 4. Passes to the prompt template
    # 5. Sends to the LLM
    # 6. Parses the output as a string
    
    rag_chain = (
        RunnableParallel(
            context=retriever | format_docs,
            question=RunnablePassthrough()
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


def create_rag_chain_with_sources(
    vectorstore: Chroma,
    model_name: str = CHAT_MODEL,
    k: int = RETRIEVAL_K
):
    """
    Create a RAG chain that also returns the source documents.
    
    This is useful when you want to show users which documents
    were used to generate the answer.
    
    Args:
        vectorstore: The Chroma vector store to retrieve from
        model_name: Name of the Ollama model to use
        k: Number of documents to retrieve
        
    Returns:
        A runnable that returns both answer and source documents
    """
    llm = ChatOllama(model=model_name)
    retriever = get_retriever(vectorstore, k=k)
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    def format_docs_and_keep(docs: List[Document]) -> dict:
        """Format docs and keep references for sources."""
        return {
            "formatted": format_docs(docs),
            "sources": docs
        }
    
    # This chain returns both the answer and the source documents
    def run_rag_with_sources(question: str) -> dict:
        # Retrieve documents
        docs = retriever.invoke(question)
        
        # Format context
        context = format_docs(docs)
        
        # Generate answer
        messages = prompt.invoke({"context": context, "question": question})
        response = llm.invoke(messages)
        answer = response.content
        
        return {
            "answer": answer,
            "sources": docs,
            "question": question
        }
    
    return run_rag_with_sources


def query_rag(
    chain,
    question: str,
    verbose: bool = False
) -> str:
    """
    Query the RAG chain with a question.
    
    Args:
        chain: The RAG chain to query
        question: The question to ask
        verbose: If True, print additional information
        
    Returns:
        The generated answer
    """
    if verbose:
        print(f"Question: {question}")
        print("Retrieving relevant documents...")
    
    answer = chain.invoke(question)
    
    if verbose:
        print(f"\nAnswer: {answer}")
    
    return answer


if __name__ == "__main__":
    # Example usage
    from src.document_loader import load_and_split
    from src.vectorstore import get_or_create_vectorstore
    
    # Load documents and create/load vector store
    chunks = load_and_split()
    vectorstore = get_or_create_vectorstore(chunks)
    
    # Create RAG chain
    rag_chain = create_rag_chain(vectorstore)
    
    # Test with a question
    question = "What is supervised learning?"
    print(f"\nQuestion: {question}")
    print("-" * 50)
    
    answer = query_rag(rag_chain, question)
    print(f"\nAnswer:\n{answer}")
