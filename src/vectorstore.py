"""
Vector store operations using ChromaDB.

This module provides functions to:
1. Create and persist a vector store
2. Load an existing vector store
3. Add documents to the store
4. Perform similarity searches
"""

from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.config import CHROMA_PERSIST_DIR, COLLECTION_NAME, RETRIEVAL_K
from src.embeddings import get_embeddings


def create_vectorstore(
    documents: List[Document],
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME
) -> Chroma:
    """
    Create a new vector store from documents and persist it to disk.
    
    Args:
        documents: List of Document objects to add to the store
        persist_directory: Directory to save the vector store
        collection_name: Name for the collection
        
    Returns:
        Chroma vector store instance
    """
    embeddings = get_embeddings()
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    print(f"Created vector store with {len(documents)} documents")
    print(f"Persisted to: {persist_directory}")
    
    return vectorstore


def load_vectorstore(
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME
) -> Optional[Chroma]:
    """
    Load an existing vector store from disk.
    
    Args:
        persist_directory: Directory where the vector store is saved
        collection_name: Name of the collection to load
        
    Returns:
        Chroma vector store instance, or None if not found
    """
    if not Path(persist_directory).exists():
        print(f"No vector store found at {persist_directory}")
        return None
    
    embeddings = get_embeddings()
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    
    # Get the count of documents
    count = vectorstore._collection.count()
    print(f"Loaded vector store with {count} documents")
    
    return vectorstore


def get_or_create_vectorstore(
    documents: Optional[List[Document]] = None,
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME
) -> Chroma:
    """
    Get existing vector store or create a new one.
    
    If documents are provided and the store doesn't exist, creates a new one.
    If the store exists, loads it (ignoring any provided documents).
    
    Args:
        documents: Optional list of documents to create a new store
        persist_directory: Directory for the vector store
        collection_name: Name of the collection
        
    Returns:
        Chroma vector store instance
    """
    existing = load_vectorstore(persist_directory, collection_name)
    
    if existing is not None:
        return existing
    
    if documents is None or len(documents) == 0:
        raise ValueError("No existing store found and no documents provided")
    
    return create_vectorstore(documents, persist_directory, collection_name)


def similarity_search(
    vectorstore: Chroma,
    query: str,
    k: int = RETRIEVAL_K
) -> List[Document]:
    """
    Search for documents similar to the query.
    
    Args:
        vectorstore: The Chroma vector store to search
        query: The search query
        k: Number of results to return
        
    Returns:
        List of similar Document objects
    """
    results = vectorstore.similarity_search(query, k=k)
    return results


def similarity_search_with_score(
    vectorstore: Chroma,
    query: str,
    k: int = RETRIEVAL_K
) -> List[tuple]:
    """
    Search for documents with similarity scores.
    
    Args:
        vectorstore: The Chroma vector store to search
        query: The search query
        k: Number of results to return
        
    Returns:
        List of (Document, score) tuples
    """
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results


def get_retriever(vectorstore: Chroma, k: int = RETRIEVAL_K):
    """
    Get a retriever from the vector store for use in chains.
    
    Args:
        vectorstore: The Chroma vector store
        k: Number of documents to retrieve
        
    Returns:
        A retriever object
    """
    return vectorstore.as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    # Example usage
    from src.document_loader import load_and_split
    
    # Load and split documents
    chunks = load_and_split()
    
    if chunks:
        # Create vector store
        vectorstore = create_vectorstore(chunks)
        
        # Test similarity search
        query = "What is supervised learning?"
        results = similarity_search(vectorstore, query, k=2)
        
        print(f"\nQuery: {query}")
        print(f"Found {len(results)} relevant chunks:")
        for i, doc in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(doc.page_content[:200] + "...")
