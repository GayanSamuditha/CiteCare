"""
Embedding utilities for the RAG system.

This module provides functions to create embeddings using Ollama's
local embedding models.
"""

from typing import List

from langchain_ollama import OllamaEmbeddings

from src.config import EMBEDDING_MODEL


def get_embeddings() -> OllamaEmbeddings:
    """
    Get an embeddings instance configured with the local Ollama model.
    
    Returns:
        OllamaEmbeddings instance ready to generate embeddings
    """
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def embed_text(text: str) -> List[float]:
    """
    Generate an embedding vector for a single text.
    
    Args:
        text: The text to embed
        
    Returns:
        A list of floats representing the embedding vector
    """
    embeddings = get_embeddings()
    return embeddings.embed_query(text)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embedding vectors for multiple texts.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding vectors
    """
    embeddings = get_embeddings()
    return embeddings.embed_documents(texts)


if __name__ == "__main__":
    # Example usage
    print(f"Using embedding model: {EMBEDDING_MODEL}")
    
    test_text = "This is a test sentence for embedding."
    vector = embed_text(test_text)
    
    print(f"Text: {test_text}")
    print(f"Embedding dimension: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")
