"""
Configuration settings for the LangChain RAG system.

This module centralizes all configuration including model names,
chunk sizes, and other parameters.
"""

# Ollama Model Configuration
CHAT_MODEL = "llama3.2:3b"  # Model for text generation
EMBEDDING_MODEL = "nomic-embed-text"  # Model for embeddings

# Text Splitting Configuration
CHUNK_SIZE = 500  # Number of characters per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks for context continuity

# Vector Store Configuration
CHROMA_PERSIST_DIR = "chroma_db"  # Directory to persist vector database
COLLECTION_NAME = "documents"  # Name of the collection in ChromaDB

# Retrieval Configuration
RETRIEVAL_K = 3  # Number of documents to retrieve

# Paths
DOCUMENTS_DIR = "data/documents"  # Directory containing documents to index
