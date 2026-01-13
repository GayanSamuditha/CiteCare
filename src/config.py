"""
Configuration settings for the LangChain RAG system.
"""

# Ollama Model Configuration
CHAT_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"

# Improved chunking for better accuracy
CHUNK_SIZE = 400  # Smaller chunks for more precise retrieval
CHUNK_OVERLAP = 100  # More overlap for better context

# Vector Store Configuration
CHROMA_PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "default"

# Retrieval Configuration  
RETRIEVAL_K = 5  # Retrieve more documents for better context

# Paths
DOCUMENTS_DIR = "data/documents"
COLLECTIONS_DIR = "data/collections"  # For multiple collections

# Retention (days) default for collections
DEFAULT_RETENTION_DAYS = 30
