"""
Enhanced document loading and text splitting utilities.

Features:
- Page number tracking for PDFs
- Improved chunking with better overlap
- Metadata preservation for source citations
"""

from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, DOCUMENTS_DIR


def load_pdf_with_pages(file_path: str) -> List[Document]:
    """
    Load a PDF file with page number metadata.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of Document objects with page metadata
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    # Enhance metadata with file info
    file_name = Path(file_path).name
    for i, page in enumerate(pages):
        page.metadata["source"] = file_path
        page.metadata["file_name"] = file_name
        page.metadata["page"] = page.metadata.get("page", i) + 1  # 1-indexed
        page.metadata["total_pages"] = len(pages)
    
    return pages


def load_text_file(file_path: str) -> List[Document]:
    """
    Load a text file with metadata.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        List containing Document objects
    """
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    
    file_name = Path(file_path).name
    for doc in docs:
        doc.metadata["source"] = file_path
        doc.metadata["file_name"] = file_name
        doc.metadata["page"] = 1
    
    return docs


def load_directory(
    directory_path: str = DOCUMENTS_DIR,
    collection_name: Optional[str] = None
) -> List[Document]:
    """
    Load all documents from a directory with enhanced metadata.
    
    Args:
        directory_path: Path to the directory
        collection_name: Optional collection name to tag documents
        
    Returns:
        List of Document objects
    """
    documents = []
    dir_path = Path(directory_path)
    
    if not dir_path.exists():
        print(f"Warning: Directory {directory_path} does not exist")
        return documents
    
    # Load PDFs with page tracking
    pdf_files = list(dir_path.glob("**/*.pdf"))
    for pdf_file in pdf_files:
        try:
            docs = load_pdf_with_pages(str(pdf_file))
            if collection_name:
                for doc in docs:
                    doc.metadata["collection"] = collection_name
            documents.extend(docs)
            print(f"✓ Loaded {len(docs)} pages from {pdf_file.name}")
        except Exception as e:
            print(f"✗ Error loading {pdf_file}: {e}")
    
    # Load text files
    txt_files = list(dir_path.glob("**/*.txt")) + list(dir_path.glob("**/*.md"))
    for txt_file in txt_files:
        try:
            docs = load_text_file(str(txt_file))
            if collection_name:
                for doc in docs:
                    doc.metadata["collection"] = collection_name
            documents.extend(docs)
            print(f"✓ Loaded {txt_file.name}")
        except Exception as e:
            print(f"✗ Error loading {txt_file}: {e}")
    
    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Document]:
    """
    Split documents into chunks while preserving metadata.
    
    Uses smaller chunks with more overlap for better context.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        keep_separator=True
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Add chunk index to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks


def load_and_split(
    directory_path: str = DOCUMENTS_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    collection_name: Optional[str] = None
) -> List[Document]:
    """
    Load and split documents in one step.
    """
    documents = load_directory(directory_path, collection_name)
    
    if not documents:
        print("No documents found")
        return []
    
    return split_documents(documents, chunk_size, chunk_overlap)
