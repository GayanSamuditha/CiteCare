"""
Document loading and text splitting utilities.

This module provides functions to:
1. Load documents from various file formats (PDF, TXT, MD)
2. Split documents into smaller chunks for embedding
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, DOCUMENTS_DIR


def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF file and return a list of Document objects.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of Document objects, one per page
    """
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_text(file_path: str) -> List[Document]:
    """
    Load a text file (.txt or .md) and return a list of Document objects.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        List containing a single Document object
    """
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def load_directory(
    directory_path: str = DOCUMENTS_DIR,
    glob_pattern: str = "**/*.*"
) -> List[Document]:
    """
    Load all supported documents from a directory.
    
    Args:
        directory_path: Path to the directory containing documents
        glob_pattern: Pattern to match files (default: all files)
        
    Returns:
        List of Document objects from all files
    """
    documents = []
    dir_path = Path(directory_path)
    
    if not dir_path.exists():
        print(f"Warning: Directory {directory_path} does not exist")
        return documents
    
    # Load PDFs
    pdf_files = list(dir_path.glob("**/*.pdf"))
    for pdf_file in pdf_files:
        try:
            docs = load_pdf(str(pdf_file))
            documents.extend(docs)
            print(f"Loaded {len(docs)} pages from {pdf_file.name}")
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
    
    # Load text files
    txt_files = list(dir_path.glob("**/*.txt")) + list(dir_path.glob("**/*.md"))
    for txt_file in txt_files:
        try:
            docs = load_text(str(txt_file))
            documents.extend(docs)
            print(f"Loaded {txt_file.name}")
        except Exception as e:
            print(f"Error loading {txt_file}: {e}")
    
    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.
    
    Uses RecursiveCharacterTextSplitter which tries to split on:
    1. Paragraphs (\\n\\n)
    2. Lines (\\n)
    3. Sentences (. ! ?)
    4. Words (spaces)
    
    Args:
        documents: List of Document objects to split
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of smaller Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    
    return chunks


def load_and_split(
    directory_path: str = DOCUMENTS_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Document]:
    """
    Convenience function to load documents and split them in one step.
    
    Args:
        directory_path: Path to directory containing documents
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunked Document objects ready for embedding
    """
    documents = load_directory(directory_path)
    
    if not documents:
        print("No documents found to process")
        return []
    
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    return chunks


if __name__ == "__main__":
    # Example usage
    print("Loading documents from:", DOCUMENTS_DIR)
    chunks = load_and_split()
    
    if chunks:
        print(f"\nFirst chunk preview:")
        print("-" * 50)
        print(chunks[0].page_content[:200])
        print("-" * 50)
