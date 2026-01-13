#!/usr/bin/env python3
"""
Command-line interface for the RAG system.

This module provides an interactive CLI for:
1. Loading and indexing documents
2. Asking questions and getting answers
3. Viewing source documents used for answers
4. Maintaining conversation history
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from langchain_core.messages import HumanMessage, AIMessage

from src.config import DOCUMENTS_DIR, CHROMA_PERSIST_DIR
from src.document_loader import load_and_split
from src.vectorstore import get_or_create_vectorstore, create_vectorstore
from src.rag_chain import create_rag_chain_with_sources


class ConversationMemory:
    """Simple conversation memory to track chat history."""
    
    def __init__(self, max_history: int = 10):
        self.history: List[tuple] = []  # List of (question, answer) tuples
        self.max_history = max_history
    
    def add(self, question: str, answer: str):
        """Add a Q&A pair to history."""
        self.history.append((question, answer))
        # Keep only recent history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_context(self) -> str:
        """Get formatted conversation history for context."""
        if not self.history:
            return ""
        
        context_parts = ["Previous conversation:"]
        for q, a in self.history[-3:]:  # Last 3 exchanges
            context_parts.append(f"Human: {q}")
            context_parts.append(f"Assistant: {a[:200]}...")  # Truncate long answers
        
        return "\n".join(context_parts)
    
    def clear(self):
        """Clear the conversation history."""
        self.history = []


def print_welcome():
    """Print welcome message."""
    print("\n" + "=" * 60)
    print("  📚 LangChain RAG System - Document Q&A")
    print("=" * 60)
    print("\nCommands:")
    print("  • Type your question and press Enter")
    print("  • 'sources' - Show source documents from last answer")
    print("  • 'history' - Show conversation history")
    print("  • 'clear'   - Clear conversation history")
    print("  • 'reload'  - Reload documents from disk")
    print("  • 'quit'    - Exit the program")
    print("-" * 60 + "\n")


def print_sources(sources):
    """Print source documents."""
    if not sources:
        print("No sources available.")
        return
    
    print("\n📄 Source Documents:")
    print("-" * 40)
    for i, doc in enumerate(sources, 1):
        print(f"\n[Source {i}]")
        # Show metadata if available
        if doc.metadata:
            source_file = doc.metadata.get('source', 'Unknown')
            print(f"File: {Path(source_file).name}")
        print(f"Content: {doc.page_content[:300]}...")
    print("-" * 40 + "\n")


def index_documents(force_reindex: bool = False) -> bool:
    """
    Index documents from the documents directory.
    
    Args:
        force_reindex: If True, reindex even if index exists
        
    Returns:
        True if successful, False otherwise
    """
    docs_path = Path(DOCUMENTS_DIR)
    
    if not docs_path.exists():
        print(f"❌ Documents directory not found: {DOCUMENTS_DIR}")
        print(f"   Please create it and add documents to index.")
        return False
    
    # Check for existing index
    if not force_reindex and Path(CHROMA_PERSIST_DIR).exists():
        print("📦 Found existing index. Use 'reload' to reindex.")
        return True
    
    print(f"📂 Loading documents from: {DOCUMENTS_DIR}")
    chunks = load_and_split()
    
    if not chunks:
        print("❌ No documents found to index.")
        return False
    
    print(f"🔄 Creating vector index...")
    create_vectorstore(chunks)
    print("✅ Indexing complete!")
    
    return True


def run_interactive(rag_func):
    """
    Run the interactive Q&A loop.
    
    Args:
        rag_func: Function that takes a question and returns answer + sources
    """
    memory = ConversationMemory()
    last_sources = []
    
    print_welcome()
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == 'quit':
                print("\nGoodbye! 👋\n")
                break
            
            elif user_input.lower() == 'sources':
                print_sources(last_sources)
                continue
            
            elif user_input.lower() == 'history':
                if memory.history:
                    print("\n📜 Conversation History:")
                    for i, (q, a) in enumerate(memory.history, 1):
                        print(f"\n{i}. Q: {q}")
                        print(f"   A: {a[:150]}...")
                    print()
                else:
                    print("\nNo conversation history yet.\n")
                continue
            
            elif user_input.lower() == 'clear':
                memory.clear()
                print("\n🗑️  Conversation history cleared.\n")
                continue
            
            elif user_input.lower() == 'reload':
                print("\n🔄 Reloading documents...")
                if index_documents(force_reindex=True):
                    print("Please restart the application to use the new index.\n")
                continue
            
            # Process the question
            print("\n🤔 Thinking...\n")
            
            result = rag_func(user_input)
            answer = result["answer"]
            last_sources = result["sources"]
            
            # Display the answer
            print(f"Assistant: {answer}\n")
            
            # Add to memory
            memory.add(user_input, answer)
            
            # Hint about sources
            print("(Type 'sources' to see the documents used)\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="LangChain RAG System - Ask questions about your documents"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force reindexing of documents"
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default=DOCUMENTS_DIR,
        help=f"Directory containing documents (default: {DOCUMENTS_DIR})"
    )
    
    args = parser.parse_args()
    
    # Index documents
    if not index_documents(force_reindex=args.reindex):
        sys.exit(1)
    
    # Load the vector store
    print("📦 Loading vector store...")
    try:
        chunks = load_and_split()
        vectorstore = get_or_create_vectorstore(chunks)
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        sys.exit(1)
    
    # Create RAG function with sources
    print("🔧 Initializing RAG chain...")
    rag_func = create_rag_chain_with_sources(vectorstore)
    
    # Run interactive loop
    run_interactive(rag_func)


if __name__ == "__main__":
    main()
