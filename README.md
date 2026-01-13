# LangChain RAG Learning Project

A hands-on project for learning LangChain by building a RAG (Retrieval-Augmented Generation) system with local LLMs via Ollama.

## What You'll Learn

- **LangChain Basics** - Prompts, chains, and LCEL
- **Document Processing** - Loading and splitting documents
- **Embeddings** - Converting text to vectors for semantic search
- **Vector Stores** - Storing and querying with ChromaDB
- **RAG Chains** - Combining retrieval with generation
- **Conversation Memory** - Enabling follow-up questions

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running

## Quick Start

### 1. Set Up Environment

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install Ollama Models

```bash
# Pull the chat model
ollama pull llama3.2:3b

# Pull the embedding model
ollama pull nomic-embed-text
```

### 3. Add Your Documents

Place your documents (PDF, TXT, MD files) in the `data/documents/` folder.

A sample document is included to get you started.

### 4. Run the CLI

```bash
python -m src.cli
```

## Project Structure

```
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration settings
│   ├── document_loader.py  # Load and split documents
│   ├── embeddings.py       # Embedding functions
│   ├── vectorstore.py      # ChromaDB operations
│   ├── rag_chain.py        # RAG chain implementation
│   └── cli.py              # Command-line interface
├── data/
│   └── documents/          # Your documents go here
└── notebooks/
    └── learning/           # Jupyter notebooks for learning
        ├── 01_langchain_basics.ipynb
        ├── 02_document_processing.ipynb
        ├── 03_rag_system.ipynb
        └── 04_conversation_memory.ipynb
```

## Learning Path

### Phase 1: Foundations (Day 1-2)
Open `notebooks/learning/01_langchain_basics.ipynb` to learn:
- Connecting to Ollama
- Using prompt templates
- Building chains with LCEL

### Phase 2: Document Processing (Day 3-4)
Open `notebooks/learning/02_document_processing.ipynb` to learn:
- Loading different document types
- Text splitting strategies
- Creating embeddings

### Phase 3: Build RAG System (Day 5-7)
Open `notebooks/learning/03_rag_system.ipynb` to learn:
- Vector stores with ChromaDB
- Similarity search
- Building the RAG chain

### Phase 4: Polish (Day 8+)
Open `notebooks/learning/04_conversation_memory.ipynb` to learn:
- Conversation memory
- Source citations
- Using the CLI

## CLI Commands

When running the CLI (`python -m src.cli`):

| Command | Description |
|---------|-------------|
| *(your question)* | Ask a question about your documents |
| `sources` | Show source documents from last answer |
| `history` | Show conversation history |
| `clear` | Clear conversation history |
| `reload` | Reload documents from disk |
| `quit` | Exit the program |

## Configuration

Edit `src/config.py` to customize:

```python
CHAT_MODEL = "llama3.2:3b"      # Change the chat model
EMBEDDING_MODEL = "nomic-embed-text"  # Change embedding model
CHUNK_SIZE = 500                # Adjust chunk size
CHUNK_OVERLAP = 50              # Adjust overlap
RETRIEVAL_K = 3                 # Number of documents to retrieve
```

## Next Steps

After completing this project, you can:

1. **Add more documents** - Try PDFs, different topics
2. **Experiment with prompts** - Customize the system prompt in `rag_chain.py`
3. **Try different models** - Swap `llama3.2:3b` for `mistral` or others
4. **Add a web UI** - Build with Streamlit or Gradio
5. **Explore agents** - LangChain agents can use tools and make decisions

## Resources

- [LangChain Documentation](https://python.langchain.com)
- [Ollama](https://ollama.ai)
- [ChromaDB](https://www.trychroma.com)

## License

MIT License - See LICENSE file for details.
