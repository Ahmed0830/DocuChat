# pdf-rag-chatbot

A CLI chatbot that lets you have multi-turn conversations with your PDF documents.  
Uses Retrieval-Augmented Generation (RAG) with Qdrant for vector storage and Azure OpenAI for responses.

## Stack

- **LangChain** — LCEL chain, prompt templates, conversation history
- **Qdrant** — local on-disk vector store (no server required)
- **HuggingFace** — `all-MiniLM-L6-v2` for embeddings
- **Azure OpenAI** — LLM backend

## Project Structure

```
main.py              # CLI chatbot entry point
src/
  data_loader.py     # PDF ingestion via PyMuPDF
  vectorstore.py     # Qdrant-backed vector store
  search.py          # RAG chain with session history
  llm.py             # Azure OpenAI initialisation
  prompt.py          # System prompt
data/
  pdfs/              # Drop your PDFs here
```

## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/<your-username>/pdf-rag-chatbot.git
cd pdf-rag-chatbot
uv sync
```

**2. Configure environment variables**

Create a `.env` file in the project root:

```env
DIAL_API_KEY=your_api_key
DIAL_ENDPOINT=https://your-endpoint.openai.azure.com
DIAL_API_VERSION=2024-02-01
DIAL_DEPLOYMENT=your_deployment_name
```

**3. Ingest your PDFs**

Drop PDF files into `data/pdfs/`, then run the ingestion once:

```python
from src.data_loader import load_documents
from src.vectorstore import QdrantStore

store = QdrantStore(collection_name="rag_documents", location="./qdrant_store")
docs = load_documents("data/pdfs")
store.build_from_documents(docs)
```

To add more PDFs later, just drop them in and call `build_from_documents()` again — existing embeddings are not re-processed, new ones are appended.

**4. Start the chatbot**

```bash
uv run main.py
```

```
RAG Chatbot  (type 'exit' or 'quit' to stop)

You: what is the main topic of the document?
```
