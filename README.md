# DocuChat

A chatbot that lets you have multi-turn conversations with your PDF documents.  
Upload PDFs through a web UI, ask questions, and get answers grounded in your documents.  
Embeddings are **session-based** — they live only in memory and vanish when the session is cleared or the server restarts.

## Stack

- **LangChain** — LCEL chain, prompt templates, conversation history
- **Qdrant** — in-memory vector store per session (no server required)
- **HuggingFace** — `all-MiniLM-L6-v2` for embeddings
- **CrossEncoder** — `ms-marco-MiniLM-L-6-v2` for re-ranking retrieved chunks
- **RAGAS** — reference-free evaluation (faithfulness, relevancy, context precision)
- **Azure OpenAI** — LLM backend
- **FastAPI** — REST API backend
- **React + Vite + Tailwind CSS** — web frontend

## Project Structure

```
backend/                 # Python project root
  pyproject.toml
  uv.lock
  .env                   # ← create this (see below)
  app.py                 # FastAPI application
  session_manager.py     # Per-session in-memory RAG lifecycle
  rag/
    data_loader.py       # PDF ingestion via PyMuPDF
    vectorstore.py       # Qdrant-backed vector store
    search.py            # RAG chain with session history
    llm.py               # Azure OpenAI initialisation
    prompt.py            # System prompt
  evaluate.py            # RAGAS evaluation: baseline vs re-ranked
  data/
    pdfs/                # Drop your PDFs here (evaluation)
frontend/                # React + Vite + Tailwind frontend
README.md
```

## Setup

**1. Clone and install Python dependencies**

```bash
git clone https://github.com/<your-username>/DocuChat.git
cd DocuChat/backend
uv sync
```

**2. Configure environment variables**

Create a `.env` file inside `backend/`:

```env
DIAL_API_KEY=your_api_key
DIAL_ENDPOINT=https://your-endpoint.openai.azure.com
DIAL_API_VERSION=2024-02-01
DIAL_DEPLOYMENT=your_deployment_name
```

**3. Install frontend dependencies**

```bash
cd frontend
npm install
cd ..
```

## Running the Web App

Open two terminals:

**Terminal 1 — Backend**

```bash
cd backend
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`.  
The embedding model is pre-warmed at startup (first launch may take a few seconds to download `all-MiniLM-L6-v2`).

**Terminal 2 — Frontend**

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173` in your browser.

### Usage

1. The app creates a new session automatically on page load.
2. Click the upload area on the left to select one or more PDF files and hit **Upload**.
3. Once documents are indexed, the chat input unlocks — ask anything about your PDFs.
4. Click **Clear Session** in the header to wipe all uploaded documents and start fresh.

> Each browser tab gets its own isolated session. Refreshing the page starts a new session.

---

## Evaluation

The project includes a RAGAS-based evaluation script that compares **baseline** (plain ANN retrieval) against **re-ranked** (ANN + CrossEncoder) pipelines.

```bash
cd backend
uv run python evaluate.py
```

The script:

1. Ingests all PDFs from `data/pdfs/` into an in-memory Qdrant collection.
2. Runs curated questions through both pipelines.
3. Evaluates with three reference-free RAGAS metrics: **Faithfulness**, **Response Relevancy**, and **Context Precision**.
4. Prints a side-by-side comparison table and saves `results_baseline.json` / `results_reranked.json` (git-ignored).

> Make sure `data/pdfs/` contains at least one PDF before running.
