import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel
from rag.data_loader import SUPPORTED_EXTENSIONS, load_from_paths
from session_manager import SessionManager

load_dotenv()
logging.disable(logging.CRITICAL)

_manager: SessionManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _manager
    # Pre-warm the embedding model once so the first upload is not slow
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    _manager = SessionManager(shared_embeddings=embeddings)
    yield
    _manager = None


app = FastAPI(title="RAG Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_manager() -> SessionManager:
    if _manager is None:  # pragma: no cover
        raise RuntimeError("Session manager not initialised")
    return _manager


# ── Pydantic models ────────────────────────────────────────────────────────────


class SessionResponse(BaseModel):
    session_id: str


class StatusResponse(BaseModel):
    has_documents: bool
    file_count: int


class UploadResponse(BaseModel):
    files_processed: int
    chunks_stored: int


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.post(
    "/api/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_session():
    session_id = get_manager().create_session()
    return SessionResponse(session_id=session_id)


@app.get("/api/sessions/{session_id}/status", response_model=StatusResponse)
async def session_status(session_id: str):
    try:
        result = get_manager().session_status(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    return StatusResponse(**result)


@app.post("/api/sessions/{session_id}/upload", response_model=UploadResponse)
async def upload_documents(
    session_id: str,
    files: Annotated[list[UploadFile], File()],
):
    manager = get_manager()
    try:
        manager.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    tmp_paths: list[str] = []
    try:
        for file in files:
            suffix = Path(file.filename or "").suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
                )
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                tmp_paths.append(tmp.name)

        docs = load_from_paths(tmp_paths)
        chunk_count = manager.add_documents(session_id, docs, file_count=len(files))
    finally:
        for path in tmp_paths:
            try:
                os.unlink(path)
            except OSError:
                pass

    return UploadResponse(files_processed=len(files), chunks_stored=chunk_count)


@app.post("/api/sessions/{session_id}/chat", response_model=ChatResponse)
async def chat(session_id: str, body: ChatRequest):
    try:
        rag = get_manager().get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    answer = rag.search(body.query, session_id=session_id)
    return ChatResponse(answer=answer)


@app.delete("/api/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str):
    try:
        get_manager().delete_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
