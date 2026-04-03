import uuid
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rag.search import RAGSearch
from rag.vectorstore import QdrantStore


@dataclass
class _Session:
    rag: RAGSearch
    file_count: int = 0


class SessionManager:
    """Manages per-session in-memory RAG pipelines.

    Each session gets its own QdrantStore(location=":memory:") so
    embeddings are never persisted to disk and vanish when the session
    is deleted (or the server restarts).

    A single HuggingFaceEmbeddings instance is shared across all sessions
    so the model is only loaded once at server startup.
    """

    def __init__(self, shared_embeddings: Embeddings) -> None:
        self._embeddings = shared_embeddings
        self._sessions: dict[str, _Session] = {}

    def _build_session(self, session_id: str) -> "_Session":
        """Factory method — override in subclasses to swap the RAG implementation."""
        store = QdrantStore(
            collection_name=session_id,
            location=":memory:",
            embeddings=self._embeddings,
        )
        rag = RAGSearch(vectorstore=store)
        return _Session(rag=rag)

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = self._build_session(session_id)
        return session_id

    def get_session(self, session_id: str) -> RAGSearch:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session.rag

    def delete_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.rag.vectorstore.client.close()

    def add_documents(
        self, session_id: str, docs: list[Document], file_count: int
    ) -> int:
        """Ingest documents into the session's vector store.

        Returns the number of chunks stored.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        chunk_count = session.rag.vectorstore.build_from_documents(docs)
        session.file_count += file_count
        return chunk_count

    def session_status(self, session_id: str) -> dict:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return {
            "has_documents": session.rag.vectorstore.vectorstore is not None,
            "file_count": session.file_count,
        }
