"""
Qdrant-backed vector store for the RAG pipeline.

Deployment modes (set via `location`):
  ":memory:"          – ephemeral, in-process (unit tests / quick experiments)
  "./qdrant_store"    – on-disk persistence, no server needed (default)
  "http://host:6333"  – remote Qdrant server or Docker container
  "https://…qdrant.io"– Qdrant Cloud (pair with QDRANT_API_KEY env var)
"""

import logging
import os
import uuid

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

logger = logging.getLogger(__name__)

# Vector output sizes per model — extend as needed
EMBEDDING_DIMENSIONS: dict[str, int] = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "multi-qa-MiniLM-L6-cos-v1": 384,
}


class QdrantStore:
    """
    Production-grade vector store backed by Qdrant + langchain-qdrant.

    Usage
    -----
    # First run — ingest documents
    store = QdrantStore(collection_name="rag_docs")
    store.build_from_documents(documents)

    # Subsequent runs — reuse existing collection
    store = QdrantStore(collection_name="rag_docs")
    store.load()
    results = store.similarity_search("what is RAG?")
    """

    def __init__(
        self,
        collection_name: str = "rag_documents",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1500,
        chunk_overlap: int = 300,
        location: str = "./qdrant_store",
        api_key: str | None = None,
        embeddings: Embeddings | None = None,
    ):
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.location = location
        self.api_key = api_key or os.environ.get("QDRANT_API_KEY")

        self.embeddings = embeddings or HuggingFaceEmbeddings(
            model_name=embedding_model
        )
        self.client = self._build_client()
        self.vectorstore: QdrantVectorStore | None = None

        logger.info(
            "Qdrant client ready | location=%r | model=%s", location, embedding_model
        )

    def _build_client(self) -> QdrantClient:
        """Return a QdrantClient wired to the chosen backend."""
        if self.location == ":memory:":
            return QdrantClient(location=":memory:")
        if self.location.startswith(("http://", "https://")):
            return QdrantClient(url=self.location, api_key=self.api_key)
        # Default: local on-disk store (no Docker or internet required)
        return QdrantClient(path=self.location)

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist."""
        dim = EMBEDDING_DIMENSIONS.get(self.embedding_model_name, 384)
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            logger.info(
                "Creating collection '%s' (dim=%d, distance=Cosine)...",
                self.collection_name,
                dim,
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        else:
            logger.info("Collection '%s' already exists.", self.collection_name)

    def _chunk_id(self, source: str, index: int) -> str:
        """Deterministic UUID from (source path, chunk index).

        Qdrant upserts by ID, so re-ingesting the same file with the same
        chunk settings overwrites existing points instead of duplicating them.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}::{index}"))

    def build_from_documents(self, documents: list[Document]) -> None:
        """
        Chunk → embed → upsert documents into Qdrant.

        Uses deterministic point IDs (source path + chunk index), so this
        method is safe to call multiple times:
        - Same file re-ingested → existing points are overwritten in-place.
        - New files → appended without touching existing vectors.
        """
        logger.info("Ingesting %d document(s) into Qdrant...", len(documents))
        self._ensure_collection()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)
        logger.info("Split into %d chunk(s)", len(chunks))

        logger.info(
            "Upserting %d chunk(s) into '%s'...", len(chunks), self.collection_name
        )
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
        ids = [
            self._chunk_id(chunk.metadata.get("source", ""), i)
            for i, chunk in enumerate(chunks)
        ]
        self.vectorstore.add_documents(chunks, ids=ids)
        logger.info("Done. %d chunk(s) stored in Qdrant.", len(chunks))
        return len(chunks)

    def load(self) -> None:
        """
        Attach to an existing Qdrant collection without re-ingesting.
        Call this on startup if the data was already ingested in a previous run.
        """
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            raise RuntimeError(
                f"Collection '{self.collection_name}' not found. "
                "Run build_from_documents() first."
            )
        logger.info("Loading collection '%s'...", self.collection_name)
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
        logger.info("Vector store ready.")

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Return the top-k most relevant chunks for a natural-language query."""
        if self.vectorstore is None:
            raise RuntimeError(
                "Vector store not initialised. "
                "Call build_from_documents() or load() first."
            )
        return self.vectorstore.similarity_search(query, k=k)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from .data_loader import load_documents  # local import for script-mode only

    # Quick test to verify the store can be built and queried
    documents = load_documents("data/pdfs")
    store = QdrantStore(collection_name="test_collection", location=":memory:")
    store.build_from_documents(documents)
    results = store.similarity_search("What is RAG?", k=2)
    for i, doc in enumerate(results, 1):
        print(f"\n[Result {i}]\n{doc.page_content[:500]}...")
