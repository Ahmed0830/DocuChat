from pathlib import Path
from typing import Any

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyMuPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

# Registry mapping file extension → (LoaderClass, extra_kwargs).
# To support a new format, add an entry here — load_from_paths never needs
# modification (Open/Closed Principle).
_LOADER_REGISTRY: dict[str, tuple[Any, dict[str, Any]]] = {
    ".pdf": (PyMuPDFLoader, {}),
    ".docx": (Docx2txtLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf-8"}),
    ".md": (TextLoader, {"encoding": "utf-8"}),
}

SUPPORTED_EXTENSIONS: set[str] = set(_LOADER_REGISTRY.keys())


def load_from_paths(paths: list[str]) -> list[Document]:
    """Load documents from a list of file-system paths.

    Supported formats are determined by _LOADER_REGISTRY.
    """
    documents: list[Document] = []
    for path in paths:
        suffix = Path(path).suffix.lower()
        entry = _LOADER_REGISTRY.get(suffix)
        if entry is None:
            raise ValueError(f"Unsupported file type: {suffix}")
        loader_cls, kwargs = entry
        documents.extend(loader_cls(path, **kwargs).load())
    return documents
