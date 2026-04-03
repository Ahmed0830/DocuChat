from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_core.documents import Document


def load_documents(directory_path: str) -> list[Document]:
    # Load all PDF documents from the specified directory
    directory_path = Path(directory_path).resolve()
    loader = DirectoryLoader(
        directory_path, glob="**/*.pdf", show_progress=False, loader_cls=PyMuPDFLoader
    )

    documents = loader.load()

    return documents


if __name__ == "__main__":
    documents = load_documents("data/pdfs")
    print(f"Loaded {len(documents)} documents.")
