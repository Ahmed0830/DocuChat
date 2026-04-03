import logging

from dotenv import load_dotenv

from src.search import RAGSearch
from src.vectorstore import QdrantStore


def main() -> None:
    load_dotenv()
    # Suppress library noise so only the conversation is visible
    logging.disable(logging.CRITICAL)

    store = QdrantStore(collection_name="rag_documents", location="./qdrant_store")
    store.load()
    rag = RAGSearch(vectorstore=store)

    print("RAG Chatbot  (type 'exit' or 'quit' to stop)\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        response = rag.search(query)
        print(f"\nAssistant: {response}\n")

    store.client.close()


if __name__ == "__main__":
    main()
