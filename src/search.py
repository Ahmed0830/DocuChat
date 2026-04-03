from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.llm import get_llm
from src.prompt import _SYSTEM_PROMPT
from src.vectorstore import QdrantStore


class RAGSearch:
    def __init__(
        self,
        location: str = "./qdrant_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        vectorstore: QdrantStore | None = None,
    ):
        if vectorstore is not None:
            self.vectorstore = vectorstore
        else:
            self.vectorstore = QdrantStore(
                location=location, embedding_model=embedding_model
            )
            self.vectorstore.load()

        self.llm = get_llm()

        self._sessions: dict[str, InMemoryChatMessageHistory] = {}

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                MessagesPlaceholder("history"),
                ("human", "{query}"),
            ]
        )

        # LCEL chain: prompt → LLM → plain string
        self._chain = RunnableWithMessageHistory(
            prompt | self.llm | StrOutputParser(),
            self._get_session_history,
            input_messages_key="query",
            history_messages_key="history",
        )

    def _get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self._sessions:
            self._sessions[session_id] = InMemoryChatMessageHistory()
        return self._sessions[session_id]

    def search(self, query: str, k: int = 3, session_id: str = "default") -> str:
        docs = self.vectorstore.similarity_search(query, k=k)
        context = "\n\n".join(doc.page_content for doc in docs)
        return self._chain.invoke(
            {"query": query, "context": context},
            config={"configurable": {"session_id": session_id}},
        )

    def clear_history(self, session_id: str = "default") -> None:
        """Wipe the conversation history for a given session."""
        self._sessions.pop(session_id, None)
