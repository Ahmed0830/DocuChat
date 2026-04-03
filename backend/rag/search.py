from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from sentence_transformers import CrossEncoder

from .llm import get_llm
from .prompt import _SYSTEM_PROMPT
from .vectorstore import QdrantStore

# Multiply k by this factor when fetching candidates for re-ranking
_RERANK_CANDIDATE_MULTIPLIER = 3


class RAGSearch:
    def __init__(
        self,
        location: str = "./qdrant_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        vectorstore: QdrantStore | None = None,
        llm: BaseChatModel | None = None,
        use_reranker: bool = False,
    ):
        if vectorstore is not None:
            self.vectorstore = vectorstore
        else:
            self.vectorstore = QdrantStore(
                location=location, embedding_model=embedding_model
            )
            self.vectorstore.load()

        self.llm = llm if llm is not None else get_llm()
        self.use_reranker = use_reranker
        self._reranker = (
            CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            if use_reranker
            else None
        )

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

    def retrieve(self, query: str, k: int = 6) -> list[Document]:
        """Return the top-k most relevant chunks for *query*.

        When *use_reranker* is True, fetches k * _RERANK_CANDIDATE_MULTIPLIER
        candidates via ANN search and re-ranks them with a CrossEncoder before
        returning the top k.  When False, behaviour is identical to a plain
        similarity search.
        """
        if self.vectorstore.vectorstore is None:
            return []

        if self.use_reranker and self._reranker is not None:
            candidates = self.vectorstore.similarity_search(
                query, k=k * _RERANK_CANDIDATE_MULTIPLIER
            )
            pairs = [[query, doc.page_content] for doc in candidates]
            scores = self._reranker.predict(pairs)
            reranked = [
                doc
                for _, doc in sorted(
                    zip(scores, candidates), key=lambda x: x[0], reverse=True
                )
            ]
            return reranked[:k]

        return self.vectorstore.similarity_search(query, k=k)

    def search(self, query: str, k: int = 6, session_id: str = "default") -> str:
        if self.vectorstore.vectorstore is None:
            return "Please upload documents first."
        docs = self.retrieve(query, k=k)
        context = "\n\n".join(doc.page_content for doc in docs)
        return self._chain.invoke(
            {"query": query, "context": context},
            config={"configurable": {"session_id": session_id}},
        )

    def clear_history(self, session_id: str = "default") -> None:
        """Wipe the conversation history for a given session."""
        self._sessions.pop(session_id, None)
