import { useEffect, useState } from "react";
import { createSession, deleteSession } from "./api";
import ChatInterface from "./components/ChatInterface";
import FileUpload from "./components/FileUpload";

export default function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [fileCount, setFileCount] = useState(0);
  const [starting, setStarting] = useState(true);
  const [clearing, setClearing] = useState(false);

  useEffect(() => {
    createSession()
      .then((r) => setSessionId(r.session_id))
      .finally(() => setStarting(false));
  }, []);

  async function handleClear() {
    if (!sessionId || clearing) return;
    setClearing(true);
    try {
      await deleteSession(sessionId);
    } catch {
      // session may already be gone — that's fine
    }
    const { session_id } = await createSession();
    setSessionId(session_id);
    setFileCount(0);
    setClearing(false);
  }

  if (starting) {
    return (
      <div className="flex items-center justify-center min-h-screen text-gray-400 text-sm">
        Starting session…
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between">
        <h1 className="text-lg font-semibold text-gray-900">RAG Chatbot</h1>
        <button
          onClick={handleClear}
          disabled={clearing}
          className="text-sm text-red-500 hover:text-red-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {clearing ? "Clearing…" : "Clear Session"}
        </button>
      </header>

      {/* Main layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left panel — upload */}
        <aside className="w-72 shrink-0 border-r border-gray-200 bg-white p-5 overflow-y-auto">
          {sessionId && (
            <FileUpload
              key={sessionId}
              sessionId={sessionId}
              fileCount={fileCount}
              onUploaded={(count) => setFileCount(count)}
            />
          )}
        </aside>

        {/* Right panel — chat */}
        <main className="flex-1 overflow-hidden flex flex-col">
          {sessionId && (
            <ChatInterface
              key={sessionId}
              sessionId={sessionId}
              hasDocuments={fileCount > 0}
            />
          )}
        </main>
      </div>
    </div>
  );
}
