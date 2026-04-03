import { useRef, useState } from "react";
import { uploadDocuments } from "../api";

interface Props {
  sessionId: string;
  fileCount: number;
  onUploaded: (newFileCount: number) => void;
}

export default function FileUpload({
  sessionId,
  fileCount,
  onUploaded,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [queued, setQueued] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [lastResult, setLastResult] = useState<{
    files: number;
    chunks: number;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  function handleFilePick(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    setQueued((prev) => [...prev, ...files]);
    setLastResult(null);
    setError(null);
    // Reset so the same file can be re-selected
    e.target.value = "";
  }

  function removeQueued(index: number) {
    setQueued((prev) => prev.filter((_, i) => i !== index));
  }

  async function handleUpload() {
    if (!queued.length || uploading) return;
    setUploading(true);
    setError(null);
    setLastResult(null);
    try {
      const result = await uploadDocuments(sessionId, queued);
      setLastResult({
        files: result.files_processed,
        chunks: result.chunks_stored,
      });
      onUploaded(fileCount + result.files_processed);
      setQueued([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-base font-semibold text-gray-800">
        Upload Documents
      </h2>

      {/* Drop zone / file picker */}
      <div
        onClick={() => inputRef.current?.click()}
        className="border-2 border-dashed border-gray-300 hover:border-indigo-400 rounded-xl p-6 text-center cursor-pointer transition-colors"
      >
        <p className="text-sm text-gray-500">
          Click to select{" "}
          <span className="font-medium text-indigo-600">
            PDF, DOCX, TXT, or MD files
          </span>
        </p>
        <p className="text-xs text-gray-400 mt-1">Multiple files supported</p>
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx,.txt,.md"
          multiple
          className="hidden"
          onChange={handleFilePick}
        />
      </div>

      {/* Queued file list */}
      {queued.length > 0 && (
        <ul className="space-y-1">
          {queued.map((f, i) => (
            <li
              key={i}
              className="flex items-center justify-between bg-gray-50 rounded-lg px-3 py-1.5 text-sm text-gray-700"
            >
              <span className="truncate max-w-[160px]" title={f.name}>
                {f.name}
              </span>
              <button
                onClick={() => removeQueued(i)}
                className="text-gray-400 hover:text-red-500 ml-2 text-xs"
              >
                ✕
              </button>
            </li>
          ))}
        </ul>
      )}

      {/* Upload button */}
      <button
        onClick={handleUpload}
        disabled={!queued.length || uploading}
        className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-xl px-4 py-2 text-sm font-medium transition-colors flex items-center justify-center gap-2"
      >
        {uploading ? (
          <>
            <svg
              className="animate-spin h-4 w-4 text-white"
              viewBox="0 0 24 24"
              fill="none"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8v8H4z"
              />
            </svg>
            Processing…
          </>
        ) : (
          "Upload"
        )}
      </button>

      {/* Feedback */}
      {lastResult && (
        <p className="text-xs text-green-600">
          ✓ {lastResult.files} file{lastResult.files !== 1 ? "s" : ""} ingested
          — {lastResult.chunks} chunk{lastResult.chunks !== 1 ? "s" : ""}{" "}
          indexed
        </p>
      )}
      {error && <p className="text-xs text-red-500">⚠ {error}</p>}

      {/* Session summary */}
      {fileCount > 0 && (
        <p className="text-xs text-gray-400 mt-1">
          {fileCount} file{fileCount !== 1 ? "s" : ""} in this session
        </p>
      )}
    </div>
  );
}
