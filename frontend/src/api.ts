const BASE = "http://localhost:8000";

export interface SessionResponse {
  session_id: string;
}

export interface StatusResponse {
  has_documents: boolean;
  file_count: number;
}

export interface UploadResponse {
  files_processed: number;
  chunks_stored: number;
}

export interface ChatResponse {
  answer: string;
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail?.detail ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

export async function createSession(): Promise<SessionResponse> {
  const res = await fetch(`${BASE}/api/sessions`, { method: "POST" });
  return handleResponse<SessionResponse>(res);
}

export async function getStatus(sessionId: string): Promise<StatusResponse> {
  const res = await fetch(`${BASE}/api/sessions/${sessionId}/status`);
  return handleResponse<StatusResponse>(res);
}

export async function uploadDocuments(
  sessionId: string,
  files: File[],
): Promise<UploadResponse> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  const res = await fetch(`${BASE}/api/sessions/${sessionId}/upload`, {
    method: "POST",
    body: form,
  });
  return handleResponse<UploadResponse>(res);
}

export async function sendChat(
  sessionId: string,
  query: string,
): Promise<ChatResponse> {
  const res = await fetch(`${BASE}/api/sessions/${sessionId}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  return handleResponse<ChatResponse>(res);
}

export async function deleteSession(sessionId: string): Promise<void> {
  await fetch(`${BASE}/api/sessions/${sessionId}`, { method: "DELETE" });
}
