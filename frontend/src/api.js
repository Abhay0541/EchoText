const BASE = "/api";

export async function uploadChat(file, targetUser) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/sessions/upload?target_user=${encodeURIComponent(targetUser)}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || "Upload failed");
  }
  return res.json();
}

export async function sendMessage(sessionId, incomingMessage) {
  const res = await fetch(`${BASE}/sessions/${sessionId}/reply`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ incoming_message: incomingMessage }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || "Request failed");
  }
  return res.json();
}
