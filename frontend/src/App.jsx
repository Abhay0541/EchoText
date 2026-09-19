import { useRef, useState, useEffect } from "react";
import Sidebar from "./components/Sidebar.jsx";
import MessageRow from "./components/MessageRow.jsx";
import { uploadChat, sendMessage } from "./api.js";

export default function App() {
  const [session, setSession] = useState(null); // { session_id, messages_parsed, training_pairs_built }
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [turns, setTurns] = useState([]); // [{ incoming, result }]
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const threadRef = useRef(null);

  useEffect(() => {
    threadRef.current?.scrollTo({ top: threadRef.current.scrollHeight, behavior: "smooth" });
  }, [turns]);

  const handleUpload = async (file, targetUser) => {
    setUploading(true);
    setUploadError(null);
    try {
      const data = await uploadChat(file, targetUser);
      setSession(data);
      setTurns([]);
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
    }
  };

  const handleSend = async () => {
    const message = draft.trim();
    if (!message || !session || sending) return;
    setSending(true);
    setDraft("");
    try {
      const result = await sendMessage(session.session_id, message);
      setTurns((prev) => [...prev, { incoming: message, result }]);
    } catch (err) {
      setTurns((prev) => [
        ...prev,
        {
          incoming: message,
          result: { route: "unknown", reply: null, retrieved_examples: [], reasoning: err.message },
        },
      ]);
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="app">
      <Sidebar session={session} onUpload={handleUpload} uploading={uploading} error={uploadError} />

      <main className="main">
        <div className="thread" ref={threadRef}>
          {!session && (
            <div className="empty-state">
              <span className="headline">Start by uploading a chat</span>
              Export any WhatsApp conversation as a .txt file (Chat → More →
              Export chat → Without media) and drop it in on the left.
            </div>
          )}
          {session && turns.length === 0 && (
            <div className="empty-state">
              <span className="headline">Ready</span>
              Type a message below as if someone just sent it to you, and
              see how EchoText would route and reply to it.
            </div>
          )}
          {turns.map((turn, i) => (
            <MessageRow key={i} turn={turn} />
          ))}
        </div>

        {session && (
          <>
            <div className="hint">Simulating an incoming message — this won't be sent anywhere.</div>
            <div className="composer">
              <input
                className="text-input"
                placeholder="Type a message someone might send you…"
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSend()}
              />
              <button className="primary-button" onClick={handleSend} disabled={sending || !draft.trim()}>
                {sending ? "Thinking…" : "Send"}
              </button>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
