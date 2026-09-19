import { useRef, useState } from "react";

export default function Sidebar({ session, onUpload, uploading, error }) {
  const [targetUser, setTargetUser] = useState("");
  const [fileName, setFileName] = useState("");
  const fileRef = useRef(null);

  const handleFile = (e) => {
    const file = e.target.files?.[0];
    if (file) setFileName(file.name);
  };

  const handleSubmit = () => {
    const file = fileRef.current?.files?.[0];
    if (!file || !targetUser.trim()) return;
    onUpload(file, targetUser.trim());
  };

  return (
    <aside className="sidebar">
      <div>
        <div className="wordmark">
          Echo<span>Text</span>
        </div>
        <p className="tagline">Learns how you text, then replies in your own voice.</p>
      </div>

      <div>
        <label className="field-label" htmlFor="target-user">
          Your name, exactly as it appears in the export
        </label>
        <input
          id="target-user"
          className="text-input"
          placeholder="e.g. Abhay"
          value={targetUser}
          onChange={(e) => setTargetUser(e.target.value)}
          style={{ marginBottom: 12 }}
        />

        <label className="field-label" htmlFor="chat-file">
          WhatsApp chat export (.txt)
        </label>
        <button
          type="button"
          className="file-drop"
          onClick={() => fileRef.current?.click()}
        >
          {fileName || "Choose a file…"}
        </button>
        <input
          ref={fileRef}
          id="chat-file"
          type="file"
          accept=".txt"
          onChange={handleFile}
          style={{ display: "none" }}
        />

        <div style={{ marginTop: 12 }}>
          <button
            className="primary-button"
            disabled={uploading || !fileName || !targetUser.trim()}
            onClick={handleSubmit}
          >
            {uploading ? "Building your index…" : "Build my reply model"}
          </button>
        </div>

        {error && <div className="error-banner" style={{ marginTop: 12 }}>{error}</div>}
      </div>

      {session && (
        <div className="session-stats">
          <div className="stat-row">
            <span>Messages parsed</span>
            <b>{session.messages_parsed}</b>
          </div>
          <div className="stat-row">
            <span>Reply examples indexed</span>
            <b>{session.training_pairs_built}</b>
          </div>
        </div>
      )}

      <div className="route-legend">
        <div className="legend-heading">How routing works</div>
        <div className="legend-item">
          <span className="legend-dot" style={{ background: "var(--safe)" }} />
          Auto-replied — a close match was found in your history.
        </div>
        <div className="legend-item">
          <span className="legend-dot" style={{ background: "var(--warn)" }} />
          Needs context — no strong match; flagged for you.
        </div>
        <div className="legend-item">
          <span className="legend-dot" style={{ background: "var(--danger)" }} />
          Sensitive topic — always held back, never auto-sent.
        </div>
      </div>
    </aside>
  );
}
