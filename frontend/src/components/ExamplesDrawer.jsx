import { useState } from "react";

export default function ExamplesDrawer({ examples }) {
  const [open, setOpen] = useState(false);
  if (!examples || examples.length === 0) return null;

  return (
    <div>
      <button className="reveal-link" onClick={() => setOpen((v) => !v)}>
        {open ? "Hide reasoning" : "Why this reply?"}
      </button>
      {open && (
        <div className="examples-drawer">
          {examples.map((ex, i) => (
            <div className="example" key={i}>
              <div>
                <span className="score">match {(ex.score * 100).toFixed(0)}%</span>
              </div>
              <div>{ex.context}</div>
              <div style={{ color: "var(--text)" }}>→ {ex.reply}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
