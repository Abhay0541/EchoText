const ROUTE_META = {
  quick_reply: { label: "Auto-replied", color: "var(--safe)", soft: "var(--safe-soft)" },
  needs_context: { label: "Needs context", color: "var(--warn)", soft: "var(--warn-soft)" },
  sensitive: { label: "Held back — sensitive", color: "var(--danger)", soft: "var(--danger-soft)" },
  unknown: { label: "Held back — unsure", color: "var(--text-faint)", soft: "var(--surface-raised)" },
};

export default function RouteBadge({ route }) {
  const meta = ROUTE_META[route] || ROUTE_META.unknown;
  return (
    <span className="route-badge" style={{ background: meta.soft, color: meta.color }}>
      <span className="dot" style={{ background: meta.color }} />
      {meta.label}
    </span>
  );
}
