const MODE_COLORS = {
  "World":       { background: "#e0f2fe", color: "#0369a1" },
  "RAG":         { background: "#dcfce7", color: "#15803d" },
  "Agentic RAG": { background: "#fef3c7", color: "#b45309" },
};

function ModeBadge({ mode }) {
  if (!mode) return null;
  const { background, color } = MODE_COLORS[mode] || { background: "#f3f4f6", color: "#555" };
  return (
    <span style={{ background, color, borderRadius: "12px", padding: "0.1rem 0.5rem",
                   fontSize: "0.75rem", fontWeight: 600 }}>
      {mode}
    </span>
  );
}

export default function ModelCard({ result }) {
  const reasoningTag = result.reasoning_effort ? (
    <span style={styles.reasoningTag}>reasoning: {result.reasoning_effort}</span>
  ) : null;

  if (result.error) {
    return (
      <div style={{ ...styles.card, ...styles.errorCard }}>
        <div style={styles.header}>
          <span style={styles.titleWrap}>
            <span style={styles.model}>{result.model}</span>
            <ModeBadge mode={result.mode} />
            {reasoningTag}
          </span>
          <span style={styles.errorBadge}>failed</span>
        </div>
        <p style={styles.errorMsg}>{result.error}</p>
      </div>
    );
  }

  return (
    <div style={styles.card}>
      <div style={styles.header}>
        <span style={styles.titleWrap}>
          <span style={styles.model}>{result.model}</span>
          <ModeBadge mode={result.mode} />
          {reasoningTag}
        </span>
        <div style={styles.badges}>
          <span style={styles.badge}>{result.elapsed_seconds.toFixed(2)}s</span>
          <span style={styles.badge}>{result.total_tokens.toLocaleString()} tokens</span>
          {result.reasoning_tokens > 0 && (
            <span style={styles.reasoningBadge}>{result.reasoning_tokens.toLocaleString()} reasoning</span>
          )}
        </div>
      </div>
      <p style={styles.answer}>{result.answer}</p>
    </div>
  );
}

const styles = {
  card: {
    background: "#fff",
    borderRadius: "8px",
    padding: "1.25rem",
    boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: "0.75rem",
    flexWrap: "wrap",
    gap: "0.5rem",
  },
  titleWrap: {
    display: "inline-flex",
    alignItems: "center",
    gap: "0.5rem",
    flexWrap: "wrap",
  },
  model: {
    fontWeight: 700,
    fontSize: "0.9rem",
    color: "var(--hsg-green)",
  },
  reasoningTag: {
    background: "#fef3c7",
    color: "#b45309",
    borderRadius: "12px",
    padding: "0.1rem 0.5rem",
    fontSize: "0.75rem",
    fontWeight: 600,
  },
  badges: {
    display: "flex",
    gap: "0.5rem",
    flexWrap: "wrap",
  },
  badge: {
    background: "#e8f1ec",
    color: "var(--hsg-green-dark)",
    borderRadius: "12px",
    padding: "0.15rem 0.6rem",
    fontSize: "0.8rem",
    fontWeight: 600,
  },
  reasoningBadge: {
    background: "#fef3c7",
    color: "#b45309",
    borderRadius: "12px",
    padding: "0.15rem 0.6rem",
    fontSize: "0.8rem",
    fontWeight: 600,
  },
  answer: {
    lineHeight: 1.6,
    whiteSpace: "pre-wrap",
    fontSize: "0.9rem",
  },
  errorCard: {
    border: "1px solid #fecaca",
    background: "#fef2f2",
  },
  errorBadge: {
    background: "#fee2e2",
    color: "#b91c1c",
    borderRadius: "12px",
    padding: "0.15rem 0.6rem",
    fontSize: "0.8rem",
    fontWeight: 600,
  },
  errorMsg: {
    color: "#b91c1c",
    fontSize: "0.85rem",
    whiteSpace: "pre-wrap",
    margin: 0,
  },
};
