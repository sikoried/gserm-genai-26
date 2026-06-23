export default function ModelCard({ result }) {
  return (
    <div style={styles.card}>
      <div style={styles.header}>
        <span style={styles.model}>{result.model}</span>
        <div style={styles.badges}>
          <span style={styles.badge}>{result.elapsed_seconds.toFixed(2)}s</span>
          <span style={styles.badge}>{result.total_tokens.toLocaleString()} tokens</span>
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
  model: {
    fontWeight: 700,
    fontSize: "0.9rem",
    color: "#2563eb",
  },
  badges: {
    display: "flex",
    gap: "0.5rem",
  },
  badge: {
    background: "#e0f2fe",
    color: "#0369a1",
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
};
