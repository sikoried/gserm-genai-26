import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

const COLORS = ["#2563eb", "#16a34a", "#dc2626", "#d97706", "#7c3aed", "#0891b2"];

function shortName(model) {
  return model.split("/").pop();
}

// Distinguishes the same model added with different reasoning settings.
function entryLabel(r) {
  return shortName(r.model) + (r.reasoning_effort ? ` · ${r.reasoning_effort}` : "");
}

export default function ComparisonCharts({ results }) {
  // Only successful responses can be charted (failures have no metrics).
  const ok = results.filter((r) => !r.error);

  const timeData = ok.map((r, i) => ({
    name: entryLabel(r),
    value: r.elapsed_seconds,
    color: COLORS[i % COLORS.length],
  }));

  const tokenData = ok.map((r, i) => ({
    name: entryLabel(r),
    prompt: r.prompt_tokens,
    reasoning: r.reasoning_tokens || 0,
    answer: Math.max(0, r.completion_tokens - (r.reasoning_tokens || 0)),
    color: COLORS[i % COLORS.length],
  }));

  return (
    <div style={styles.wrapper}>
      <h2 style={styles.heading}>Cost Comparison</h2>
      {ok.length === 0 ? (
        <p style={{ color: "#888", fontSize: "0.9rem" }}>No successful responses to chart.</p>
      ) : (
      <div style={styles.grid}>
        {/* Response time chart */}
        <div style={styles.chartBox}>
          <h3 style={styles.chartTitle}>Response Time (seconds)</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={timeData} margin={{ top: 8, right: 16, left: 0, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="name"
                tick={{ fontSize: 12 }}
                angle={-30}
                textAnchor="end"
                interval={0}
              />
              <YAxis tick={{ fontSize: 12 }} unit="s" />
              <Tooltip formatter={(v) => [`${v.toFixed(2)}s`, "time"]} />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {timeData.map((entry, i) => (
                  <Cell key={i} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Token cost chart */}
        <div style={styles.chartBox}>
          <h3 style={styles.chartTitle}>Token Cost</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={tokenData} margin={{ top: 8, right: 16, left: 0, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="name"
                tick={{ fontSize: 12 }}
                angle={-30}
                textAnchor="end"
                interval={0}
              />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip
                formatter={(v, name) => [v.toLocaleString(), { prompt: "Prompt", reasoning: "Reasoning", answer: "Answer" }[name] || name]}
              />
              <Bar dataKey="prompt" stackId="a" fill="#93c5fd" name="prompt" radius={[0, 0, 0, 0]} />
              <Bar dataKey="reasoning" stackId="a" fill="#f59e0b" name="reasoning" radius={[0, 0, 0, 0]} />
              <Bar dataKey="answer" stackId="a" fill="#2563eb" name="answer" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <div style={styles.legend}>
            <span style={{ ...styles.dot, background: "#93c5fd" }} /> Prompt
            <span style={{ ...styles.dot, background: "#f59e0b", marginLeft: "1rem" }} /> Reasoning
            <span style={{ ...styles.dot, background: "#2563eb", marginLeft: "1rem" }} /> Answer
          </div>
        </div>
      </div>
      )}

      {/* Summary table */}
      <table style={styles.table}>
        <thead>
          <tr>
            <th style={styles.th}>Model</th>
            <th style={styles.th}>Time (s)</th>
            <th style={styles.th}>Prompt tokens</th>
            <th style={styles.th}>Completion tokens</th>
            <th style={styles.th}>Reasoning tokens</th>
            <th style={styles.th}>Total tokens</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r, i) =>
            r.error ? (
              <tr key={i} style={{ background: i % 2 === 0 ? "#f9fafb" : "#fff" }}>
                <td style={styles.td}>{entryLabel(r)}</td>
                <td style={{ ...styles.td, color: "#b91c1c" }} colSpan={5}>
                  error: {r.error}
                </td>
              </tr>
            ) : (
              <tr key={i} style={{ background: i % 2 === 0 ? "#f9fafb" : "#fff" }}>
                <td style={styles.td}>{entryLabel(r)}</td>
                <td style={{ ...styles.td, textAlign: "right" }}>{r.elapsed_seconds.toFixed(2)}</td>
                <td style={{ ...styles.td, textAlign: "right" }}>{r.prompt_tokens.toLocaleString()}</td>
                <td style={{ ...styles.td, textAlign: "right" }}>{r.completion_tokens.toLocaleString()}</td>
                <td style={{ ...styles.td, textAlign: "right" }}>{(r.reasoning_tokens || 0).toLocaleString()}</td>
                <td style={{ ...styles.td, textAlign: "right", fontWeight: 600 }}>{r.total_tokens.toLocaleString()}</td>
              </tr>
            )
          )}
        </tbody>
      </table>
    </div>
  );
}

const styles = {
  wrapper: {
    background: "#fff",
    borderRadius: "8px",
    padding: "1.5rem",
    marginTop: "1.5rem",
    boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
  },
  heading: {
    marginBottom: "1rem",
    fontSize: "1.15rem",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "1.5rem",
    marginBottom: "1.5rem",
  },
  chartBox: {
    minWidth: 0,
  },
  chartTitle: {
    fontSize: "0.9rem",
    fontWeight: 600,
    color: "#555",
    marginBottom: "0.5rem",
  },
  legend: {
    fontSize: "0.8rem",
    color: "#555",
    marginTop: "0.25rem",
    display: "flex",
    alignItems: "center",
  },
  dot: {
    display: "inline-block",
    width: 10,
    height: 10,
    borderRadius: "50%",
    marginRight: "0.3rem",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    fontSize: "0.875rem",
  },
  th: {
    background: "#f3f4f6",
    padding: "0.5rem 0.75rem",
    textAlign: "left",
    fontWeight: 600,
    borderBottom: "1px solid #e5e7eb",
  },
  td: {
    padding: "0.5rem 0.75rem",
    borderBottom: "1px solid #e5e7eb",
  },
};
