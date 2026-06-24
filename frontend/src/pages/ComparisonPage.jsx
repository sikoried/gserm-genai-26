import { useEffect, useState } from "react";
import ComparisonCharts from "../components/ComparisonCharts.jsx";
import ModelCard from "../components/ModelCard.jsx";

const DEFAULT_ENDPOINT = "https://kiz1.in.ohmportal.de/llmproxy/v1";

function shortName(model) {
  return model.split("/").pop();
}

export default function ComparisonPage() {
  const [question, setQuestion] = useState("");
  const [endpoint, setEndpoint] = useState(DEFAULT_ENDPOINT);
  const [models, setModels] = useState([]);
  // The comparison list: each entry is { model, reasoning_effort }.
  const [entries, setEntries] = useState([]);
  const [pickModel, setPickModel] = useState("");
  const [pickReasoning, setPickReasoning] = useState("off");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Load the available models (and their reasoning options) from the backend.
  useEffect(() => {
    fetch("/api/models")
      .then((r) => r.json())
      .then((list) => {
        setModels(list);
        if (list.length) setPickModel(list[0].id);
      })
      .catch(() => setError("Could not load the model list from /api/models."));
  }, []);

  const selectedModel = models.find((m) => m.id === pickModel);
  const efforts = selectedModel?.efforts || [];

  function changeModel(id) {
    setPickModel(id);
    setPickReasoning("off"); // reasoning options are model-specific
  }

  function addEntry() {
    if (!pickModel) return;
    const reasoning_effort = pickReasoning === "off" ? null : pickReasoning;
    // Skip exact duplicates (same model + same reasoning setting).
    if (entries.some((e) => e.model === pickModel && e.reasoning_effort === reasoning_effort)) {
      return;
    }
    setEntries((prev) => [...prev, { model: pickModel, reasoning_effort }]);
  }

  function removeEntry(i) {
    setEntries((prev) => prev.filter((_, idx) => idx !== i));
  }

  async function handleCompare() {
    if (!question.trim()) return;
    if (entries.length === 0) {
      setError("Add at least one model to the comparison.");
      return;
    }
    setLoading(true);
    setError("");
    setResults([]);
    try {
      const resp = await fetch("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.trim(), entries, endpoint }),
      });
      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({}));
        throw new Error(detail.detail || `HTTP ${resp.status}`);
      }
      setResults(await resp.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={styles.scroll}>
      <div style={styles.container}>
        <h1 style={{ marginBottom: "0.25rem" }}>Model Comparison</h1>
        <p style={{ color: "#666", marginBottom: "1.5rem" }}>
          Build a comparison list — add the same or different models, with or without
          reasoning — and compare response time and token cost.
        </p>

        {/* Question + endpoint */}
        <section style={styles.card}>
          <label style={styles.label}>Question</label>
          <textarea
            style={styles.textarea}
            rows={3}
            placeholder="Enter your question…"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />

          <label style={styles.label}>Endpoint</label>
          <input
            style={styles.input}
            value={endpoint}
            onChange={(e) => setEndpoint(e.target.value)}
          />
        </section>

        {/* Comparison list builder */}
        <section style={styles.card}>
          <label style={styles.label}>Add model to comparison</label>
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", alignItems: "center" }}>
            <select
              style={{ ...styles.input, flex: "2 1 240px", width: "auto" }}
              value={pickModel}
              onChange={(e) => changeModel(e.target.value)}
            >
              {models.length === 0 && <option value="">Loading models…</option>}
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
            <select
              style={{ ...styles.input, flex: "1 1 150px", width: "auto" }}
              value={pickReasoning}
              onChange={(e) => setPickReasoning(e.target.value)}
              disabled={efforts.length === 0}
              title={efforts.length === 0 ? "This model has no reasoning control" : undefined}
            >
              <option value="off">No reasoning</option>
              {efforts.map((eff) => (
                <option key={eff} value={eff}>
                  Reasoning: {eff}
                </option>
              ))}
            </select>
            <button style={styles.btnSecondary} onClick={addEntry}>
              Add
            </button>
          </div>

          {/* Comparison list as pills */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", marginTop: "0.85rem" }}>
            {entries.length === 0 && (
              <span style={{ color: "#888", fontSize: "0.85rem" }}>No models added yet.</span>
            )}
            {entries.map((e, i) => (
              <span key={i} style={styles.pill}>
                <span>{shortName(e.model)}</span>
                {e.reasoning_effort && (
                  <span style={styles.pillReasoning}>🧠 {e.reasoning_effort}</span>
                )}
                <button
                  style={styles.pillRemove}
                  onClick={() => removeEntry(i)}
                  aria-label={`remove ${e.model}`}
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        </section>

        <button
          style={{ ...styles.btnPrimary, opacity: loading ? 0.6 : 1 }}
          onClick={handleCompare}
          disabled={loading}
        >
          {loading ? "Comparing…" : "Compare"}
        </button>

        {error && <div style={styles.error}>{error}</div>}

        {results.length > 0 && (
          <>
            <ComparisonCharts results={results} />

            <h2 style={{ marginTop: "2rem", marginBottom: "1rem" }}>Answers</h2>
            <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
              {results.map((r, i) => (
                <ModelCard key={i} result={r} />
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

const styles = {
  scroll: { height: "100%", overflowY: "auto", background: "var(--hsg-gray-50)" },
  container: { maxWidth: "1100px", margin: "0 auto", padding: "2rem 1rem" },
  card: {
    background: "#fff",
    borderRadius: "8px",
    padding: "1.25rem",
    marginBottom: "1rem",
    boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
  },
  label: {
    display: "block",
    fontWeight: 600,
    fontSize: "0.875rem",
    marginBottom: "0.4rem",
    marginTop: "0.75rem",
  },
  input: {
    width: "100%",
    padding: "0.5rem 0.75rem",
    border: "1px solid var(--hsg-gray-300)",
    borderRadius: "6px",
    fontSize: "0.9rem",
    boxSizing: "border-box",
  },
  textarea: {
    width: "100%",
    padding: "0.5rem 0.75rem",
    border: "1px solid var(--hsg-gray-300)",
    borderRadius: "6px",
    fontSize: "0.9rem",
    resize: "vertical",
    fontFamily: "inherit",
    boxSizing: "border-box",
  },
  btnPrimary: {
    background: "var(--hsg-green)",
    color: "#fff",
    border: "none",
    borderRadius: "6px",
    padding: "0.6rem 1.5rem",
    fontSize: "1rem",
    fontWeight: 600,
    cursor: "pointer",
    marginBottom: "1rem",
  },
  btnSecondary: {
    background: "var(--hsg-gray-200)",
    color: "var(--hsg-gray-900)",
    border: "none",
    borderRadius: "6px",
    padding: "0.5rem 1rem",
    fontSize: "0.9rem",
    cursor: "pointer",
    whiteSpace: "nowrap",
  },
  error: {
    background: "#fee2e2",
    color: "#b91c1c",
    borderRadius: "6px",
    padding: "0.75rem 1rem",
    marginBottom: "1rem",
    fontSize: "0.9rem",
  },
  pill: {
    display: "inline-flex",
    alignItems: "center",
    gap: "0.4rem",
    background: "#e8f1ec",
    color: "var(--hsg-green-dark)",
    border: "1px solid #bcdcca",
    borderRadius: "20px",
    padding: "0.3rem 0.4rem 0.3rem 0.8rem",
    fontSize: "0.85rem",
    fontWeight: 500,
  },
  pillReasoning: {
    background: "#fef3c7",
    color: "#b45309",
    borderRadius: "12px",
    padding: "0.05rem 0.45rem",
    fontSize: "0.75rem",
    fontWeight: 600,
  },
  pillRemove: {
    background: "transparent",
    border: "none",
    color: "var(--hsg-green)",
    cursor: "pointer",
    fontSize: "1.05rem",
    lineHeight: 1,
    padding: "0 0.2rem",
  },
};
