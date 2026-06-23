import { useState } from "react";
import ComparisonCharts from "./components/ComparisonCharts.jsx";
import ModelCard from "./components/ModelCard.jsx";

const DEFAULT_MODELS = [
  "mistralai/Mistral-Medium-3.5-128B",
  "mistralai/Mistral-Small-3.1-24B",
  "mistralai/Mistral-7B-Instruct-v0.3",
];

const DEFAULT_ENDPOINT = "https://kiz1.in.ohmportal.de/llmproxy/v1";

export default function App() {
  const [question, setQuestion] = useState("");
  const [endpoint, setEndpoint] = useState(DEFAULT_ENDPOINT);
  const [selectedModels, setSelectedModels] = useState([DEFAULT_MODELS[0]]);
  const [customModel, setCustomModel] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function toggleModel(model) {
    setSelectedModels((prev) =>
      prev.includes(model) ? prev.filter((m) => m !== model) : [...prev, model]
    );
  }

  function addCustomModel() {
    const m = customModel.trim();
    if (m && !selectedModels.includes(m)) {
      setSelectedModels((prev) => [...prev, m]);
    }
    setCustomModel("");
  }

  async function handleCompare() {
    if (!question.trim()) return;
    if (selectedModels.length === 0) {
      setError("Select at least one model.");
      return;
    }
    setLoading(true);
    setError("");
    setResults([]);
    try {
      const resp = await fetch("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: question.trim(),
          models: selectedModels,
          endpoint,
        }),
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
    <div>
      <h1 style={{ marginBottom: "0.25rem" }}>Oracle — Model Comparison</h1>
      <p style={{ color: "#666", marginBottom: "1.5rem" }}>
        Ask the same question across multiple models and compare response time and token cost.
      </p>

      {/* Question input */}
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

      {/* Model selection */}
      <section style={styles.card}>
        <label style={styles.label}>Models</label>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", marginBottom: "0.75rem" }}>
          {DEFAULT_MODELS.map((m) => (
            <button
              key={m}
              onClick={() => toggleModel(m)}
              style={{
                ...styles.chip,
                background: selectedModels.includes(m) ? "#2563eb" : "#e5e7eb",
                color: selectedModels.includes(m) ? "#fff" : "#222",
              }}
            >
              {m.split("/").pop()}
            </button>
          ))}
        </div>
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <input
            style={{ ...styles.input, flex: 1 }}
            placeholder="Add custom model ID…"
            value={customModel}
            onChange={(e) => setCustomModel(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && addCustomModel()}
          />
          <button style={styles.btnSecondary} onClick={addCustomModel}>
            Add
          </button>
        </div>
        {selectedModels.length > 0 && (
          <div style={{ marginTop: "0.5rem", fontSize: "0.85rem", color: "#555" }}>
            Selected: {selectedModels.join(", ")}
          </div>
        )}
      </section>

      <button
        style={{ ...styles.btnPrimary, opacity: loading ? 0.6 : 1 }}
        onClick={handleCompare}
        disabled={loading}
      >
        {loading ? "Comparing…" : "Compare Models"}
      </button>

      {error && (
        <div style={styles.error}>{error}</div>
      )}

      {results.length > 0 && (
        <>
          <ComparisonCharts results={results} />

          <h2 style={{ marginTop: "2rem", marginBottom: "1rem" }}>Answers</h2>
          <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
            {results.map((r) => (
              <ModelCard key={r.model} result={r} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}

const styles = {
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
    border: "1px solid #d1d5db",
    borderRadius: "6px",
    fontSize: "0.9rem",
  },
  textarea: {
    width: "100%",
    padding: "0.5rem 0.75rem",
    border: "1px solid #d1d5db",
    borderRadius: "6px",
    fontSize: "0.9rem",
    resize: "vertical",
    fontFamily: "inherit",
  },
  chip: {
    border: "none",
    borderRadius: "20px",
    padding: "0.3rem 0.8rem",
    cursor: "pointer",
    fontSize: "0.85rem",
    fontWeight: 500,
    transition: "background 0.15s",
  },
  btnPrimary: {
    background: "#2563eb",
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
    background: "#e5e7eb",
    color: "#222",
    border: "none",
    borderRadius: "6px",
    padding: "0.5rem 1rem",
    fontSize: "0.9rem",
    cursor: "pointer",
  },
  error: {
    background: "#fee2e2",
    color: "#b91c1c",
    borderRadius: "6px",
    padding: "0.75rem 1rem",
    marginBottom: "1rem",
    fontSize: "0.9rem",
  },
};
