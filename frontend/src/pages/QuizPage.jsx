import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import TraceView from "../components/TraceView.jsx";
import "./ChatPage.css"; // for TraceView (.trace-*) classes
import "./QuizPage.css";
import { subscribe, getSnapshot, setConfig, startQuiz, stopQuiz } from "./quizStore.js";

const MODES = ["World", "RAG", "Agentic RAG"];
const COUNT_OPTIONS = [5, 10, 20, 50, 100];

export default function QuizPage() {
  // The quiz config + run state live in a module store so a running quiz survives
  // switching GUI windows and its results are still here when you come back.
  const store = useSyncExternalStore(subscribe, getSnapshot);
  const { config: cfg, running, results, progress, error } = store;

  const [models, setModels] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [elapsed, setElapsed] = useState(0);
  const isAgentic = cfg.mode === "Agentic RAG";

  useEffect(() => {
    fetch("/api/models").then((r) => r.json()).then((list) => {
      setModels(list);
      if (list.length && !list.some((m) => m.id === cfg.model)) setConfig({ model: list[0].id });
    }).catch(() => {});
    fetch("/api/quiz/datasets").then((r) => r.json()).then((list) => {
      setDatasets(list);
      if (list.length && !list.some((d) => d.id === cfg.dataset)) setConfig({ dataset: list[0].id });
    }).catch(() => {});
  }, []);

  // Cosmetic elapsed timer for the in-flight question (resets each answer).
  useEffect(() => {
    if (!running) return;
    setElapsed(0);
    const started = Date.now();
    const id = setInterval(() => setElapsed((Date.now() - started) / 1000), 200);
    return () => clearInterval(id);
  }, [running, progress.done]);

  const upd = (patch) => setConfig(patch);

  // --- statistics -----------------------------------------------------------
  const answered = results.length;
  const correct = results.filter((r) => r.right).length;
  const wrong = answered - correct;
  const sum = (f) => results.reduce((a, r) => a + (f(r) || 0), 0);
  const totalTime = sum((r) => r.metrics.elapsed_seconds);
  const totalIn = sum((r) => r.metrics.input_tokens);
  const totalOut = sum((r) => r.metrics.output_tokens);
  const totalRsn = sum((r) => r.metrics.reasoning_tokens);
  const totalTok = totalIn + totalOut + totalRsn;
  const avg = (v) => (answered ? v / answered : 0);
  const done = !running && answered > 0 && answered === progress.total;

  return (
    <div className="quiz-scroll">
      <div className="quiz-container">
        <h1 style={{ marginBottom: "0.25rem" }}>Quiz</h1>
        <p style={{ color: "#666", marginBottom: "1.25rem" }}>
          The host asks questions from a dataset; your chosen player answers and the
          host marks each right or wrong. A running quiz keeps going if you switch tabs.
        </p>

        {/* Setup */}
        <section className="quiz-card">
          <div className="quiz-grid">
            <label>
              <span className="quiz-label">Dataset (host)</span>
              <select value={cfg.dataset} onChange={(e) => upd({ dataset: e.target.value })} disabled={running}>
                {datasets.length === 0 && <option value="millionaire">Millionaire</option>}
                {datasets.map((d) => <option key={d.id} value={d.id}>{d.label}</option>)}
              </select>
            </label>
            <label>
              <span className="quiz-label">Questions</span>
              <select value={cfg.count} onChange={(e) => upd({ count: e.target.value })} disabled={running}>
                {COUNT_OPTIONS.map((n) => <option key={n} value={String(n)}>{n}</option>)}
                <option value="all">All</option>
              </select>
            </label>
            <label>
              <span className="quiz-label">Player mode</span>
              <select value={cfg.mode} onChange={(e) => upd({ mode: e.target.value })} disabled={running}>
                {MODES.map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            </label>
            <label>
              <span className="quiz-label">Player model</span>
              <select value={cfg.model} onChange={(e) => upd({ model: e.target.value })} disabled={running}>
                {models.length === 0 && <option value="">Loading…</option>}
                {models.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
              </select>
            </label>
            <label>
              <span className="quiz-label">Temperature</span>
              <input type="number" step="0.1" min="0" max="2" value={cfg.temperature}
                     onChange={(e) => upd({ temperature: e.target.value })} disabled={running} />
            </label>
          </div>

          {isAgentic && (
            <div className="quiz-agentic">
              <div className="quiz-checks">
                <label className="quiz-check"><input type="checkbox" checked={cfg.enableOnline}
                  onChange={(e) => upd({ enableOnline: e.target.checked })} disabled={running} /> Web + YouTube tools</label>
                <label className="quiz-check"><input type="checkbox" checked={cfg.multiHop}
                  onChange={(e) => upd({ multiHop: e.target.checked })} disabled={running} /> Multi-hop</label>
                <label className="quiz-check"><input type="checkbox" checked={cfg.verify}
                  onChange={(e) => upd({ verify: e.target.checked })} disabled={running} /> Verify answers</label>
              </div>
              <div className="quiz-grid">
                <label><span className="quiz-label">Planner</span>
                  <select value={cfg.plannerModel} onChange={(e) => upd({ plannerModel: e.target.value })} disabled={running || !cfg.multiHop}>
                    <option value="router">Small (fast)</option>
                    <option value="answer">Big model (better)</option>
                  </select></label>
                <label><span className="quiz-label">Max steps</span>
                  <input type="number" min="1" max="20" value={cfg.maxSteps} onChange={(e) => upd({ maxSteps: e.target.value })} disabled={running} /></label>
                <label><span className="quiz-label">Max sub-questions</span>
                  <input type="number" min="1" max="8" value={cfg.maxHops} onChange={(e) => upd({ maxHops: e.target.value })} disabled={running} /></label>
                <label><span className="quiz-label">Max depth</span>
                  <input type="number" min="1" max="4" value={cfg.maxDepth} onChange={(e) => upd({ maxDepth: e.target.value })} disabled={running} /></label>
              </div>
            </div>
          )}

          <div style={{ marginTop: "1rem" }}>
            {!running ? (
              <button className="quiz-btn-primary" onClick={startQuiz}>Start quiz</button>
            ) : (
              <button className="quiz-btn-stop" onClick={stopQuiz}>Stop</button>
            )}
          </div>
          {error && <div className="quiz-error">{error}</div>}
        </section>

        {/* Progress + statistics */}
        {(running || answered > 0) && (
          <section className="quiz-card">
            <div className="quiz-progress-row">
              <span>{done ? "Finished" : running ? "Running…" : "Stopped"} — {progress.done}/{progress.total} answered</span>
              {running && <span className="quiz-elapsed">{elapsed.toFixed(1)}s on this question</span>}
            </div>
            {progress.total > 0 && (
              <div className="quiz-progress-track">
                <div className="quiz-progress-bar" style={{ width: `${(progress.done / progress.total) * 100}%` }} />
              </div>
            )}
            <div className="quiz-stats">
              <Stat label="Correct" value={correct} tone="good" />
              <Stat label="Wrong" value={wrong} tone="bad" />
              <Stat label="Accuracy" value={answered ? `${Math.round((correct / answered) * 100)}%` : "—"} />
              <Stat label="Total time" value={`${totalTime.toFixed(1)}s`} />
              <Stat label="Avg / question" value={`${avg(totalTime).toFixed(1)}s`} />
              <Stat label="Tokens (in/out/rsn)" value={`${totalIn.toLocaleString()}/${totalOut.toLocaleString()}/${totalRsn.toLocaleString()}`} />
              <Stat label="Total tokens" value={totalTok.toLocaleString()} />
              <Stat label="Avg tokens / q" value={Math.round(avg(totalTok)).toLocaleString()} />
            </div>
          </section>
        )}

        {/* Per-question results */}
        {results.map((r, i) => (
          <section key={i} className={`quiz-card quiz-q ${r.right ? "q-right" : "q-wrong"}`}>
            <div className="quiz-q-head">
              <span className="quiz-q-num">Q{i + 1}</span>
              <span className={`quiz-verdict ${r.right ? "v-right" : "v-wrong"}`}>
                {r.verdict === "error" ? "error" : r.right ? "✓ right" : "✗ wrong"}
              </span>
              <span className="quiz-q-meta">
                {r.metrics.elapsed_seconds.toFixed(1)}s · {r.metrics.total_tokens.toLocaleString()} tok
              </span>
            </div>
            <div className="quiz-q-question"><b>Q:</b> {r.question}</div>
            <div className="quiz-q-answer"><b>Player:</b> {r.answer || <i>(no answer)</i>}</div>
            <div className="quiz-q-ref"><b>Reference:</b> {r.reference}</div>
            {r.judge_reasoning && <div className="quiz-q-judge">Host: {r.judge_reasoning}</div>}
            {r.trace && <TraceView trace={r.trace} />}
          </section>
        ))}
      </div>
    </div>
  );
}

function Stat({ label, value, tone }) {
  return (
    <div className="quiz-stat">
      <div className={`quiz-stat-value ${tone ? `tone-${tone}` : ""}`}>{value}</div>
      <div className="quiz-stat-label">{label}</div>
    </div>
  );
}
