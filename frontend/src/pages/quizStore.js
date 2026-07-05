// Module-level quiz store: holds the quiz config, run state, and results, and runs
// the quiz loop OUTSIDE any React component. This way a running quiz keeps going
// when the user switches GUI windows (Chat/Comparison), and the results are still
// there when they return to the Quiz page. State survives route changes (it does
// not survive a full page reload).

const DEFAULT_CONFIG = {
  dataset: "millionaire",
  count: "10", // "all" or a number-string
  mode: "Agentic RAG",
  model: "",
  temperature: "0.0",
  enableOnline: true,
  multiHop: true,
  plannerModel: "router",
  maxHops: 3,
  maxSteps: 6,
  maxDepth: 1,
  verify: true,
};

let state = {
  config: { ...DEFAULT_CONFIG },
  running: false,
  results: [],
  progress: { done: 0, total: 0 },
  error: "",
};

let runToken = 0; // bumping this cancels an in-flight run
const listeners = new Set();

function emit() {
  listeners.forEach((l) => l());
}

function set(patch) {
  state = { ...state, ...patch };
  emit();
}

export function subscribe(cb) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

export function getSnapshot() {
  return state;
}

export function setConfig(patch) {
  set({ config: { ...state.config, ...patch } });
}

export function stopQuiz() {
  runToken += 1;
  set({ running: false });
}

function errorResult(index, detail) {
  return {
    index, question: "(error)", reference: "", answer: "", verdict: "error",
    right: false, judge_reasoning: detail || "error",
    metrics: { input_tokens: 0, output_tokens: 0, reasoning_tokens: 0, total_tokens: 0, elapsed_seconds: 0 },
  };
}

export async function startQuiz() {
  const myToken = ++runToken;
  const cfg = state.config;
  set({ running: true, results: [], error: "", progress: { done: 0, total: 0 } });

  const playerBody = {
    dataset: cfg.dataset,
    mode: cfg.mode,
    model: cfg.model,
    temperature: parseFloat(cfg.temperature) || 0,
    enable_online_tools: cfg.enableOnline,
    multi_hop: cfg.multiHop,
    planner_model: cfg.plannerModel,
    max_hops: parseInt(cfg.maxHops, 10) || 3,
    max_steps: parseInt(cfg.maxSteps, 10) || 6,
    max_depth: parseInt(cfg.maxDepth, 10) || 1,
    verify: cfg.verify,
  };

  try {
    const startRes = await fetch("/api/quiz/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset: cfg.dataset, count: cfg.count === "all" ? null : parseInt(cfg.count, 10) }),
    });
    if (!startRes.ok) throw new Error((await startRes.json()).detail || "start failed");
    const { indices } = await startRes.json();
    set({ progress: { done: 0, total: indices.length } });

    for (let i = 0; i < indices.length; i++) {
      if (myToken !== runToken) return; // cancelled / restarted
      let data;
      try {
        const res = await fetch("/api/quiz/answer", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...playerBody, index: indices[i] }),
        });
        data = res.ok ? await res.json() : errorResult(indices[i], (await res.json()).detail);
      } catch (e) {
        data = errorResult(indices[i], e.message);
      }
      if (myToken !== runToken) return;
      set({
        results: [...state.results, data],
        progress: { done: state.progress.done + 1, total: indices.length },
      });
    }
  } catch (e) {
    set({ error: e.message });
  } finally {
    if (myToken === runToken) set({ running: false });
  }
}
