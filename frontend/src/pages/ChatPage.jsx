import { useEffect, useRef, useState } from "react";
import TraceView from "../components/TraceView.jsx";
import "./ChatPage.css";

const MODES = ["World", "RAG", "Agentic RAG"];

// Persist chats + UI settings for the running browser session so navigating to
// Comparison and back to Chat doesn't lose them (ChatPage unmounts on route change).
const STORE_KEY = "oracle.chats.v1";
// Bump the version to discard settings persisted under older defaults (e.g. so the
// new "Verify answers = on" default applies instead of a stale stored value).
const SETTINGS_KEY = "oracle.settings.v2";
const DEFAULT_CHATS = [{ id: 1, title: "New chat", messages: [] }];

function loadSession() {
  try {
    const raw = sessionStorage.getItem(STORE_KEY);
    if (!raw) return null;
    const s = JSON.parse(raw);
    if (!Array.isArray(s.chats) || s.chats.length === 0) return null;
    return s;
  } catch {
    return null;
  }
}

function loadSettings() {
  try {
    return JSON.parse(sessionStorage.getItem(SETTINGS_KEY) || "{}") || {};
  } catch {
    return {};
  }
}

export default function ChatPage() {
  const saved = loadSession();
  const cfg = loadSettings();
  const [models, setModels] = useState([]);
  const [chats, setChats] = useState(saved?.chats ?? DEFAULT_CHATS);
  const [activeChatId, setActiveChatId] = useState(saved?.activeChatId ?? 1);
  const [input, setInput] = useState("");
  const [model, setModel] = useState(cfg.model ?? "");
  const [mode, setMode] = useState(cfg.mode ?? MODES[0]);
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [temperature, setTemperature] = useState(cfg.temperature ?? "0.0");
  const [enableOnline, setEnableOnline] = useState(cfg.enableOnline ?? true);
  const [multiHop, setMultiHop] = useState(cfg.multiHop ?? true);
  const [plannerModel, setPlannerModel] = useState(cfg.plannerModel ?? "router");
  const [maxHops, setMaxHops] = useState(cfg.maxHops ?? 3);
  const [maxSteps, setMaxSteps] = useState(cfg.maxSteps ?? 6);
  const [maxDepth, setMaxDepth] = useState(cfg.maxDepth ?? 1);
  const [verify, setVerify] = useState(cfg.verify ?? true);
  const [hopInfoOpen, setHopInfoOpen] = useState(false);
  const chatEndRef = useRef(null);
  const textareaRef = useRef(null);
  const nextId = useRef(saved?.nextId ?? 2);
  const isAgentic = mode === "Agentic RAG";

  const activeChat = chats.find((c) => c.id === activeChatId);

  useEffect(() => {
    fetch("/api/models")
      .then((r) => r.json())
      .then((list) => {
        setModels(list);
        // Keep a persisted model choice if it's still valid; else default to the first.
        setModel((cur) => (list.some((m) => m.id === cur) ? cur : list[0]?.id ?? ""));
      })
      .catch(() => {});
  }, []);

  // Persist UI settings for the session.
  useEffect(() => {
    try {
      sessionStorage.setItem(
        SETTINGS_KEY,
        JSON.stringify({ model, mode, temperature, enableOnline, multiHop,
          plannerModel, maxHops, maxSteps, maxDepth, verify })
      );
    } catch {
      /* non-fatal */
    }
  }, [model, mode, temperature, enableOnline, multiHop, plannerModel,
      maxHops, maxSteps, maxDepth, verify]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeChat?.messages]);

  // Persist chats for the session (survives Chat <-> Comparison navigation).
  useEffect(() => {
    try {
      sessionStorage.setItem(
        STORE_KEY,
        JSON.stringify({ chats, activeChatId, nextId: nextId.current })
      );
    } catch {
      /* storage full / unavailable — non-fatal */
    }
  }, [chats, activeChatId]);

  // Tick an elapsed-seconds counter while a request is in flight, so long
  // agentic runs visibly progress instead of looking hung.
  useEffect(() => {
    if (!loading) return;
    setElapsed(0);
    const started = Date.now();
    const id = setInterval(() => setElapsed((Date.now() - started) / 1000), 100);
    return () => clearInterval(id);
  }, [loading]);

  function updateChat(chatId, updater) {
    setChats((prev) => prev.map((c) => (c.id === chatId ? updater(c) : c)));
  }

  function newChat() {
    const id = nextId.current++;
    setChats((prev) => [...prev, { id, title: "New chat", messages: [] }]);
    setActiveChatId(id);
  }

  function deleteChat(id) {
    const remaining = chats.filter((c) => c.id !== id);
    if (remaining.length === 0) {
      // Never leave zero chats — replace with a fresh empty one.
      const fresh = { id: nextId.current++, title: "New chat", messages: [] };
      setChats([fresh]);
      setActiveChatId(fresh.id);
      return;
    }
    setChats(remaining);
    // If the active chat was deleted, select a neighbouring one.
    if (activeChatId === id) {
      setActiveChatId(remaining[remaining.length - 1].id);
    }
  }

  async function send() {
    const text = input.trim();
    if (!text || loading) return;

    // Each question is answered independently (no prior turns sent) so a new
    // question isn't influenced by earlier ones in the same chat.
    const userMsg = { role: "user", content: text };
    const chatId = activeChatId;

    updateChat(chatId, (c) => {
      const title = c.messages.length === 0 ? text.slice(0, 40) : c.title;
      return { ...c, title, messages: [...c.messages, userMsg] };
    });
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: text,
          mode,
          model,
          temperature: parseFloat(temperature) || 0,
          enable_online_tools: enableOnline,
          multi_hop: multiHop,
          planner_model: plannerModel,
          max_hops: parseInt(maxHops, 10) || 3,
          max_steps: parseInt(maxSteps, 10) || 6,
          max_depth: parseInt(maxDepth, 10) || 1,
          verify,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Server error (${res.status})`);
      }
      const data = await res.json();
      updateChat(chatId, (c) => ({
        ...c,
        messages: [...c.messages, { role: "assistant", content: data.answer, reasoning: data.reasoning, trace: data.trace }],
      }));
    } catch (err) {
      updateChat(chatId, (c) => ({
        ...c,
        messages: [...c.messages, { role: "assistant", content: `Error: ${err.message}` }],
      }));
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  function autoResize() {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  }

  return (
    <div className="app">
      {/* ── Sidebar: Chat History ── */}
      <aside className="sidebar">
        <div className="sidebar-header">Chat History</div>
        <div className="sidebar-chats">
          {chats.map((c) => (
            <div
              key={c.id}
              className={`sidebar-chat-item${c.id === activeChatId ? " active" : ""}`}
              onClick={() => setActiveChatId(c.id)}
            >
              <span className="sidebar-chat-title">{c.title}</span>
              <button
                type="button"
                className="sidebar-chat-delete"
                aria-label={`Delete chat: ${c.title}`}
                title="Delete chat"
                onClick={(e) => {
                  e.stopPropagation();
                  deleteChat(c.id);
                }}
              >
                ×
              </button>
            </div>
          ))}
        </div>
        <div className="sidebar-new-chat" onClick={newChat}>
          + New Chat
        </div>
        <div className="sidebar-footer">
          <button className="settings-btn" onClick={() => setSettingsOpen(true)}>
            Settings
          </button>
        </div>
      </aside>

      {/* ── Main Area ── */}
      <div className="main">
        <div className="topbar">
          <div className="topbar-group">
            <span className="topbar-label">Model</span>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {models.length === 0 && <option value="">Loading…</option>}
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </div>
          <div className="topbar-group">
            <span className="topbar-label">Mode</span>
            <select value={mode} onChange={(e) => setMode(e.target.value)}>
              {MODES.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
          {isAgentic && (
            <div className="topbar-group">
              <label className="topbar-toggle">
                <input
                  type="checkbox"
                  checked={multiHop}
                  onChange={(e) => setMultiHop(e.target.checked)}
                />
                Multi-hop
              </label>
              <div className="topbar-info-wrap">
                <button
                  type="button"
                  className="topbar-info"
                  aria-label="What is multi-hop?"
                  aria-expanded={hopInfoOpen}
                  onClick={() => setHopInfoOpen((v) => !v)}
                >
                  ⓘ
                </button>
                {hopInfoOpen && (
                  <>
                    <div className="info-backdrop" onClick={() => setHopInfoOpen(false)} />
                    <div className="info-popover" role="dialog">
                      <p>
                        <b>Multi-hop ON</b> — the question is broken into a chain of
                        sub-questions, each answered in turn with the previous answer
                        fed into the next. Best when the answer depends on an
                        intermediate fact, e.g. <i>"In which country was the director
                        of the highest-grossing 1997 film born?"</i>
                      </p>
                      <p>
                        <b>Multi-hop OFF</b> (single-hop) — the question is answered in
                        one pass with direct tool calls. Faster, and best for simple,
                        self-contained questions, e.g. <i>"What is the capital of
                        France?"</i>
                      </p>
                    </div>
                  </>
                )}
              </div>
              {multiHop && (
                <label className="topbar-toggle" style={{ marginLeft: 10 }}>
                  <span className="topbar-label" style={{ marginRight: 2 }}>Planner</span>
                  <select
                    value={plannerModel}
                    onChange={(e) => setPlannerModel(e.target.value)}
                    title="Which model decomposes the question into sub-questions"
                  >
                    <option value="router">Small (fast)</option>
                    <option value="answer">Big model (better)</option>
                  </select>
                </label>
              )}
            </div>
          )}
        </div>

        <div className="chat-window">
          {activeChat?.messages.length === 0 && (
            <div className="chat-window-empty">Ask a question to get started</div>
          )}
          {activeChat?.messages.map((msg, i) => (
            <div key={i} className={`message ${msg.role}`}>
              <div className="message-role">{msg.role === "user" ? "You" : "Oracle"}</div>
              <div className="message-content">{msg.content}</div>
              {msg.trace ? (
                <TraceView trace={msg.trace} />
              ) : (
                msg.reasoning && (
                  <details className="message-reasoning">
                    <summary>Reasoning</summary>
                    <pre>{msg.reasoning}</pre>
                  </details>
                )
              )}
            </div>
          ))}
          {loading && (
            <div className="message assistant">
              <div className="message-role">Oracle</div>
              <div className="message-content loading">
                {isAgentic
                  ? "Working… routing tools, searching, and composing the answer"
                  : "Thinking…"}{" "}
                <span className="loading-elapsed">{elapsed.toFixed(1)}s</span>
                <div className="progress-track">
                  <div className="progress-bar" />
                </div>
                {isAgentic && (
                  <div className="loading-hint">
                    Agentic runs can take a while (local routing model + tools). This is normal.
                  </div>
                )}
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <div className="chat-input-bar">
          <div className="chat-input-wrapper">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                autoResize();
              }}
              onKeyDown={handleKeyDown}
              placeholder="Type your question..."
              rows={1}
            />
            <button className="send-btn" onClick={send} disabled={loading || !input.trim()}>
              Send
            </button>
          </div>
        </div>
      </div>

      {/* ── Settings Modal ── */}
      {settingsOpen && (
        <div className="modal-overlay" onClick={() => setSettingsOpen(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h2>Settings</h2>
            <div className="modal-field">
              <label>Temperature</label>
              <input
                type="number"
                step="0.1"
                min="0"
                max="2"
                value={temperature}
                onChange={(e) => setTemperature(e.target.value)}
              />
            </div>

            <div className="modal-field modal-field-check">
              <label>
                <input
                  type="checkbox"
                  checked={enableOnline}
                  onChange={(e) => setEnableOnline(e.target.checked)}
                />
                Web + YouTube tools (Agentic RAG)
              </label>
              <span className="modal-hint">
                On by default. Lets the agent use google_search and youtube for
                current/video facts. Turn off for a deliberately offline run.
              </span>
            </div>

            <div className="modal-section">Agentic RAG budgets</div>

            <div className="modal-field">
              <label>Max steps (max_steps)</label>
              <input
                type="number" min="1" max="20" step="1"
                value={maxSteps}
                onChange={(e) => setMaxSteps(e.target.value)}
              />
              <span className="modal-hint">
                The most tool calls the agent may make for one question before it must
                stop and answer with what it has. Higher = can gather more evidence but
                slower; lower = faster but may answer with less.
              </span>
            </div>

            <div className="modal-field">
              <label>Max sub-questions per plan (max_hops)</label>
              <input
                type="number" min="1" max="8" step="1"
                value={maxHops}
                onChange={(e) => setMaxHops(e.target.value)}
              />
              <span className="modal-hint">
                When multi-hop is on, the most sub-questions the planner may split a
                question into at each level. Higher handles more complex chains but
                costs more.
              </span>
            </div>

            <div className="modal-field">
              <label>Max planning depth (max_depth)</label>
              <input
                type="number" min="1" max="4" step="1"
                value={maxDepth}
                onChange={(e) => setMaxDepth(e.target.value)}
              />
              <span className="modal-hint">
                How many levels a hard question may be recursively broken down: 1 = plan
                once (flat); 2+ = a sub-question can itself be decomposed by a nested
                sub-agent. Deeper solves harder chains but multiplies cost.
              </span>
            </div>

            <div className="modal-field modal-field-check">
              <label>
                <input
                  type="checkbox"
                  checked={verify}
                  onChange={(e) => setVerify(e.target.checked)}
                />
                Verify answer (verification / backtracking hop)
              </label>
              <span className="modal-hint">
                On by default. After answering, run one extra check against the gathered
                facts; if the answer is contradicted, correct it. More reliable, but adds
                one model call. Bounded — it runs at most once.
              </span>
            </div>

            <div className="modal-actions">
              <button className="primary" onClick={() => setSettingsOpen(false)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
