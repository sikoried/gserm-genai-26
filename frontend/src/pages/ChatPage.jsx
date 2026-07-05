import { useEffect, useRef, useState } from "react";
import TraceView from "../components/TraceView.jsx";
import "./ChatPage.css";

const MODES = ["World", "RAG", "Agentic RAG"];

export default function ChatPage() {
  const [models, setModels] = useState([]);
  const [chats, setChats] = useState([{ id: 1, title: "New chat", messages: [] }]);
  const [activeChatId, setActiveChatId] = useState(1);
  const [input, setInput] = useState("");
  const [model, setModel] = useState("");
  const [mode, setMode] = useState(MODES[0]);
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [temperature, setTemperature] = useState("0.0");
  const [enableOnline, setEnableOnline] = useState(true);
  const [multiHop, setMultiHop] = useState(true);
  const chatEndRef = useRef(null);
  const textareaRef = useRef(null);
  const nextId = useRef(2);
  const isAgentic = mode === "Agentic RAG";

  const activeChat = chats.find((c) => c.id === activeChatId);

  useEffect(() => {
    fetch("/api/models")
      .then((r) => r.json())
      .then((list) => {
        setModels(list);
        if (list.length) setModel(list[0].id);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeChat?.messages]);

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

  async function send() {
    const text = input.trim();
    if (!text || loading) return;

    // Prior turns (before this message) — lets RAG anchor retrieval to the first message.
    const history = (activeChat?.messages || []).map((m) => ({ role: m.role, content: m.content }));
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
          history,
          mode,
          model,
          temperature: parseFloat(temperature) || 0,
          enable_online_tools: enableOnline,
          multi_hop: multiHop,
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
              {c.title}
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
              <label className="topbar-toggle" title="Break complex questions into sub-questions and answer each hop">
                <input
                  type="checkbox"
                  checked={multiHop}
                  onChange={(e) => setMultiHop(e.target.checked)}
                />
                Multi-hop
              </label>
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
