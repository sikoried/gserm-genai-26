import { useEffect, useRef, useState } from "react";
import "./ChatPage.css";

const MODES = ["World", "RAG", "Agentic RAG"];

export default function ChatPage() {
  const [models, setModels] = useState([]);
  const [ragProfiles, setRagProfiles] = useState([]);
  const [ragProfile, setRagProfile] = useState("");
  const [chats, setChats] = useState([{ id: 1, title: "New chat", messages: [] }]);
  const [activeChatId, setActiveChatId] = useState(1);
  const [input, setInput] = useState("");
  const [model, setModel] = useState("");
  const [mode, setMode] = useState(MODES[0]);
  const [loading, setLoading] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [temperature, setTemperature] = useState("0.0");
  const chatEndRef = useRef(null);
  const textareaRef = useRef(null);
  const nextId = useRef(2);

  const activeChat = chats.find((c) => c.id === activeChatId);

  useEffect(() => {
    fetch("/api/models")
      .then((r) => r.json())
      .then((list) => {
        setModels(list);
        if (list.length) setModel(list[0].id);
      })
      .catch(() => {});
    fetch("/api/rag-profiles")
      .then((r) => r.json())
      .then((list) => setRagProfiles(Array.isArray(list) ? list : []))
      .catch(() => {});
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeChat?.messages]);

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
          // A profile (when chosen in RAG mode) drives the advanced pipeline server-side.
          rag_profile: mode === "RAG" && ragProfile ? ragProfile : null,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Server error (${res.status})`);
      }
      const data = await res.json();
      updateChat(chatId, (c) => ({
        ...c,
        messages: [
          ...c.messages,
          { role: "assistant", content: data.answer, reasoning: data.reasoning, sources: data.sources },
        ],
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
          {mode === "RAG" && ragProfiles.length > 0 && (
            <div className="topbar-group">
              <span className="topbar-label">Profile</span>
              <select value={ragProfile} onChange={(e) => setRagProfile(e.target.value)}>
                <option value="">Default</option>
                {ragProfiles.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
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
              {msg.reasoning && (
                <details className="message-reasoning">
                  <summary>Reasoning</summary>
                  <pre>{msg.reasoning}</pre>
                </details>
              )}
              {msg.sources && msg.sources.length > 0 && (
                <details className="message-reasoning">
                  <summary>Sources ({msg.sources.length})</summary>
                  <ul className="message-sources">
                    {msg.sources.map((s, j) => (
                      <li key={j}>
                        <span className="source-score">{s.score.toFixed(2)}</span>{" "}
                        {s.url ? (
                          <a href={s.url} target="_blank" rel="noreferrer">
                            {s.title}
                          </a>
                        ) : (
                          s.title
                        )}
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </div>
          ))}
          {loading && (
            <div className="message assistant">
              <div className="message-role">Oracle</div>
              <div className="message-content loading">Thinking...</div>
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
