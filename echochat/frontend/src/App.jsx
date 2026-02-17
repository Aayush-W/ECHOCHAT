import { useEffect, useMemo, useRef, useState } from "react";
import {
  answerQuestionnaire,
  sendMessage,
  setPerson,
  startQuestionnaire,
  uploadChat
} from "./api.js";

function formatTime(date) {
  const hours = date.getHours();
  const minutes = date.getMinutes().toString().padStart(2, "0");
  const suffix = hours >= 12 ? "pm" : "am";
  const hour = hours % 12 || 12;
  return `${hour}:${minutes} ${suffix}`;
}

function getInitials(name) {
  if (!name) return "EC";
  return name
    .split(" ")
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0].toUpperCase())
    .join("");
}

function nextId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export default function App() {
  const [sessionId, setSessionId] = useState(null);
  const [echoPerson, setEchoPerson] = useState(null);
  const [senders, setSenders] = useState([]);
  const [messageCount, setMessageCount] = useState(null);
  const [questionnaire, setQuestionnaire] = useState({
    active: false,
    completed: false
  });
  const [messages, setMessages] = useState([]);
  const [uploadStatus, setUploadStatus] = useState({
    text: "Waiting for upload",
    tone: "info"
  });
  const [isReady, setIsReady] = useState(false);
  const [includeMemories, setIncludeMemories] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileLabel, setFileLabel] = useState("Choose chat.txt");
  const [messageInput, setMessageInput] = useState("");
  const chatRef = useRef(null);

  useEffect(() => {
    if (!chatRef.current) return;
    chatRef.current.scrollTop = chatRef.current.scrollHeight;
  }, [messages]);

  const filteredSenders = useMemo(() => {
    const search = (searchTerm || "").toLowerCase();
    if (!search) return senders;
    return senders.filter((sender) =>
      sender.name.toLowerCase().includes(search)
    );
  }, [senders, searchTerm]);

  const chatName = echoPerson || "EchoChat";
  const chatSubtitle = echoPerson
    ? "Echo persona online"
    : "Upload a chat to begin";
  const chatAvatar = getInitials(echoPerson);
  const sessionLabel = sessionId || "-";
  const messageCountLabel = Number.isFinite(messageCount)
    ? messageCount
    : "-";
  const readyForChat = isReady && Boolean(sessionId);
  const canSelectPerson = readyForChat && senders.length > 0;

  function addMessage(text, role) {
    setMessages((prev) => [
      ...prev,
      {
        id: nextId(),
        text,
        role,
        ts: Date.now()
      }
    ]);
  }

  function applySession(data) {
    setSessionId(data.session_id);
    setEchoPerson(data.echo_person);
    setSenders(data.senders || []);
    setMessageCount(data.message_count);
    setQuestionnaire({ active: false, completed: false });
    setIsReady(true);
    setUploadStatus({
      text: `Ready. Echoing ${data.echo_person}.`,
      tone: "success"
    });
  }

  function renderQuestionnairePrompt(payload) {
    if (!payload) return;
    if (payload.status === "completed") {
      setQuestionnaire({ active: false, completed: true });
      addMessage(
        "Questionnaire complete. Training is running in the background.",
        "system"
      );
      return;
    }
    setQuestionnaire({ active: true, completed: false });
    const next = payload.next_question;
    if (!next) return;
    const index = payload.answered_count + 1;
    const total = payload.total_questions || 7;
    const parts = [`Q${index}/${total}: ${next.text}`];
    if (next.options && next.options.length) {
      parts.push(`Options: ${next.options.join(" | ")}`);
    }
    if (next.hint) {
      parts.push(`Hint: ${next.hint}`);
    }
    addMessage(parts.join("\n"), "system");
  }

  async function handleUpload(event) {
    event.preventDefault();
    if (!selectedFile) {
      setUploadStatus({ text: "Select a chat export first.", tone: "warn" });
      return;
    }

    setIsReady(false);
    setUploadStatus({
      text: "Processing chat... this can take a moment.",
      tone: "pending"
    });

    try {
      const data = await uploadChat(selectedFile);
      setMessages([]);
      applySession(data);
      addMessage("Chat loaded. Quick 7-question personality check.", "system");
      try {
        const q = await startQuestionnaire(data.session_id);
        renderQuestionnairePrompt(q);
      } catch (err) {
        addMessage(`Questionnaire error: ${err.message}`, "system");
      }
    } catch (err) {
      setUploadStatus({ text: err.message, tone: "error" });
      setIsReady(false);
    }
  }

  async function switchPerson(nextPerson) {
    if (!sessionId || !nextPerson || nextPerson === echoPerson) return;
    setIsReady(false);
    setUploadStatus({
      text: `Switching to ${nextPerson}...`,
      tone: "pending"
    });
    try {
      const data = await setPerson(sessionId, nextPerson);
      applySession(data);
      addMessage(`Now echoing ${data.echo_person}.`, "system");
    } catch (err) {
      setUploadStatus({ text: err.message, tone: "error" });
      setIsReady(true);
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    const text = messageInput.trim();
    if (!text) return;

    if (!sessionId) {
      addMessage("Upload a chat export to begin.", "system");
      setMessageInput("");
      return;
    }

    addMessage(text, "user");
    setMessageInput("");

    try {
      if (questionnaire.active) {
        const q = await answerQuestionnaire(sessionId, text);
        renderQuestionnairePrompt(q);
        return;
      }
      const data = await sendMessage(text, includeMemories, sessionId);
      addMessage(data.response || "(no response)", "bot");
    } catch (err) {
      addMessage(`Error: ${err.message}`, "bot");
    }
  }

  return (
    <div className="wa-shell">
      <aside className="wa-sidebar">
        <div className="wa-sidebar-top">
          <div className="wa-title">Chats</div>
          <div className="wa-actions">
            <button className="icon-btn" type="button" aria-label="New chat">
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path
                  d="M12 5v14M5 12h14"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                />
              </svg>
            </button>
            <button className="icon-btn" type="button" aria-label="Menu">
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <circle cx="12" cy="5" r="1.6" fill="currentColor" />
                <circle cx="12" cy="12" r="1.6" fill="currentColor" />
                <circle cx="12" cy="19" r="1.6" fill="currentColor" />
              </svg>
            </button>
          </div>
        </div>

        <div className="wa-search">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <circle
              cx="11"
              cy="11"
              r="7"
              stroke="currentColor"
              strokeWidth="2"
              fill="none"
            />
            <path
              d="M16.5 16.5l4 4"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
            />
          </svg>
          <input
            id="chat-search"
            type="text"
            placeholder="Search or start a new chat"
            autoComplete="off"
            value={searchTerm}
            onChange={(event) => setSearchTerm(event.target.value)}
          />
        </div>

        <div className="wa-filters">
          <button className="chip active" type="button">
            All
          </button>
          <button className="chip" type="button">
            Unread
          </button>
          <button className="chip" type="button">
            Favourites
          </button>
          <button className="chip" type="button">
            Groups
          </button>
        </div>

        <div className="wa-card upload-card">
          <div className="wa-card-head">
            <div>
              <h2>Upload chat</h2>
              <p>WhatsApp .txt export. Builds persona + memories.</p>
            </div>
            <div className="wa-pill">Local</div>
          </div>
          <form className="upload-form" onSubmit={handleUpload}>
            <label className="file">
              <input
                id="chat-file"
                type="file"
                accept=".txt"
                onChange={(event) => {
                  const file = event.target.files && event.target.files[0];
                  setSelectedFile(file || null);
                  setFileLabel(file ? file.name : "Choose chat.txt");
                }}
              />
              <span>{fileLabel}</span>
            </label>
            <button id="upload-btn" type="submit">
              Process
            </button>
          </form>
          <div className="status" data-tone={uploadStatus.tone}>
            {uploadStatus.text}
          </div>
          <div className="upload-meta">
            <div>
              <span className="label">Session</span>
              <span>{sessionLabel}</span>
            </div>
            <div>
              <span className="label">Messages</span>
              <span>{messageCountLabel}</span>
            </div>
          </div>
        </div>

        <div className="wa-section">
          <div className="wa-section-title">Echo person</div>
          <select
            id="person-select"
            disabled={!canSelectPerson}
            value={echoPerson || ""}
            onChange={(event) => switchPerson(event.target.value)}
          >
            {senders.length === 0 ? (
              <option value="">No senders found</option>
            ) : (
              senders.map((sender) => (
                <option key={sender.name} value={sender.name}>
                  {sender.name} ({sender.count})
                </option>
              ))
            )}
          </select>
        </div>

        <div className="wa-chat-list">
          {filteredSenders.length === 0 ? (
            <div className="chat-item">No chats yet. Upload a file.</div>
          ) : (
            filteredSenders.map((sender) => (
              <div
                key={sender.name}
                className={`chat-item${
                  sender.name === echoPerson ? " active" : ""
                }`}
                onClick={() => switchPerson(sender.name)}
                role="button"
                tabIndex={0}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    switchPerson(sender.name);
                  }
                }}
              >
                <div className="avatar">{getInitials(sender.name)}</div>
                <div className="chat-meta">
                  <div className="chat-name">{sender.name}</div>
                  <div className="chat-preview">
                    Messages: {sender.count}
                  </div>
                </div>
                <div className="chat-time">Now</div>
              </div>
            ))
          )}
        </div>
      </aside>

      <section className="wa-main">
        <header className="wa-chat-header">
          <div className="chat-info">
            <div className="avatar" id="chat-avatar">
              {chatAvatar}
            </div>
            <div>
              <div className="chat-name" id="chat-name">
                {chatName}
              </div>
              <div className="chat-status" id="chat-subtitle">
                {chatSubtitle}
              </div>
            </div>
          </div>
          <div className="chat-actions">
            <label className="mem-toggle">
              <input
                id="memories"
                type="checkbox"
                checked={includeMemories}
                onChange={(event) => setIncludeMemories(event.target.checked)}
              />
              <span>Use memories</span>
            </label>
            <button className="icon-btn" type="button" aria-label="Search">
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <circle
                  cx="11"
                  cy="11"
                  r="7"
                  stroke="currentColor"
                  strokeWidth="2"
                  fill="none"
                />
                <path
                  d="M16.5 16.5l4 4"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                />
              </svg>
            </button>
            <button className="icon-btn" type="button" aria-label="More">
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <circle cx="12" cy="5" r="1.6" fill="currentColor" />
                <circle cx="12" cy="12" r="1.6" fill="currentColor" />
                <circle cx="12" cy="19" r="1.6" fill="currentColor" />
              </svg>
            </button>
          </div>
        </header>

        <main id="chat" className="wa-chat-window" ref={chatRef}>
          {messages.map((msg) => (
            <div key={msg.id} className={`bubble-row ${msg.role}`}>
              <div className="bubble">
                <div className="bubble-text">{msg.text}</div>
                {msg.role !== "system" && (
                  <div className="bubble-meta">
                    {formatTime(new Date(msg.ts))}
                  </div>
                )}
              </div>
            </div>
          ))}
        </main>

        <form id="composer" className="wa-composer" onSubmit={handleSubmit}>
          <button className="icon-btn ghost" type="button" aria-label="Attach">
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path
                d="M7 12.5l7.5-7.5a3.5 3.5 0 0 1 5 5l-8.5 8.5a5 5 0 0 1-7-7l8.2-8.2"
                stroke="currentColor"
                strokeWidth="1.8"
                fill="none"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
          <button className="icon-btn ghost" type="button" aria-label="Emoji">
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <circle
                cx="12"
                cy="12"
                r="9"
                stroke="currentColor"
                strokeWidth="1.8"
                fill="none"
              />
              <circle cx="9" cy="10" r="1" fill="currentColor" />
              <circle cx="15" cy="10" r="1" fill="currentColor" />
              <path
                d="M8 15c1.1 1 2.5 1.5 4 1.5s2.9-.5 4-1.5"
                stroke="currentColor"
                strokeWidth="1.6"
                fill="none"
                strokeLinecap="round"
              />
            </svg>
          </button>
          <input
            id="message"
            type="text"
            placeholder="Type a message"
            autoComplete="off"
            value={messageInput}
            onChange={(event) => setMessageInput(event.target.value)}
            disabled={!readyForChat}
          />
          <button className="send-btn" type="submit" disabled={!readyForChat}>
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path
                d="M3 11.5l17-8-5 18-3-7-9-3z"
                fill="currentColor"
              />
            </svg>
          </button>
        </form>
      </section>
    </div>
  );
}
