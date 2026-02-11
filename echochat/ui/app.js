const chat = document.getElementById("chat");
const form = document.getElementById("composer");
const messageInput = document.getElementById("message");
const memoriesToggle = document.getElementById("memories");
const sendButton = form.querySelector(".send-btn");

const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("chat-file");
const uploadStatus = document.getElementById("upload-status");
const sessionIdEl = document.getElementById("session-id");
const messageCountEl = document.getElementById("message-count");
const personSelect = document.getElementById("person-select");

const chatList = document.getElementById("chat-list");
const chatSearch = document.getElementById("chat-search");
const chatNameEl = document.getElementById("chat-name");
const chatSubtitleEl = document.getElementById("chat-subtitle");
const chatAvatarEl = document.getElementById("chat-avatar");

const state = {
  sessionId: null,
  echoPerson: null,
  senders: [],
  questionnaireActive: false,
  questionnaireCompleted: false,
};

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

function addMessage(text, role) {
  const row = document.createElement("div");
  row.className = `bubble-row ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const content = document.createElement("div");
  content.className = "bubble-text";
  content.textContent = text;

  const meta = document.createElement("div");
  meta.className = "bubble-meta";
  meta.textContent = formatTime(new Date());

  bubble.appendChild(content);
  if (role !== "system") {
    bubble.appendChild(meta);
  }
  row.appendChild(bubble);
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
}

function setStatus(text, tone = "info") {
  uploadStatus.textContent = text;
  uploadStatus.dataset.tone = tone;
}

function setReady(isReady) {
  messageInput.disabled = !isReady;
  sendButton.disabled = !isReady;
  personSelect.disabled = !isReady || state.senders.length === 0;
}

function updateSessionMeta() {
  sessionIdEl.textContent = state.sessionId || "—";
}

function updateMessageCount(count) {
  messageCountEl.textContent = Number.isFinite(count) ? count : "—";
}

function updateChatHeader() {
  if (!state.echoPerson) {
    chatNameEl.textContent = "EchoChat";
    chatSubtitleEl.textContent = "Upload a chat to begin";
    chatAvatarEl.textContent = "EC";
    return;
  }
  chatNameEl.textContent = state.echoPerson;
  chatSubtitleEl.textContent = "Echo persona online";
  chatAvatarEl.textContent = getInitials(state.echoPerson);
}

function populatePersonSelect(senders, current) {
  personSelect.innerHTML = "";
  if (!senders.length) {
    const option = document.createElement("option");
    option.textContent = "No senders found";
    option.value = "";
    personSelect.appendChild(option);
    return;
  }
  senders.forEach((sender) => {
    const option = document.createElement("option");
    option.value = sender.name;
    option.textContent = `${sender.name} (${sender.count})`;
    if (sender.name === current) {
      option.selected = true;
    }
    personSelect.appendChild(option);
  });
}

function renderChatList() {
  const search = (chatSearch.value || "").toLowerCase();
  chatList.innerHTML = "";

  const filtered = state.senders.filter((sender) =>
    sender.name.toLowerCase().includes(search)
  );

  if (!filtered.length) {
    const empty = document.createElement("div");
    empty.className = "chat-item";
    empty.textContent = "No chats yet. Upload a file.";
    chatList.appendChild(empty);
    return;
  }

  filtered.forEach((sender) => {
    const item = document.createElement("div");
    item.className = "chat-item";
    if (sender.name === state.echoPerson) {
      item.classList.add("active");
    }

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = getInitials(sender.name);

    const meta = document.createElement("div");
    meta.className = "chat-meta";

    const name = document.createElement("div");
    name.className = "chat-name";
    name.textContent = sender.name;

    const preview = document.createElement("div");
    preview.className = "chat-preview";
    preview.textContent = `Messages: ${sender.count}`;

    meta.appendChild(name);
    meta.appendChild(preview);

    const time = document.createElement("div");
    time.className = "chat-time";
    time.textContent = "Now";

    item.appendChild(avatar);
    item.appendChild(meta);
    item.appendChild(time);

    item.addEventListener("click", () => {
      if (!state.sessionId) return;
      if (sender.name === state.echoPerson) return;
      personSelect.value = sender.name;
      switchPerson(sender.name);
    });

    chatList.appendChild(item);
  });
}

const API_BASE = (
  window.ECHOCHAT_API_BASE ||
  window.location.origin ||
  "http://127.0.0.1:5000"
).replace(/\/$/, "");

async function uploadChat(file) {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body,
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(error || "Upload failed");
  }
  return res.json();
}

async function setPerson(sessionId, echoPerson) {
  const res = await fetch(`${API_BASE}/session/${sessionId}/set_person`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ echo_person: echoPerson }),
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(error || "Update failed");
  }
  return res.json();
}

async function sendMessage(message, includeMemories) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      include_memories: includeMemories,
      session_id: state.sessionId,
    }),
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(error || "Request failed");
  }
  return res.json();
}

async function startQuestionnaire(sessionId) {
  const res = await fetch(`${API_BASE}/session/${sessionId}/questionnaire/start`, {
    method: "POST",
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(error || "Questionnaire start failed");
  }
  return res.json();
}

async function answerQuestionnaire(sessionId, answer) {
  const res = await fetch(`${API_BASE}/session/${sessionId}/questionnaire/answer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ answer }),
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(error || "Questionnaire answer failed");
  }
  return res.json();
}

function renderQuestionnairePrompt(payload) {
  if (!payload) return;
  if (payload.status === "completed") {
    state.questionnaireActive = false;
    state.questionnaireCompleted = true;
    addMessage(
      "Questionnaire complete. Training is running in the background.",
      "system"
    );
    return;
  }
  state.questionnaireActive = true;
  state.questionnaireCompleted = false;
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

function applySession(data) {
  state.sessionId = data.session_id;
  state.echoPerson = data.echo_person;
  state.senders = data.senders || [];
  state.questionnaireActive = false;
  state.questionnaireCompleted = false;

  updateSessionMeta();
  updateMessageCount(data.message_count);
  populatePersonSelect(state.senders, state.echoPerson);
  updateChatHeader();
  renderChatList();
  setReady(true);
  setStatus(`Ready. Echoing ${state.echoPerson}.`, "success");
}

async function switchPerson(nextPerson) {
  setReady(false);
  setStatus(`Switching to ${nextPerson}...`, "pending");
  try {
    const data = await setPerson(state.sessionId, nextPerson);
    applySession(data);
    addMessage(`Now echoing ${data.echo_person}.`, "system");
  } catch (err) {
    setStatus(err.message, "error");
    setReady(true);
  }
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files && fileInput.files[0];
  const label = fileInput.closest(".file").querySelector("span");
  label.textContent = file ? file.name : "Choose chat.txt";
});

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    setStatus("Select a chat export first.", "warn");
    return;
  }

  setReady(false);
  setStatus("Processing chat... this can take a moment.", "pending");

  try {
    const data = await uploadChat(file);
    chat.innerHTML = "";
    applySession(data);
    addMessage("Chat loaded. Quick 7-question personality check.", "system");
    try {
      const q = await startQuestionnaire(state.sessionId);
      renderQuestionnairePrompt(q);
    } catch (err) {
      addMessage(`Questionnaire error: ${err.message}`, "system");
    }
  } catch (err) {
    setStatus(err.message, "error");
  }
});

personSelect.addEventListener("change", async () => {
  if (!state.sessionId) return;
  const nextPerson = personSelect.value;
  if (!nextPerson) return;
  switchPerson(nextPerson);
});

chatSearch.addEventListener("input", () => {
  renderChatList();
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = messageInput.value.trim();
  if (!text) return;
  if (!state.sessionId) {
    addMessage("Upload a chat export to begin.", "system");
    return;
  }

  addMessage(text, "user");
  messageInput.value = "";

  try {
    if (state.questionnaireActive) {
      const q = await answerQuestionnaire(state.sessionId, text);
      renderQuestionnairePrompt(q);
      return;
    }
    const data = await sendMessage(text, memoriesToggle.checked);
    addMessage(data.response || "(no response)", "bot");
  } catch (err) {
    addMessage(`Error: ${err.message}`, "bot");
  }
});

setReady(false);
setStatus("Waiting for upload", "info");
updateChatHeader();
renderChatList();
