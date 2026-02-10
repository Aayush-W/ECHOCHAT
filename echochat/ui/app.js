const chat = document.getElementById("chat");
const form = document.getElementById("composer");
const messageInput = document.getElementById("message");
const memoriesToggle = document.getElementById("memories");
const sendButton = form.querySelector("button");

const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("chat-file");
const uploadStatus = document.getElementById("upload-status");
const sessionIdEl = document.getElementById("session-id");
const messageCountEl = document.getElementById("message-count");
const personSelect = document.getElementById("person-select");

const state = {
  sessionId: null,
  echoPerson: null,
  senders: [],
};

function addMessage(text, role) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
  chat.appendChild(el);
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

async function uploadChat(file) {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch("/upload", {
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
  const res = await fetch(`/session/${sessionId}/set_person`, {
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
  const res = await fetch("/chat", {
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

function applySession(data) {
  state.sessionId = data.session_id;
  state.echoPerson = data.echo_person;
  state.senders = data.senders || [];

  updateSessionMeta();
  updateMessageCount(data.message_count);
  populatePersonSelect(state.senders, state.echoPerson);
  setReady(true);
  setStatus(`Ready. Echoing ${state.echoPerson}.`, "success");
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
    applySession(data);
    addMessage("Chat loaded. Ask your first question.", "system");
  } catch (err) {
    setStatus(err.message, "error");
  }
});

personSelect.addEventListener("change", async () => {
  if (!state.sessionId) return;
  const nextPerson = personSelect.value;
  if (!nextPerson) return;

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
    const data = await sendMessage(text, memoriesToggle.checked);
    addMessage(data.response || "(no response)", "bot");
  } catch (err) {
    addMessage(`Error: ${err.message}`, "bot");
  }
});

setReady(false);
setStatus("Waiting for upload", "info");
