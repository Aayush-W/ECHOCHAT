const chat = document.getElementById("chat");
const form = document.getElementById("composer");
const messageInput = document.getElementById("message");
const memoriesToggle = document.getElementById("memories");

function addMessage(text, role) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
}

async function sendMessage(message, includeMemories) {
  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, include_memories: includeMemories }),
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(error || "Request failed");
  }
  return res.json();
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = messageInput.value.trim();
  if (!text) return;

  addMessage(text, "user");
  messageInput.value = "";

  try {
    const data = await sendMessage(text, memoriesToggle.checked);
    addMessage(data.response || "(no response)", "bot");
  } catch (err) {
    addMessage(`Error: ${err.message}`, "bot");
  }
});
