const rawBase = import.meta.env.VITE_ECHOCHAT_API_BASE || "";
const API_BASE = rawBase.replace(/\/$/, "");

function buildUrl(path) {
  return API_BASE ? `${API_BASE}${path}` : path;
}

async function parseError(res) {
  const text = await res.text();
  if (text) return text;
  return `${res.status} ${res.statusText}`;
}

async function assertOk(res, fallbackMessage) {
  if (!res.ok) {
    const message = (await parseError(res)) || fallbackMessage;
    throw new Error(message);
  }
  return res;
}

export async function uploadChat(file) {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch(buildUrl("/upload"), {
    method: "POST",
    body
  });
  await assertOk(res, "Upload failed");
  return res.json();
}

export async function setPerson(sessionId, echoPerson) {
  const res = await fetch(buildUrl(`/session/${sessionId}/set_person`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ echo_person: echoPerson })
  });
  await assertOk(res, "Update failed");
  return res.json();
}

export async function sendMessage(message, includeMemories, sessionId) {
  const res = await fetch(buildUrl("/chat"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      include_memories: includeMemories,
      session_id: sessionId
    })
  });
  await assertOk(res, "Request failed");
  return res.json();
}

export async function startQuestionnaire(sessionId) {
  const res = await fetch(buildUrl(`/session/${sessionId}/questionnaire/start`), {
    method: "POST"
  });
  await assertOk(res, "Questionnaire start failed");
  return res.json();
}

export async function answerQuestionnaire(sessionId, answer) {
  const res = await fetch(buildUrl(`/session/${sessionId}/questionnaire/answer`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ answer })
  });
  await assertOk(res, "Questionnaire answer failed");
  return res.json();
}
