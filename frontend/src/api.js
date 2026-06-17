const BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function json(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* ignore */ }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

export const listPatients = () => fetch(`${BASE}/api/patients`).then(json);

export const getProfile = (pid) =>
  fetch(`${BASE}/api/patients/${pid}`).then(json);

export const getAssessment = (pid) =>
  fetch(`${BASE}/api/patients/${pid}/assessment`, { method: "POST" }).then(json);

export const sendChat = (pid, query) =>
  fetch(`${BASE}/api/patients/${pid}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  }).then(json);
