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

// The opening assessment runs the full RAG pipeline (~tens of seconds cold). React StrictMode
// double-invokes effects in dev, and re-mounts re-fire them — so we dedupe per patient: concurrent
// callers share ONE in-flight request, and a settled result is cached. This stops two heavy
// pipeline runs from racing on a single-GPU backend (which was wedging it → "Failed to fetch").
const _assessment = new Map(); // pid -> Promise
export const getAssessment = (pid) => {
  if (!_assessment.has(pid)) {
    const p = fetch(`${BASE}/api/patients/${pid}/assessment`, { method: "POST" })
      .then(json)
      .catch((e) => { _assessment.delete(pid); throw e; }); // let a failed run be retried
    _assessment.set(pid, p);
  }
  return _assessment.get(pid);
};

export const sendChat = (pid, query) =>
  fetch(`${BASE}/api/patients/${pid}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  }).then(json);

// Push-to-talk: POST the raw WAV blob, get back {text, latency_s, suggestions}. The UI puts
// `text` into the editable composer — it is never auto-sent.
export const transcribeAudio = (blob) =>
  fetch(`${BASE}/api/asr/transcribe`, {
    method: "POST",
    headers: { "Content-Type": "audio/wav" },
    body: blob,
  }).then(json);
