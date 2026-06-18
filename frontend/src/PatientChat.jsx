import { useEffect, useRef, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { getProfile, getAssessment, sendChat, transcribeAudio } from "./api.js";
import { createRecorder } from "./audio.js";

// Turn a backend answer payload into a chat message. A fallback is a DELIBERATE safety
// refusal, so it gets its own emphasized kind — never a muted/error look.
function toMessage(resp, { opener = false } = {}) {
  const kind = resp.fallback ? "fallback" : resp.alerts?.length ? "alert" : "normal";
  return { role: "ai", kind, opener, ...resp };
}

export default function PatientChat() {
  const { pid } = useParams();
  const [profile, setProfile] = useState(null);
  const [profileErr, setProfileErr] = useState(null);
  const [messages, setMessages] = useState([]);
  const [openerLoading, setOpenerLoading] = useState(true);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const scrollRef = useRef(null);
  const recorderRef = useRef(null);

  useEffect(() => {
    let alive = true;
    setProfile(null); setMessages([]); setOpenerLoading(true); setProfileErr(null);
    getProfile(pid).then((p) => alive && setProfile(p)).catch((e) => alive && setProfileErr(e.message));
    getAssessment(pid)
      .then((a) => alive && setMessages([toMessage(a, { opener: true })]))
      .catch((e) => alive && setMessages([{ role: "ai", kind: "error", answer: `Không tạo được đánh giá ban đầu: ${e.message}` }]))
      .finally(() => alive && setOpenerLoading(false));
    return () => { alive = false; };
  }, [pid]);

  useEffect(() => {
    scrollRef.current?.scrollTo(0, scrollRef.current.scrollHeight);
  }, [messages, busy]);

  async function submit(e) {
    e.preventDefault();
    const q = input.trim();
    if (!q || busy) return;
    setMessages((m) => [...m, { role: "user", answer: q }]);
    setInput(""); setSuggestions([]); setBusy(true);
    try {
      const resp = await sendChat(pid, q);
      setMessages((m) => [...m, toMessage(resp)]);
    } catch (err) {
      setMessages((m) => [...m, { role: "ai", kind: "error", answer: `Lỗi: ${err.message}` }]);
    } finally {
      setBusy(false);
    }
  }

  // Push-to-talk: toggle record. On stop we transcribe and drop the text into the EDITABLE box —
  // never auto-send (the doctor confirms, F-ASR-04/05). Drug suggestions are hints only.
  async function toggleRecord() {
    if (busy || transcribing) return;
    if (!recording) {
      try {
        const rec = createRecorder();
        await rec.start();
        recorderRef.current = rec;
        setRecording(true);
      } catch (err) {
        setMessages((m) => [...m, { role: "ai", kind: "error",
          answer: `Không truy cập được micro: ${err.message}` }]);
      }
      return;
    }
    setRecording(false);
    setTranscribing(true);
    try {
      const blob = await recorderRef.current.stop();
      const resp = await transcribeAudio(blob);
      setInput((cur) => (cur ? `${cur} ` : "") + resp.text);
      setSuggestions(resp.suggestions || []);
    } catch (err) {
      setMessages((m) => [...m, { role: "ai", kind: "error",
        answer: `Lỗi nhận dạng giọng nói: ${err.message}` }]);
    } finally {
      setTranscribing(false);
    }
  }

  // Doctor-initiated correction: replace the misheard span with the suggested drug name in the
  // editable box. We only touch the box if the span is still present (doctor hasn't edited it) —
  // we never rewrite silently.
  function applySuggestion(span, name) {
    setInput((cur) => (span && cur.includes(span) ? cur.replace(span, name) : cur));
  }

  return (
    <div className="chat-layout">
      <ProfilePanel pid={pid} profile={profile} error={profileErr} />
      <main className="chat-main">
        <div className="chat-scroll" ref={scrollRef}>
          {messages.map((m, i) =>
            m.role === "user" ? (
              <div key={i} className="msg msg-user">{m.answer}</div>
            ) : (
              <AiMessage key={i} m={m} />
            )
          )}
          {openerLoading && <div className="msg msg-ai loading">Đang đánh giá bệnh nhân…</div>}
          {busy && <div className="msg msg-ai loading">Đang phân tích…</div>}
        </div>
        {suggestions.length > 0 && (
          <div className="drug-suggest">
            <span className="drug-suggest-head">💊 Gợi ý tên thuốc (hãy kiểm tra trước khi gửi):</span>
            {suggestions.map((s, i) => (
              <span key={i} className="drug-suggest-group">
                <button type="button" className="drug-chip"
                  title={`nghe ra "${s.span}" → đề xuất ${s.suggestion}`}
                  onClick={() => applySuggestion(s.span, s.suggestion)}>
                  {s.suggestion}
                </button>
                {s.alternatives?.map((alt) => (
                  <button key={alt} type="button" className="drug-chip drug-chip-alt"
                    onClick={() => applySuggestion(s.span, alt)}>
                    {alt}?
                  </button>
                ))}
              </span>
            ))}
          </div>
        )}
        <form className="composer" onSubmit={submit}>
          <button
            type="button"
            className={`mic-btn${recording ? " recording" : ""}`}
            onClick={toggleRecord}
            disabled={busy || transcribing}
            title={recording ? "Dừng ghi âm" : "Nói câu hỏi"}
          >
            {recording ? "⏹" : transcribing ? "…" : "🎙"}
          </button>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={recording ? "Đang nghe… nhấn ⏹ để dừng"
              : transcribing ? "Đang nhận dạng giọng nói…" : "Hỏi về bệnh nhân này…"}
            disabled={busy}
          />
          <button type="submit" disabled={busy || !input.trim()}>Gửi</button>
        </form>
      </main>
    </div>
  );
}

function AiMessage({ m }) {
  if (m.kind === "error") return <div className="msg msg-ai error-box">{m.answer}</div>;

  if (m.kind === "fallback") {
    // Deliberate safety decision — clearly communicated, not hidden as a bug.
    return (
      <div className="msg msg-ai safety-decision">
        <div className="safety-decision-head">🛡️ Quyết định an toàn — không trả lời tự do</div>
        <p>{m.answer}</p>
        {m.fallback_reason && <div className="subtle small">Lý do hệ thống: {m.fallback_reason}</div>}
        <Badge m={m} />
      </div>
    );
  }

  return (
    <div className={`msg msg-ai${m.kind === "alert" ? " has-alert" : ""}`}>
      {m.kind === "alert" && <div className="alert-tag">⚠️ Cảnh báo an toàn</div>}
      <p className="answer-text">{m.answer}</p>
      {m.cited_sources?.length > 0 && (
        <div className="sources">
          <div className="sources-head">Nguồn</div>
          {m.cited_sources.map((s) => (
            <div key={s.n} className="source-item">[{s.n}] {s.source} — {s.title}</div>
          ))}
        </div>
      )}
      <Badge m={m} />
    </div>
  );
}

function Badge({ m }) {
  const t = m.timings_s?.total;
  const branch = m.verify?.branch;
  if (t == null && !branch) return null;
  return (
    <div className="badge">
      {t != null && <span>{t}s</span>}
      {branch && <span>verify: {branch}</span>}
    </div>
  );
}

function ProfilePanel({ pid, profile, error }) {
  return (
    <aside className="profile">
      <Link to="/" className="back-link">← Đổi bệnh nhân</Link>
      {error && <div className="error-box">Không tải được hồ sơ: {error}</div>}
      {!profile && !error && <div className="subtle">Đang tải hồ sơ…</div>}
      {profile && (
        <>
          <h2>{profile.name}</h2>
          <div className="subtle">
            {[profile.gender, profile.age != null ? `${profile.age} tuổi` : null, pid]
              .filter(Boolean).join(" · ")}
          </div>
          {profile.encounter?.service_type && (
            <div className="subtle small">{profile.encounter.service_type}</div>
          )}

          {profile.allergies.length > 0 && (
            <Section title="Dị ứng">
              {profile.allergies.map((a, i) => (
                <div key={i} className="allergy">
                  <strong>{a.allergen}</strong>
                  {a.criticality && <span className="pill pill-danger">{a.criticality}</span>}
                  {a.reaction && <span className="subtle small"> — {a.reaction}</span>}
                </div>
              ))}
            </Section>
          )}

          <Section title="Điểm lâm sàng">
            <div className="scores">
              <Score label="qSOFA" value={profile.scores.qsofa} danger={profile.scores.qsofa_positive} suffix="/3" />
              <Score label="SOFA" value={profile.scores.sofa} suffix="/24" />
              <Score label="NEWS2" value={profile.scores.news2} sub={profile.scores.news2_risk} />
              <Score label="MAP" value={profile.scores.map} />
              <Score label="eGFR" value={profile.scores.egfr} sub={profile.scores.egfr_stage} />
            </div>
          </Section>

          {profile.conditions.length > 0 && (
            <Section title="Chẩn đoán">
              <ul className="bullets">{profile.conditions.map((c, i) => <li key={i}>{c}</li>)}</ul>
            </Section>
          )}

          {profile.medications.length > 0 && (
            <Section title="Thuốc đang dùng">
              <ul className="bullets">
                {profile.medications.map((m, i) => (
                  <li key={i}>{m.name}{m.dose ? ` — ${m.dose}` : ""}</li>
                ))}
              </ul>
            </Section>
          )}

          {profile.vitals.length > 0 && (
            <Section title="Chỉ số">
              <div className="vitals">
                {profile.vitals.map((v, i) => (
                  <span key={i} className="vital">{v.key}: <strong>{v.value}{v.unit}</strong></span>
                ))}
              </div>
            </Section>
          )}
        </>
      )}
    </aside>
  );
}

const Section = ({ title, children }) => (
  <section className="psection"><div className="psection-title">{title}</div>{children}</section>
);

function Score({ label, value, suffix = "", sub, danger }) {
  if (value == null) return null;
  return (
    <div className={`score-chip${danger ? " score-danger" : ""}`}>
      <span className="score-label">{label}</span>
      <span className="score-value">{value}{suffix}</span>
      {sub && <span className="score-sub">{sub}</span>}
    </div>
  );
}
