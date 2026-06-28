import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { listPatients } from "./api.js";
import { IconSearch, IconChevron, IconActivity } from "./icons.jsx";

const GENDER_VI = { male: "Nam", female: "Nữ", other: "Khác", unknown: "Không rõ" };

function fmtDate(iso) {
  const [y, m, d] = (iso || "").split("-");
  return d && m && y ? `${d}/${m}/${y}` : iso;
}

function idLine(p) {
  const parts = [];
  if (p.gender) parts.push(GENDER_VI[p.gender] || p.gender);
  if (p.age != null) parts.push(`${p.age} tuổi`);
  else if (p.birthDate) parts.push(fmtDate(p.birthDate));
  return parts.join(" · ");
}

// Initials for the avatar — last word of a Vietnamese name carries the given name.
function initials(name = "") {
  const w = name.trim().split(/\s+/);
  return ((w.at(-1)?.[0] || "") + (w.length > 1 ? w[0][0] : "")).toUpperCase();
}

// Accent-insensitive match so "Hoa" finds "Hòa" — doctors type fast without diacritics.
const norm = (s) => (s || "").normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();

export default function PatientSelect() {
  const [patients, setPatients] = useState(null);
  const [error, setError] = useState(null);
  const [q, setQ] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    listPatients().then(setPatients).catch((e) => setError(e.message));
  }, []);

  const filtered = useMemo(() => {
    if (!patients) return [];
    const nq = norm(q.trim());
    if (!nq) return patients;
    return patients.filter((p) =>
      norm(`${p.name} ${p.id} ${p.description}`).includes(nq));
  }, [patients, q]);

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true"><IconActivity width={22} height={22} /></span>
          <h1>Trợ lý lâm sàng ICU</h1>
        </div>
        <span className="gate-hint">
          <IconChevron width={15} height={15} />
          Chọn một bệnh nhân để bắt đầu — trợ lý sẽ đọc hồ sơ trước khi trả lời.
        </span>
      </header>

      {error && <div className="error-box">Không tải được danh sách bệnh nhân: {error}</div>}
      {!patients && !error && <div className="subtle pad">Đang tải danh sách bệnh nhân…</div>}

      {patients && (
        <div className="roster-toolbar">
          <label className="search">
            <IconSearch width={18} height={18} />
            <input
              type="search"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Tìm bệnh nhân theo tên, mã, chẩn đoán…"
              aria-label="Tìm bệnh nhân"
            />
          </label>
          <span className="roster-count">
            {filtered.length}/{patients.length} bệnh nhân
          </span>
        </div>
      )}

      {patients && filtered.length === 0 && (
        <div className="empty-state">Không tìm thấy bệnh nhân khớp “{q}”.</div>
      )}

      <div className="patient-grid">
        {filtered.map((p) => (
          <button key={p.id} className="patient-card" onClick={() => navigate(`/chat/${p.id}`)}>
            <div className="patient-card-top">
              <span className="patient-identity">
                <span className="avatar" aria-hidden="true">{initials(p.name)}</span>
                <span style={{ minWidth: 0 }}>
                  <span className="patient-name">{p.name}</span>
                  <div className="patient-idline">{idLine(p)} · {p.id}</div>
                </span>
              </span>
            </div>
            {p.description && <p className="patient-desc">{p.description}</p>}
            <span className="card-cta">
              Mở hồ sơ <IconChevron width={15} height={15} />
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
