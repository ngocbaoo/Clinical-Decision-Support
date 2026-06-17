import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { listPatients } from "./api.js";

const GENDER_VI = { male: "Nam", female: "Nữ", other: "Khác", unknown: "Không rõ" };

function fmtDate(iso) {
  const [y, m, d] = (iso || "").split("-");
  return d && m && y ? `${d}/${m}/${y}` : iso;
}

function idLine(p) {
  const parts = [];
  if (p.gender) parts.push(GENDER_VI[p.gender] || p.gender);
  if (p.birthDate) parts.push(fmtDate(p.birthDate));
  return parts.join(" · ");
}

export default function PatientSelect() {
  const [patients, setPatients] = useState(null);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    listPatients().then(setPatients).catch((e) => setError(e.message));
  }, []);

  return (
    <div className="page">
      <header className="topbar">
        <h1>Trợ lý lâm sàng ICU</h1>
        <span className="subtle">Chọn bệnh nhân để bắt đầu — trợ lý sẽ đọc hồ sơ trước.</span>
      </header>

      {error && <div className="error-box">Không tải được danh sách bệnh nhân: {error}</div>}
      {!patients && !error && <div className="subtle pad">Đang tải danh sách bệnh nhân…</div>}

      <div className="patient-grid">
        {patients?.map((p) => (
          <button key={p.id} className="patient-card" onClick={() => navigate(`/chat/${p.id}`)}>
            <span className="patient-name">{p.name}</span>
            <div className="patient-idline">{idLine(p)}</div>
          </button>
        ))}
      </div>
    </div>
  );
}
