import { Routes, Route, Navigate } from "react-router-dom";
import PatientSelect from "./PatientSelect.jsx";
import PatientChat from "./PatientChat.jsx";

// The "/" gate enforces "select a patient first": chat lives only under /chat/:pid.
export default function App() {
  return (
    <Routes>
      <Route path="/" element={<PatientSelect />} />
      <Route path="/chat/:pid" element={<PatientChat />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
