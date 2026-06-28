"""Deterministic comorbidity-conflict enforcement gate (Risk #1 backstop)."""

import unicodedata


def _norm(s: str) -> str:
    s = (s or "").lower().replace("đ", "d")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


RULES = [
    {
        "id": "aggressive_fluids",
        "rec": ["bù dịch nhanh", "truyền dịch nhanh", "1000-2000", "1000 - 2000", "30ml/kg",
                "30 ml/kg", "bolus dịch", "test truyền dịch", "hồi sức dịch", "bù dịch tích cực"],
        "cond": ["suy gan", "xơ gan", "bệnh não gan", "suy tim", "phù phổi", "esrd",
                 "suy thận", "bệnh thận mạn"],
        "severity": "critical",
        "message": ("Bù dịch tích cực cần THẬN TRỌNG ở bệnh nhân {cond}: nguy cơ quá tải dịch / phù "
                    "phổi (và vỡ giãn tĩnh mạch nếu xơ gan). Cá thể hóa thể tích và tốc độ, theo dõi "
                    "sát đáp ứng."),
    },
    {
        "id": "hepatotoxic_paracetamol",
        "rec": ["paracetamol", "acetaminophen", "efferalgan", "hạ sốt"],
        "cond": ["suy gan", "xơ gan", "bệnh gan", "bệnh não gan"],
        "severity": "critical",
        "message": ("Paracetamol có độc tính gan: ở bệnh nhân {cond} cần TRÁNH hoặc giảm liều sâu và "
                    "theo dõi men gan; hỏi dược sĩ lâm sàng trước khi dùng."),
    },
    {
        "id": "nephrotoxic_nsaid",
        "rec": ["nsaid", "ibuprofen", "diclofenac", "kháng viêm không steroid"],
        "cond": ["suy thận", "bệnh thận mạn", "esrd", "suy gan", "xơ gan"],
        "severity": "critical",
        "message": ("NSAID gây độc thận / giữ muối nước: ở bệnh nhân {cond} cần TRÁNH; cân nhắc thuốc "
                    "giảm đau thay thế."),
    },
]


def _patient_conditions(patient_context: dict) -> list[str]:
    out = []
    for c in (patient_context or {}).get("conditions", []) or []:
        nm = (c.get("name_vi") or c.get("display") or c.get("name_en") or "").strip()
        if nm:
            out.append(nm)
    return out


def check_comorbidity_conflicts(answer: str, patient_context: dict) -> list[dict]:
    na = _norm(answer)
    if not na:
        return []
    conds = [(_norm(nm), nm) for nm in _patient_conditions(patient_context)]
    conflicts = []
    for rule in RULES:
        if not any(_norm(k) in na for k in rule["rec"]):
            continue
        hit = next((orig for ncond, orig in conds
                    if any(_norm(ck) in ncond for ck in rule["cond"])), None)
        if hit:
            conflicts.append({"id": rule["id"], "severity": rule["severity"],
                              "comorbidity": hit, "message": rule["message"].format(cond=hit)})
    return conflicts


def apply_comorbidity_gate(response: dict, patient_context: dict) -> dict:
    if response.get("fallback"):
        return response
    conflicts = check_comorbidity_conflicts(response.get("answer", ""), patient_context)
    if not conflicts:
        return response
    banner = ("⚠️ LƯU Ý BỆNH NỀN — kiểm tra trước khi thực hiện:\n"
              + "\n".join(f"• {c['message']}" for c in conflicts))
    out = dict(response)
    out["answer"] = f"{banner}\n\n{response.get('answer', '')}"
    out["alerts"] = list(response.get("alerts") or []) + [
        {"type": "comorbidity_conflict", "drug": None, **c} for c in conflicts]
    out["comorbidity_conflicts"] = conflicts
    return out
