"""
R3 — Context builder: assemble the grounded prompt for generation.

Priority order when space is tight (PRD risk table): allergy alerts > clinical
scores > top retrieved chunks > medications > the rest. Chunks are numbered
[1]..[k] so the generator can cite them and the post-check can verify.
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on sys.path
from rag.config import CHUNK_CHAR_CAP, DISCLAIMER  # noqa: E402


def summarize_patient(ctx: dict, calc: dict) -> str:
    """Compact Vietnamese patient summary for the prompt (≈30 lines max)."""
    if not ctx:
        return "(Không có dữ liệu bệnh nhân — trả lời theo guideline chung.)"
    p = ctx.get("patient", {})
    lines = [f"Bệnh nhân: {p.get('name', '?')} | {p.get('gender', '?')} | "
             f"{p.get('age', '?')} tuổi"]

    allergies = ctx.get("allergies", [])
    if allergies:
        items = ", ".join(f"{a.get('allergen', '?')}"
                          f" ({a.get('criticality') or '?'})" for a in allergies)
        lines.append(f"DỊ ỨNG: {items}")
    else:
        lines.append("Dị ứng: không ghi nhận")

    conds = ctx.get("conditions", [])
    if conds:
        items = "; ".join(
            (c.get("name_vi") or c.get("display") or c.get("icd10_code", "?"))
            for c in conds[:6])
        lines.append(f"Chẩn đoán: {items}")

    obs = ctx.get("observations", {})
    vals, missing = [], []
    for key, o in obs.items():
        if o.get("value") is None:
            missing.append(key)
        else:
            unit = o.get("unit") or ""
            vals.append(f"{key}={o['value']}{unit}")
    if vals:
        lines.append("Chỉ số: " + ", ".join(vals))
    if missing:
        lines.append("THIẾU DỮ LIỆU: " + ", ".join(missing))

    meds = ctx.get("medications", [])
    if meds:
        items = "; ".join(f"{m.get('name', '?')} {m.get('dose', '')}".strip()
                          for m in meds[:8])
        lines.append(f"Thuốc đang dùng: {items}")

    if calc:
        s = []
        m = calc.get("map", {})
        if m.get("value") is not None:
            s.append(f"MAP={m['value']}")
        q = calc.get("qsofa", {})
        s.append(f"qSOFA={q.get('total', '?')}/3"
                 + (" (DƯƠNG TÍNH)" if q.get("positive") else ""))
        so = calc.get("sofa", {})
        s.append(f"SOFA={so.get('total', '?')}/24")
        n = calc.get("news2", {})
        s.append(f"NEWS2={n.get('total', '?')} ({n.get('risk_level', '?')})")
        e = calc.get("egfr", {})
        if e.get("egfr") is not None:
            s.append(f"eGFR={e['egfr']} ({e.get('stage', '')})")
        lines.append("Điểm số (đã tính sẵn, dùng đúng các giá trị này): " + " | ".join(s))
        alerts = calc.get("summary", {}).get("alerts", [])
        for a in alerts:
            lines.append(f"CẢNH BÁO: {a}")
    return "\n".join(lines)


def format_chunks(chunks: list[dict]) -> str:
    """Number retrieved chunks [1]..[k] with source + title for citation."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        text = c["text"][:CHUNK_CHAR_CAP]
        blocks.append(f"[{i}] ({c['source']} — {c['title']})\n{text}")
    return "\n\n".join(blocks)


SYSTEM_PROMPT = f"""\
Bạn là trợ lý lâm sàng ICU hỗ trợ bác sĩ. CHỈ trả lời dựa trên các TÀI LIỆU \
được đánh số và DỮ LIỆU BỆNH NHÂN được cung cấp — tuyệt đối không dùng kiến thức ngoài.

Quy tắc bắt buộc:
1. Mỗi khuyến nghị phải kèm trích dẫn [n] trỏ đến tài liệu tương ứng.
2. Nếu có CẢNH BÁO DỊ ỨNG trong dữ liệu, nhắc lại cảnh báo đó ĐẦU TIÊN.
3. Nếu tài liệu không đủ để trả lời, đặt "insufficient": true thay vì suy diễn.
4. Điểm số lâm sàng (NEWS2/qSOFA/SOFA/eGFR/MAP) đã được tính sẵn — dùng đúng giá trị, không tự tính lại.
5. Trả lời tiếng Việt, ngắn gọn (tối đa ~120 từ), bác sĩ đọc được trong 10 giây.
6. Kết thúc câu trả lời bằng: "{DISCLAIMER}"

Chỉ trả về JSON:
{{"answer": "...", "citations": [1, 2], "confidence": 0.0-1.0, "insufficient": false}}
"""


def build_messages(query: str, chunks: list[dict], patient_summary: str,
                   alert_text: str) -> list[dict]:
    user = []
    if alert_text:
        user.append(f"=== CẢNH BÁO AN TOÀN (bắt buộc nhắc lại đầu tiên) ===\n{alert_text}")
    user.append(f"=== DỮ LIỆU BỆNH NHÂN ===\n{patient_summary}")
    user.append(f"=== TÀI LIỆU ===\n{format_chunks(chunks)}")
    user.append(f"=== CÂU HỎI ===\n{query}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user)},
    ]
