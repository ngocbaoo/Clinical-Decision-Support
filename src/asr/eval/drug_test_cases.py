"""
Curated Vietnamese ICU test sentences, each containing >=1 specific drug name.

Why curated + TTS instead of real corpus audio: the MultiMed-ST Vietnamese test split (3437 rows)
is NOT drug-name-rich — only ~210 rows contain any drug term and nearly all are the generic word
"thuốc"; exactly one names a specific drug. So real data cannot supply "20 cases with drug names".
These sentences mirror the PRD's bedside questions (§3.1) and ICU drugs (§4.1 F-ASR-03), and the
drug names are drawn from src/rag/safety.py ALLERGY_GROUPS. The `drugs` list is the ground truth
the drug-name-accuracy metric checks for in each transcript.

Caveat (also stated in the report): TTS audio is clean — no bedside noise, overlap, or
disfluency — so the bake-off measures how each model handles VN + drug-name code-switching, not
real-world ICU word-error-rate.
"""

# Each: id, text (spoken + ground-truth transcript), drugs (specific names expected in transcript).
DRUG_TEST_CASES: list[dict] = [
    {"id": "d01", "text": "Bệnh nhân đang dùng Vancomycin một gam, có thêm Gentamicin được không?",
     "drugs": ["Vancomycin", "Gentamicin"]},
    {"id": "d02", "text": "Cho tôi liều Norepinephrine hiện tại của bệnh nhân này.",
     "drugs": ["Norepinephrine"]},
    {"id": "d03", "text": "Bệnh nhân dị ứng Penicillin, dùng Amoxicillin có an toàn không?",
     "drugs": ["Penicillin", "Amoxicillin"]},
    {"id": "d04", "text": "Bệnh nhân suy thận, điều chỉnh liều Meropenem như thế nào?",
     "drugs": ["Meropenem"]},
    {"id": "d05", "text": "Đang truyền Propofol và Fentanyl để an thần, có tương tác gì không?",
     "drugs": ["Propofol", "Fentanyl"]},
    {"id": "d06", "text": "Bệnh nhân nhiễm khuẩn huyết, nên khởi đầu Piperacillin Tazobactam hay Ceftriaxone?",
     "drugs": ["Piperacillin", "Ceftriaxone"]},
    {"id": "d07", "text": "Liều Heparin truyền tĩnh mạch cho bệnh nhân thuyên tắc phổi là bao nhiêu?",
     "drugs": ["Heparin"]},
    {"id": "d08", "text": "Bệnh nhân đang dùng Warfarin, INR cao, cần xử trí gì?",
     "drugs": ["Warfarin"]},
    {"id": "d09", "text": "Có thể phối hợp Midazolam với Morphine cho bệnh nhân thở máy không?",
     "drugs": ["Midazolam", "Morphine"]},
    {"id": "d10", "text": "Bệnh nhân tụt huyết áp, tăng liều Dobutamine hay thêm Dopamine?",
     "drugs": ["Dobutamine", "Dopamine"]},
    {"id": "d11", "text": "Khởi đầu Insulin truyền liên tục khi đường huyết bao nhiêu?",
     "drugs": ["Insulin"]},
    {"id": "d12", "text": "Bệnh nhân phù phổi cấp, liều Furosemide tĩnh mạch nên dùng thế nào?",
     "drugs": ["Furosemide"]},
    {"id": "d13", "text": "Sốc phản vệ thì tiêm Adrenalin bắp với liều nào?",
     "drugs": ["Adrenalin"]},
    {"id": "d14", "text": "Bệnh nhân rung nhĩ nhanh, dùng Amiodarone hay Digoxin tốt hơn?",
     "drugs": ["Amiodarone", "Digoxin"]},
    {"id": "d15", "text": "Viêm phổi bệnh viện, phối hợp Levofloxacin với Ceftazidime được không?",
     "drugs": ["Levofloxacin", "Ceftazidime"]},
    {"id": "d16", "text": "Bệnh nhân dị ứng Sulfamethoxazole, có thay bằng kháng sinh khác không?",
     "drugs": ["Sulfamethoxazole"]},
    {"id": "d17", "text": "Liều Dexamethasone cho bệnh nhân sốc nhiễm khuẩn kháng trị là bao nhiêu?",
     "drugs": ["Dexamethasone"]},
    {"id": "d18", "text": "Bệnh nhân co giật, dùng Midazolam tĩnh mạch hay Phenytoin trước?",
     "drugs": ["Midazolam", "Phenytoin"]},
    {"id": "d19", "text": "Nhiễm nấm xâm lấn, nên dùng Fluconazole hay Amphotericin B?",
     "drugs": ["Fluconazole", "Amphotericin"]},
    {"id": "d20", "text": "Bệnh nhân đau nhiều, có thể dùng Paracetamol kết hợp Ketorolac không?",
     "drugs": ["Paracetamol", "Ketorolac"]},
]
