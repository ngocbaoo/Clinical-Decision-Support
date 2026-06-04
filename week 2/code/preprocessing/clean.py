import re

def clean_markdown(text):
    # 1. Xóa form feed và số trang
    text = re.sub(r'\x0c\d*', '', text)
    
    # 2. Fix ký tự unicode lỗi phổ biến
    replacements = {
        'Ƣ': 'Ư', 'ƣ': 'ư',
        'Ơ': 'Ơ', 'ơ': 'ơ',  # kiểm tra thêm
        '\u201c': '"', '\u201d': '"',  # smart quotes
        '\u2018': "'", '\u2019': "'",
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)
    
    # 3. Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)  # max 2 dòng trống
    text = re.sub(r' {2,}', ' ', text)       # multiple spaces
    
    return text

def chunk_procedures(text):
    procedure_pattern = re.compile(
        r'(?=QUY TRÌNH KỸ THUẬT)',
        re.IGNORECASE
    )
    
    raw_chunks = procedure_pattern.split(text)
    chunks = []
    
    for chunk in raw_chunks:
        chunk = chunk.strip()
        
        # Fix 1: Bỏ chunks quá ngắn — không phải quy trình thật
        if len(chunk) < 800:
            continue
        
        # Fix 2: Bỏ chunks không có cấu trúc quy trình
        # Quy trình thật luôn có ít nhất CHỈ ĐỊNH hoặc TIẾN HÀNH
        has_structure = (
            'CHỈ ĐỊNH' in chunk.upper() or
            'TIẾN HÀNH' in chunk.upper() or
            'CÁC BƯỚC' in chunk.upper() or
            'CHUẨN BỊ' in chunk.upper()
        )
        if not has_structure:
            continue
        
        # Extract title
        lines = chunk.split('\n')
        title_lines = [l.strip() for l in lines[:3] if l.strip()]
        title = ' '.join(title_lines)[:150]
        
        chunks.append({
            "text":          chunk,
            "title":         title,
            "source":        "Quy trình ICU — BYT VN 2014",
            "language":      "vi",
            "type":          "procedure",
            "has_chi_dinh":  'CHỈ ĐỊNH' in chunk.upper(),
            "has_chong_chi": 'CHỐNG CHỈ ĐỊNH' in chunk.upper(),
            "has_tien_hanh": 'TIẾN HÀNH' in chunk.upper(),
            "char_count":    len(chunk)
        })
    
    return chunks

# Run
with open("d:\\VinUni\\VSF\\week 2\\data\\quy_trinh_icu_vn.md", encoding="utf-8") as f:
    raw = f.read()

cleaned = clean_markdown(raw)
chunks  = chunk_procedures(cleaned)

print(f"Total procedures chunked: {len(chunks)}")
print(f"Expected: ~232")
print()
for c in chunks[:5]:
    print(f"Title: {c['title'][:60]}")
    print(f"  Chars: {c['char_count']} | CHỈ ĐỊNH: {c['has_chi_dinh']} | CHỐNG CHỈ ĐỊNH: {c['has_chong_chi']}")
    print()