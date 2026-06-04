import re

def check_extraction_quality(md_text, source_name):
    results = {}
    
    # 1. Tỉ lệ ký tự tiếng Việt có dấu
    vi_chars = 'àáảãạăắặằẳẵâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ'
    vi_count = sum(1 for c in md_text if c in vi_chars)
    total_alpha = sum(1 for c in md_text if c.isalpha())
    results['vi_char_ratio'] = vi_count / total_alpha if total_alpha > 0 else 0
    
    # 2. Số headings được nhận dạng
    headings = re.findall(r'^#{1,4}\s+.+', md_text, re.MULTILINE)
    results['heading_count'] = len(headings)
    
    # 3. Số bảng được giữ lại
    tables = re.findall(r'\|.+\|', md_text)
    results['table_rows'] = len(tables)
    
    # 4. Tỉ lệ dòng có nội dung (không phải dòng trống)
    lines = md_text.split('\n')
    non_empty = [l for l in lines if l.strip()]
    results['content_density'] = len(non_empty) / len(lines)
    
    # 5. Garbage text detection
    garbage_patterns = [
        r'[^\x00-\x7F\u00C0-\u024F\u1E00-\u1EFF]{5,}',  # Non-latin sequences
        r'(\w)\1{4,}',  # Repeated chars: "aaaaa"
        r'–\s*\d+\s*–',  # Page numbers: "– 38 –"
    ]
    garbage_count = sum(
        len(re.findall(p, md_text))
        for p in garbage_patterns
    )
    results['garbage_count'] = garbage_count
    
    # 6. Keyword presence check (ICU-specific)
    keywords = ['quy trình', 'chỉ định', 'chống chỉ định', 
                'tiến hành', 'theo dõi', 'tai biến']
    results['keyword_hits'] = sum(
        1 for kw in keywords 
        if kw.lower() in md_text.lower()
    )
    
    # Print report
    print(f"\n=== Quality Report: {source_name} ===")
    print(f"Vietnamese char ratio:  {results['vi_char_ratio']:.1%}  {'✅' if results['vi_char_ratio'] > 0.05 else '❌'}")
    print(f"Headings detected:      {results['heading_count']}       {'✅' if results['heading_count'] > 10 else '⚠️'}")
    print(f"Table rows preserved:   {results['table_rows']}       {'✅' if results['table_rows'] > 5 else '⚠️'}")
    print(f"Content density:        {results['content_density']:.1%}  {'✅' if results['content_density'] > 0.4 else '⚠️'}")
    print(f"Garbage text found:     {results['garbage_count']}       {'✅' if results['garbage_count'] < 10 else '❌'}")
    print(f"ICU keywords found:     {results['keyword_hits']}/6     {'✅' if results['keyword_hits'] >= 4 else '❌'}")
    
    return results

# Run
with open("quy_trinh_icu_vn.md", encoding="utf-8") as f:
    icu_md = f.read()

quality = check_extraction_quality(icu_md, "quy_trinh_icu_vn")