from markitdown import MarkItDown

md = MarkItDown()

# Convert từng file
result_icu = md.convert("d:\\VinUni\\VSF\\week 2\\data\\quy_trinh_icu_vn.pdf")
result_icd = md.convert("d:\\VinUni\\VSF\\week 2\\data\\icd-10_vn.pdf")

# Save ra file để inspect
with open("quy_trinh_icu_vn.md", "w", encoding="utf-8") as f:
    f.write(result_icu.text_content)

with open("icd-10_vn.md", "w", encoding="utf-8") as f:
    f.write(result_icd.text_content)

print(f"ICU: {len(result_icu.text_content)} chars")
print(f"ICD: {len(result_icd.text_content)} chars")