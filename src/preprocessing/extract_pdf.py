from markitdown import MarkItDown

md = MarkItDown()

# Convert từng file
result_icu = md.convert("d:\\VinUni\\VSF\\data\\surviving_sepsis_campaign__international.21.pdf")

# Save ra file để inspect
with open("ssc_2021.md", "w", encoding="utf-8") as f:
    f.write(result_icu.text_content)

print(f"ICU: {len(result_icu.text_content)} chars")
