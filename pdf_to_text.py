from pypdf import PdfReader

# Change this to your PDF filename
pdf_path = "testing.pdf"

reader = PdfReader(pdf_path)

full_text = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        full_text += text + "\n"

# Save to documents.txt
with open("documents.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

print("PDF successfully converted to documents.txt")
