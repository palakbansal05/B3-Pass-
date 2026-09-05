import pdfplumber
import os

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def main():
    pdf_folder = "corpus/papers_pdfs"
    output_path = "corpus/manual_papers.txt"

    all_text = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            print(f"Extracting: {filename}")
            path = os.path.join(pdf_folder, filename)
            text = extract_text_from_pdf(path)
            all_text.append(f"--- {filename} ---\n{text}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_text))
    print(f"Saved extracted text to {output_path}")

if __name__ == "__main__":
    main()