import pdfplumber
import json


def read_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    
    return text
        

def generate_chunks(text: str, sample_name: str, chunk_size=100) -> list[tuple[str, str]]:
    overlap = chunk_size//5
    words = text.split()
    chunks = []
    
    i = 0
    chunk_count = 1
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append((chunk_text, chunk_count))
        chunk_count += 1
        i += (chunk_size - overlap)
        
    return chunks

def generate_chunks_from_pdf(pdf_path: str, chunk_size=200) -> list[tuple[str, str]]:
    text = read_pdf(pdf_path)
    return generate_chunks(text, pdf_path, chunk_size)