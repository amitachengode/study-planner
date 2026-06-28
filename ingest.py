print("Importing dependencies...")

from src.pdf_processor import generate_chunks_from_pdf
from src.database import Database
import os
if not os.path.exists("./chroma"):
    print("Creating chroma database...")
    db = Database(name="ktu_syllabus", database_path="./chroma")
    pdf_path = r"data\syllabus.pdf"
    print("Generating chunks...")
    chunks = generate_chunks_from_pdf(pdf_path)
    print("Adding chunks to database...")
    db.add_chunks(chunks, "ktu_syllabus")
    print("Done!")
else:
    print("Chroma database already exists. Skipping...")