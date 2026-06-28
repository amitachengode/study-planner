# Study Planner RAG Assistant

This is a personal project for testing my ability in Retrieval-Augmented Generation (RAG). It is designed to ingest a syllabus PDF, embed its contents, and provide a Streamlit-based chat assistant that answers questions using the retrieved syllabus context.

## Features

- Ingests syllabus PDF content from `data/syllabus.pdf`
- Creates semantic embeddings and stores them in a local Chroma database
- Uses BM25 and vector search together for hybrid retrieval
- Sends retrieved context to a Gemini-based chat model for answer generation
- Streamlit UI for chat-based syllabus Q&A

## Files

- `main.py` - Streamlit application entrypoint
- `ingest.py` - Script to ingest the PDF syllabus and build the Chroma database
- `src/pdf_processor.py` - PDF reading and chunk generation logic
- `src/database.py` - Chroma persistence, embedding, and hybrid search
- `src/agent.py` - Gemini chat agent wrapper with fallback handling
- `pyproject.toml` - Project dependencies and metadata

## Setup

1. Create and activate your virtual environment

```bash
python -m venv .venv
source .venv/Scripts/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is not present, install from `pyproject.toml` or use `pip install chromadb google-genai google-generativeai pdfplumber python-dotenv rank-bm25 sentence-transformers streamlit torch torchvision`.

3. Add your Gemini API key as an environment variable:

```bash
set GEMINI_API_KEY=your_api_key
```

4. Ingest the syllabus PDF

```bash
python ingest.py
```

5. Run the Streamlit app

```bash
streamlit run main.py
```

## Notes

- This repository is a personal project created to explore and test Retrieval-Augmented Generation.
- The assistant is expected to answer using only the retrieved syllabus context and avoid hallucinations.
- The current ingestion script assumes the syllabus PDF is located at `data/syllabus.pdf`.

## Troubleshooting

- Ensure `data/syllabus.pdf` exists before running `ingest.py`
- If you encounter missing packages, install the required dependencies in the virtual environment
- If the Gemini API key is not found, set `GEMINI_API_KEY` in your environment or Streamlit secrets

## License

This project is for personal use and learning.
