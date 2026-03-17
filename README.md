# RAG Chatbot (PDF + Audio) — Gemini + Local Vector Store

This project builds a functional **RAG** pipeline that:

- extracts text from a PDF
- transcribes lecture audio/video to text (Gemini)
- chunks + embeds the text
- stores embeddings in a **local vector store** (NumPy cosine search + persistence)
- retrieves relevant chunks and calls an LLM to answer questions

## 1) Setup

### Prereqs

- Python 3.10+ recommended
- `ffmpeg` (required for most audio/video formats)

On macOS:

```bash
brew install ffmpeg
```

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Put your knowledge base files in `data/`

Create this layout:

```text
data/
  pdf/
    Databases_for_GenAI.pdf
  media/
    lecture1.mp3
    lecture2.mp4
    ...
```

Notes:
- Video files (`.mp4`, `.mkv`, `.mov`) are fine — Gemini will process them via the Files API.
- If your lecture is split into many files, just drop them all in `data/media/`.

## 3) Configure Gemini

Set:
- `GEMINI_API_KEY` (required)
- `GEMINI_MODEL` (optional; defaults to `gemini-3-flash-preview`)
- `GEMINI_EMBED_MODEL` (optional; defaults to `gemini-embedding-001`)

Example:

```bash
export GEMINI_API_KEY="YOUR_KEY"
export GEMINI_MODEL="gemini-3-flash-preview"
export GEMINI_EMBED_MODEL="gemini-embedding-001"
```

## 4) Ingest (build the vector DB)

```bash
python -m src.ingest \
  --pdf_dir data/pdf \
  --media_dir data/media \
  --db_dir chroma_db \
  --collection genai_databases \
  --transcribe_model gemini-3-flash-preview \
  --chunk_size 1200 \
  --chunk_overlap 200
```

What this does:
- extracts all PDF text
- transcribes all audio/video
- chunks everything
- embeds chunks
- writes the local vector store to `chroma_db/`

## 5) Chat

```bash
python -m src.chat \
  --db_dir chroma_db \
  --collection genai_databases \
  --top_k 6
```

Then ask questions in the prompt. Type `exit` to quit.

## Troubleshooting

- **ffmpeg not found**: install it (see Setup).
- **Transcription slow**: use a faster Gemini model (and/or shorter media clips).
- **Poor retrieval**: increase `top_k`, or reduce chunk size for more focused passages.

