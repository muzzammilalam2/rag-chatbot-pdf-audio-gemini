from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm
from dotenv import load_dotenv

from .chunking import chunk_many
from .pdf_loader import extract_pdf_pages
from .transcribe import transcribe_media_dir
from .vectorstore import open_store, upsert_chunks


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Ingest PDF + media into a local vector store for RAG.")
    ap.add_argument("--pdf_dir", type=Path, required=True)
    ap.add_argument("--media_dir", type=Path, required=True)
    ap.add_argument("--db_dir", type=Path, required=True)
    ap.add_argument("--collection", type=str, default="kb")
    ap.add_argument(
        "--transcribe_model",
        type=str,
        default="gemini-3-flash-preview",
        help="Gemini model for transcription (used with generate_content on uploaded media)",
    )
    ap.add_argument(
        "--whisper_model",
        type=str,
        default=None,
        help="(deprecated) alias for --transcribe_model. Do not pass OpenAI model names like whisper-1.",
    )
    ap.add_argument("--chunk_size", type=int, default=1200)
    ap.add_argument("--chunk_overlap", type=int, default=200)
    ap.add_argument("--transcript_cache", type=Path, default=Path("cache/transcripts"))
    args = ap.parse_args()

    transcribe_model = args.transcribe_model
    if args.whisper_model:
        transcribe_model = args.whisper_model
    if transcribe_model.startswith("whisper-"):
        raise SystemExit(
            "You passed an OpenAI Whisper model name (e.g. whisper-1). "
            "For Gemini transcription, use a Gemini model such as 'gemini-3-flash-preview'."
        )

    store = open_store(args.db_dir, args.collection)

    # PDF
    pdf_files = [p for p in sorted(args.pdf_dir.rglob("*.pdf")) if p.is_file()]
    pdf_items: list[tuple[str, dict]] = []
    for pdf_path in tqdm(pdf_files, desc="Extracting PDF", unit="pdf"):
        pages = extract_pdf_pages(pdf_path)
        for page in pages:
            pdf_items.append(
                (
                    page.text,
                    {
                        "source_type": "pdf",
                        "source_path": page.pdf_path,
                        "page_number": page.page_number,
                    },
                )
            )

    # Media transcripts
    segs = transcribe_media_dir(
        args.media_dir,
        args.transcript_cache,
        whisper_model=transcribe_model,
    )

    media_items: list[tuple[str, dict]] = []
    for s in segs:
        media_items.append(
            (
                s.text,
                {
                    "source_type": "media",
                    "source_path": s.media_path,
                    "start_s": s.start_s,
                    "end_s": s.end_s,
                },
            )
        )

    chunks = []
    if pdf_items:
        chunks.extend(
            chunk_many(
                pdf_items,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                id_prefix="pdf",
            )
        )
    if media_items:
        chunks.extend(
            chunk_many(
                media_items,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                id_prefix="media",
            )
        )

    if not chunks:
        raise SystemExit("No text extracted/transcribed. Check your input paths and file types.")

    n = upsert_chunks(store, chunks)
    print(f"Upserted {n} chunks into collection '{args.collection}' at '{args.db_dir}'.")


if __name__ == "__main__":
    main()

