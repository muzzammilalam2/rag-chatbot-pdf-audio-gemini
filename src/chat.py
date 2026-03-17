from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from .llm import answer_question
from .vectorstore import open_store, query


def _format_sources(metas: list[dict]) -> str:
    lines: list[str] = []
    for m in metas:
        if not m:
            continue
        if m.get("source_type") == "pdf":
            lines.append(f"- pdf: {m.get('source_path')} (page {m.get('page_number')})")
        elif m.get("source_type") == "media":
            start = m.get("start_s")
            end = m.get("end_s")
            lines.append(f"- media: {m.get('source_path')} ({start:.1f}s–{end:.1f}s)")
        else:
            lines.append(f"- {m}")
    return "\n".join(lines)


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Chat with your RAG knowledge base.")
    ap.add_argument("--db_dir", type=Path, required=True)
    ap.add_argument("--collection", type=str, default="kb")
    ap.add_argument("--top_k", type=int, default=6)
    args = ap.parse_args()

    store = open_store(args.db_dir, args.collection)

    print("RAG chat ready. Type your question, or 'exit' to quit.")
    while True:
        q = input("\n> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        res = query(store, q, top_k=args.top_k)
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]

        context_chunks = [d for d in docs if d]
        if not context_chunks:
            print("No relevant context found in the vector DB.")
            continue

        ans = answer_question(question=q, context_chunks=context_chunks)
        print("\nAnswer:\n" + ans)
        if metas:
            print("\nSources:\n" + _format_sources(metas))


if __name__ == "__main__":
    main()

