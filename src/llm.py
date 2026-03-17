from __future__ import annotations

from google.genai.types import GenerateContentConfig

from .gemini_client import get_client
from .utils import env


def answer_question(*, question: str, context_chunks: list[str]) -> str:
    model = env("GEMINI_MODEL", "gemini-3-flash-preview")
    client = get_client()

    context = "\n\n---\n\n".join(context_chunks).strip()
    system = (
        "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
        "If the context is insufficient, say you don't know and suggest what to look up."
    )
    user = f"Context:\n{context}\n\nQuestion:\n{question}"

    resp = client.models.generate_content(
        model=model,
        contents=user,
        config=GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,
        ),
    )
    return (resp.text or "").strip()

