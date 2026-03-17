from __future__ import annotations

from google import genai

from .utils import env


def get_client() -> genai.Client:
    api_key = env("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key)

