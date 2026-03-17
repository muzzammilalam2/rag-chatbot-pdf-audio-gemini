from __future__ import annotations

import random
import time

from google.genai.errors import ClientError

from .gemini_client import get_client
from .utils import env


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = env("GEMINI_EMBED_MODEL", "gemini-embedding-001")
    client = get_client()

    # Gemini embed_content limits batches to 100 inputs.
    max_batch = 100
    max_retries = int(env("GEMINI_EMBED_MAX_RETRIES", "8") or "8")
    min_delay_s = float(env("GEMINI_EMBED_MIN_DELAY_S", "0.0") or "0.0")

    def parse_embeddings(res) -> list[list[float]]:
        out: list[list[float]] = []
        for e in res.embeddings:
            values = getattr(e, "values", None)
            if values is None and isinstance(e, dict):
                values = e.get("values")
            if values is None:
                raise RuntimeError("Unexpected embedding response format from Gemini SDK.")
            out.append(list(values))
        return out

    def embed_batch_with_retry(batch: list[str]) -> list[list[float]]:
        attempt = 0
        while True:
            try:
                res = client.models.embed_content(model=model, contents=batch)
                return parse_embeddings(res)
            except ClientError as e:
                # Handle rate limit / quota errors with backoff.
                status = getattr(e, "status_code", None)
                msg = str(e)
                if status == 429 or "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                    attempt += 1
                    if attempt > max_retries:
                        raise

                    # Best-effort parse "Please retry in Xs" from message; otherwise exponential backoff.
                    retry_s = None
                    for token in msg.split():
                        if token.endswith("s.") or token.endswith("s,") or token.endswith("s"):
                            t = token.rstrip(".,")
                            try:
                                if t.endswith("s"):
                                    retry_s = float(t[:-1])
                                    break
                            except ValueError:
                                pass
                    if retry_s is None:
                        retry_s = min(60.0, 1.5 ** attempt)

                    # Add small jitter to avoid thundering herd.
                    sleep_s = max(min_delay_s, retry_s) + random.uniform(0.0, 0.25)
                    time.sleep(sleep_s)
                    continue
                raise

    all_out: list[list[float]] = []
    for i in range(0, len(texts), max_batch):
        batch = texts[i : i + max_batch]
        all_out.extend(embed_batch_with_retry(batch))
        if min_delay_s > 0:
            time.sleep(min_delay_s)

    if len(all_out) != len(texts):
        raise RuntimeError(f"Embedding count mismatch: got {len(all_out)} embeddings for {len(texts)} texts.")
    return all_out

