from __future__ import annotations

import hashlib
import re
from typing import Iterable

from .utils import DocChunk, normalize_text


SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _stable_id(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()[:32]


def chunk_text(
    *,
    text: str,
    base_metadata: dict,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    id_prefix: str,
) -> list[DocChunk]:
    text = normalize_text(text)
    if not text:
        return []

    sentences = SENT_SPLIT_RE.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if not buf:
            return
        joined = normalize_text(" ".join(buf))
        if joined:
            chunks.append(joined)
        buf = []
        buf_len = 0

    for sent in sentences:
        if not buf:
            buf.append(sent)
            buf_len = len(sent)
            continue

        # if adding would exceed size, flush then start new
        if buf_len + 1 + len(sent) > chunk_size:
            flush()
            buf.append(sent)
            buf_len = len(sent)
        else:
            buf.append(sent)
            buf_len += 1 + len(sent)

    flush()

    # add overlap by reusing tail of previous chunk
    out: list[DocChunk] = []
    prev_tail = ""
    for idx, c in enumerate(chunks):
        if prev_tail:
            c = normalize_text(prev_tail + " " + c)

        meta = dict(base_metadata)
        meta["chunk_index"] = idx

        cid = _stable_id(id_prefix, str(meta), c)
        out.append(DocChunk(id=cid, text=c, metadata=meta))

        if chunk_overlap > 0:
            prev_tail = c[-chunk_overlap:]
        else:
            prev_tail = ""

    return out


def chunk_many(items: Iterable[tuple[str, dict]], *, chunk_size: int, chunk_overlap: int, id_prefix: str) -> list[DocChunk]:
    out: list[DocChunk] = []
    for text, meta in items:
        out.extend(
            chunk_text(
                text=text,
                base_metadata=meta,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                id_prefix=id_prefix,
            )
        )
    return out

