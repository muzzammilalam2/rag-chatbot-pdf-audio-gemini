from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def iter_files(root: Path, exts: set[str]) -> Iterable[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return sorted(out)


def env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip()


@dataclass(frozen=True)
class DocChunk:
    id: str
    text: str
    metadata: dict

