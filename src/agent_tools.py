from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .vectorstore import open_store, query


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: str
    meta: dict[str, Any] | None = None


def _safe_join(root: Path, rel_path: str) -> Path:
    rel = Path(rel_path)
    if rel.is_absolute():
        raise ValueError("Absolute paths are not allowed.")
    p = (root / rel).resolve()
    root_resolved = root.resolve()
    if root_resolved not in p.parents and p != root_resolved:
        raise ValueError("Path escapes workspace root.")
    return p


def knowledge_search(*, db_dir: Path, collection: str, question: str, top_k: int) -> ToolResult:
    store = open_store(db_dir, collection)
    res = query(store, question, top_k=top_k)
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    context_chunks = [d for d in docs if d]
    if not context_chunks:
        return ToolResult(ok=True, output="No relevant context found in the vector DB.", meta={"sources": metas or []})
    text = "\n\n---\n\n".join(context_chunks[:top_k]).strip()
    return ToolResult(ok=True, output=text, meta={"sources": metas or []})


def list_directory(*, workspace_root: Path, rel_path: str = ".", max_entries: int = 200) -> ToolResult:
    p = _safe_join(workspace_root, rel_path)
    if not p.exists():
        return ToolResult(ok=False, output=f"Path not found: {rel_path}")
    if not p.is_dir():
        return ToolResult(ok=False, output=f"Not a directory: {rel_path}")

    entries: list[str] = []
    for child in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        name = child.name + ("/" if child.is_dir() else "")
        entries.append(name)
        if len(entries) >= max_entries:
            break
    return ToolResult(ok=True, output="\n".join(entries))


def file_inspector(
    *,
    workspace_root: Path,
    rel_path: str,
    max_chars: int = 12_000,
) -> ToolResult:
    p = _safe_join(workspace_root, rel_path)
    if not p.exists():
        return ToolResult(ok=False, output=f"File not found: {rel_path}")
    if not p.is_file():
        return ToolResult(ok=False, output=f"Not a file: {rel_path}")
    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".mkv", ".mov", ".mp3", ".wav"}:
        return ToolResult(ok=False, output="Binary/media files are not supported by file_inspector.")

    data = p.read_text(encoding="utf-8", errors="replace")
    if len(data) > max_chars:
        data = data[:max_chars] + "\n\n[TRUNCATED]\n"
    return ToolResult(ok=True, output=data)


def code_analyzer(*, workspace_root: Path, rel_path: str) -> ToolResult:
    p = _safe_join(workspace_root, rel_path)
    if not p.exists() or not p.is_file():
        return ToolResult(ok=False, output=f"File not found: {rel_path}")
    if p.suffix.lower() != ".py":
        return ToolResult(ok=False, output="code_analyzer only supports Python (.py) files.")

    src = p.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src, filename=str(p))
    except SyntaxError as e:
        return ToolResult(ok=False, output=f"SyntaxError: {e}")

    imports: list[str] = []
    classes: list[str] = []
    functions: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.append(n.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for n in node.names:
                imports.append(f"{mod}:{n.name}")
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.AsyncFunctionDef):
            functions.append(node.name)

    out = []
    out.append(f"file: {rel_path}")
    out.append(f"imports: {len(imports)}")
    out.extend([f"  - {i}" for i in sorted(set(imports))[:200]])
    out.append(f"classes: {len(classes)}")
    out.extend([f"  - {c}" for c in sorted(set(classes))[:200]])
    out.append(f"functions: {len(functions)}")
    out.extend([f"  - {f}" for f in sorted(set(functions))[:200]])
    return ToolResult(ok=True, output="\n".join(out))


def summarize_text(*, text: str, max_bullets: int = 10) -> ToolResult:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    bullets: list[str] = []
    for ln in lines:
        if len(ln) < 8:
            continue
        bullets.append(ln if len(ln) <= 180 else ln[:177] + "...")
        if len(bullets) >= max_bullets:
            break
    if not bullets:
        bullets = [text.strip()[:300] + ("..." if len(text.strip()) > 300 else "")]
    return ToolResult(ok=True, output="\n".join(f"- {b}" for b in bullets))


def default_workspace_root() -> Path:
    # When running in Docker, we mount the repo read-only here.
    return Path(os.environ.get("AGENT_WORKSPACE", "/workspace")).resolve()

