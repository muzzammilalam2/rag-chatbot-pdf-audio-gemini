from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .embeddings import embed_texts
from .utils import DocChunk


@dataclass
class VectorStore:
    db_dir: Path
    collection: str

    # persisted arrays
    ids: list[str]
    documents: list[str]
    metadatas: list[dict]
    embeddings: np.ndarray  # shape: (n, d), normalized


def _col_dir(db_dir: Path, collection: str) -> Path:
    return db_dir / collection


def open_store(db_dir: Path, collection: str) -> VectorStore:
    db_dir.mkdir(parents=True, exist_ok=True)
    col_dir = _col_dir(db_dir, collection)
    col_dir.mkdir(parents=True, exist_ok=True)

    ids_path = col_dir / "ids.json"
    docs_path = col_dir / "documents.jsonl"
    metas_path = col_dir / "metadatas.jsonl"
    embs_path = col_dir / "embeddings.npy"

    if embs_path.exists() and ids_path.exists() and docs_path.exists() and metas_path.exists():
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        documents = [json.loads(l)["text"] for l in docs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        metadatas = [json.loads(l) for l in metas_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        embeddings = np.load(embs_path)
        if len(ids) != len(documents) or len(ids) != len(metadatas) or embeddings.shape[0] != len(ids):
            raise RuntimeError("Vector store files are inconsistent. Delete the collection folder and re-ingest.")
        return VectorStore(db_dir=db_dir, collection=collection, ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    return VectorStore(db_dir=db_dir, collection=collection, ids=[], documents=[], metadatas=[], embeddings=np.zeros((0, 1), dtype=np.float32))


def persist(store: VectorStore) -> None:
    col_dir = _col_dir(store.db_dir, store.collection)
    col_dir.mkdir(parents=True, exist_ok=True)

    (col_dir / "ids.json").write_text(json.dumps(store.ids, ensure_ascii=False, indent=2), encoding="utf-8")

    (col_dir / "documents.jsonl").write_text(
        "\n".join(json.dumps({"text": t}, ensure_ascii=False) for t in store.documents) + ("\n" if store.documents else ""),
        encoding="utf-8",
    )
    (col_dir / "metadatas.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in store.metadatas) + ("\n" if store.metadatas else ""),
        encoding="utf-8",
    )
    np.save(col_dir / "embeddings.npy", store.embeddings.astype(np.float32, copy=False))


def upsert_chunks(store: VectorStore, chunks: Iterable[DocChunk], *, batch_size: int = 96) -> int:
    # naive upsert: if id exists, replace; else append
    id_to_idx = {i: idx for idx, i in enumerate(store.ids)}
    added_or_updated = 0

    batch: list[DocChunk] = []

    def flush(b: list[DocChunk]) -> None:
        nonlocal added_or_updated
        if not b:
            return
        texts = [c.text for c in b]
        embs = np.array(embed_texts(texts), dtype=np.float32)
        if embs.ndim != 2:
            raise RuntimeError("Embedding model returned invalid shape.")

        # normalize to unit vectors
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        embs = embs / norms

        for c, e in zip(b, embs, strict=False):
            if c.id in id_to_idx:
                idx = id_to_idx[c.id]
                store.documents[idx] = c.text
                store.metadatas[idx] = c.metadata
                store.embeddings[idx] = e
            else:
                store.ids.append(c.id)
                store.documents.append(c.text)
                store.metadatas.append(c.metadata)
                if store.embeddings.shape[0] == 0:
                    store.embeddings = e.reshape(1, -1)
                else:
                    store.embeddings = np.vstack([store.embeddings, e.reshape(1, -1)])
                id_to_idx[c.id] = len(store.ids) - 1
            added_or_updated += 1

    for ch in chunks:
        batch.append(ch)
        if len(batch) >= batch_size:
            flush(batch)
            batch = []

    flush(batch)
    persist(store)
    return added_or_updated


def query(store: VectorStore, question: str, *, top_k: int = 6) -> dict:
    if store.embeddings.shape[0] == 0:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    q = np.array(embed_texts([question])[0], dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-12)

    # cosine distance = 1 - cosine_similarity
    sims = store.embeddings @ q.reshape(-1, 1)
    sims = sims.reshape(-1)
    k = min(top_k, sims.shape[0])
    idxs = np.argpartition(-sims, kth=k - 1)[:k]
    idxs = idxs[np.argsort(-sims[idxs])]

    docs = [store.documents[i] for i in idxs.tolist()]
    metas = [store.metadatas[i] for i in idxs.tolist()]
    dists = [(1.0 - float(sims[i])) for i in idxs.tolist()]

    return {"documents": [docs], "metadatas": [metas], "distances": [dists]}

