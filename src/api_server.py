from __future__ import annotations

import os
import time
import uuid
import json
import logging
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .llm import answer_question
from .vectorstore import open_store, query
from .agent import RagAgent


load_dotenv()
logger = logging.getLogger("rag-api")


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


DB_DIR = _env_path("RAG_DB_DIR", "chroma_db")
COLLECTION = os.environ.get("RAG_COLLECTION", "genai_databases")
TOP_K = int(os.environ.get("RAG_TOP_K", "6"))
MODEL_ID = os.environ.get("RAG_MODEL_ID", "rag-gemini")
SHOW_SOURCES = os.environ.get("RAG_SHOW_SOURCES", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
AGENT_ENABLED = os.environ.get("RAG_AGENT", "false").strip().lower() in {"1", "true", "yes", "y", "on"}


app = FastAPI(title="RAG OpenAI-Compatible API", version="0.1.0")


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str = ""


class ChatCompletionsRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = False
    temperature: float | None = None


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "local"


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "db_dir": str(DB_DIR),
        "collection": COLLECTION,
        "top_k": TOP_K,
        "model_id": MODEL_ID,
    }


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {"object": "list", "data": [ModelCard(id=MODEL_ID).model_dump()]}


def _extract_question(messages: list[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user" and (m.content or "").strip():
            return m.content.strip()
    return ""


def _extract_text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text" and isinstance(p.get("text"), str):
                    parts.append(p["text"])
                elif isinstance(p.get("content"), str):
                    parts.append(p["content"])
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join([t for t in parts if t.strip()])
    if isinstance(content, dict):
        v = content.get("text") or content.get("content")
        return v if isinstance(v, str) else ""
    return str(content)


def _extract_question_from_payload(payload: dict[str, Any]) -> str:
    messages = payload.get("messages")
    if isinstance(messages, list):
        for m in reversed(messages):
            if not isinstance(m, dict):
                continue
            if m.get("role") != "user":
                continue
            content = _extract_text_content(m.get("content"))
            if content.strip():
                return content.strip()
    return ""


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
            try:
                lines.append(f"- media: {m.get('source_path')} ({float(start):.1f}s–{float(end):.1f}s)")
            except Exception:
                lines.append(f"- media: {m.get('source_path')} ({start}s–{end}s)")
        else:
            lines.append(f"- {m}")
    return "\n".join(lines)


def _with_sources(content: str, metas: list[dict]) -> str:
    if not SHOW_SOURCES or not metas:
        return content
    src = _format_sources(metas).strip()
    if not src:
        return content
    return f"{content}\n\nSources:\n{src}"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> dict[str, Any]:
    body = await request.body()
    if not body:
        logger.warning("chat.completions: empty request body")
        raise HTTPException(status_code=400, detail="Empty request body.")

    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        logger.warning("chat.completions: invalid JSON (%d bytes)", len(body))
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object.")

    stream = bool(payload.get("stream", False))

    question = _extract_question_from_payload(payload)
    if not question:
        logger.warning("chat.completions: missing user message. payload keys=%s", sorted(payload.keys()))
        raise HTTPException(status_code=400, detail="No user message found in 'messages'.")

    metas: list[dict] = []
    agent_meta: dict[str, Any] | None = None

    if AGENT_ENABLED:
        try:
            agent = RagAgent(db_dir=DB_DIR, collection=COLLECTION, top_k=TOP_K)
            out = agent.answer(question=question)
            content = (out.get("answer") or "").strip()
            metas = out.get("sources") or []
            agent_meta = {
                "plan": out.get("plan"),
                "tools_used": out.get("tools_used"),
                "judge": out.get("judge"),
                "metrics": out.get("metrics"),
            }
        except Exception as e:
            logger.exception("Agent failed")
            err = str(e)
            if "RESOURCE_EXHAUSTED" in err or "quota" in err.lower() or "429" in err:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        "Gemini quota exceeded for current model. "
                        "Try again after cooldown, switch GEMINI_MODEL, disable agent mode, or upgrade billing tier."
                    ),
                )
            raise HTTPException(status_code=500, detail=f"Agent failed: {type(e).__name__}")
    else:
        store = open_store(DB_DIR, COLLECTION)
        res = query(store, question, top_k=TOP_K)
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        context_chunks = [d for d in docs if d]

        if not context_chunks:
            content = "I don't know. No relevant context was found in the vector DB."
        else:
            try:
                content = answer_question(question=question, context_chunks=context_chunks)
            except Exception as e:
                logger.exception("LLM call failed")
                err = str(e)
                if "RESOURCE_EXHAUSTED" in err or "quota" in err.lower() or "429" in err:
                    raise HTTPException(
                        status_code=429,
                        detail=(
                            "Gemini quota exceeded for current model. "
                            "Try again after cooldown, switch GEMINI_MODEL, or upgrade billing tier."
                        ),
                    )
                raise HTTPException(status_code=500, detail=f"LLM call failed: {type(e).__name__}")

    content = _with_sources(content, metas)
    if AGENT_ENABLED and agent_meta:
        plan = agent_meta.get("plan")
        tools_used = agent_meta.get("tools_used") or []
        judge = agent_meta.get("judge") or {}
        overall = judge.get("overall")
        content += "\n\n---\n\nAgent:\n"
        if plan:
            content += f"- Plan: {str(plan).strip()}\n"
        if tools_used:
            content += f"- Tools used: {', '.join(str(t) for t in tools_used)}\n"
        if overall is not None:
            content += f"- Self-score (overall): {overall}\n"

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    resp: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion" if not stream else "chat.completion.chunk",
        "created": created,
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content} if not stream else None,
                "delta": {"role": "assistant", "content": content} if stream else None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
        "rag": {
            "top_k": TOP_K,
            "sources": metas,
        },
    }

    if agent_meta:
        resp["agent"] = agent_meta

    if not stream:
        # Clean up fields not used in non-streaming mode
        resp["choices"][0].pop("delta", None)
        return resp

    # Streaming: emit minimal SSE events OpenAI-style.
    def sse() -> Any:
        chunk = dict(resp)
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


def main() -> None:
    import uvicorn

    uvicorn.run("src.api_server:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), reload=True)


if __name__ == "__main__":
    main()

