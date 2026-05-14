from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from google.genai.types import GenerateContentConfig

from .agent_tools import (
    ToolResult,
    code_analyzer,
    default_workspace_root,
    file_inspector,
    knowledge_search,
    list_directory,
    summarize_text,
)
from .gemini_client import get_client
from .utils import env


ToolName = Literal[
    "knowledge_search",
    "file_inspector",
    "code_analyzer",
    "list_directory",
    "summarize_text",
    "final",
]


@dataclass
class AgentStep:
    tool: ToolName
    tool_input: dict[str, Any]
    observation: str
    ok: bool


def _json_dumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, indent=2, sort_keys=True)


def _safe_json_loads(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    # Try to locate a JSON object in the response.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found.")
    return json.loads(text[start : end + 1])


class RagAgent:
    def __init__(self, *, db_dir: Path, collection: str, top_k: int) -> None:
        self.db_dir = db_dir
        self.collection = collection
        self.top_k = top_k
        self.workspace_root = default_workspace_root()
        self.model = env("GEMINI_MODEL", "gemini-3-flash-preview") or "gemini-3-flash-preview"
        self.client = get_client()

    def _decide_next(self, *, question: str, steps: list[AgentStep]) -> dict[str, Any]:
        tools_spec = [
            {
                "name": "knowledge_search",
                "input_schema": {"question": "string", "top_k": "int (optional)"},
                "purpose": "Semantic retrieval over the ingested knowledge base.",
            },
            {
                "name": "list_directory",
                "input_schema": {"path": "string (relative path, default '.')"},
                "purpose": "List a directory under the workspace root.",
            },
            {
                "name": "file_inspector",
                "input_schema": {"path": "string (relative file path)"},
                "purpose": "Read a text file under the workspace root.",
            },
            {
                "name": "code_analyzer",
                "input_schema": {"path": "string (relative .py path)"},
                "purpose": "Python AST summary: imports/classes/functions.",
            },
            {
                "name": "summarize_text",
                "input_schema": {"text": "string", "max_bullets": "int (optional)"},
                "purpose": "Heuristic key-point extraction for long text.",
            },
            {"name": "final", "input_schema": {}, "purpose": "Stop and produce final answer."},
        ]

        history = [
            {
                "tool": s.tool,
                "ok": s.ok,
                "tool_input": s.tool_input,
                "observation_preview": (s.observation[:400] + ("..." if len(s.observation) > 400 else "")),
            }
            for s in steps
        ]

        system = (
            "You are an autonomous assistant running in a tool-using environment.\n"
            "Decide the NEXT tool to call to answer the user's question.\n"
            "Return ONLY a JSON object with keys: tool, tool_input, short_plan.\n"
            "Rules:\n"
            "- Do not include chain-of-thought. short_plan must be 1-3 bullet points, high-level.\n"
            "- Prefer knowledge_search first for factual questions.\n"
            "- Use file_inspector/list_directory/code_analyzer only when needed.\n"
            "- Choose final only when you can answer completely.\n"
        )

        user = _json_dumps(
            {
                "question": question,
                "available_tools": tools_spec,
                "history": history,
            }
        )

        resp = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config=GenerateContentConfig(system_instruction=system, temperature=0.1),
        )

        data = _safe_json_loads(resp.text or "")
        return data

    def _judge(self, *, question: str, answer: str, sources: list[dict]) -> dict[str, Any]:
        system = (
            "You are an evaluator. Score the assistant answer for the given question.\n"
            "Return ONLY JSON with keys: relevance, accuracy, clarity, depth, completeness, overall.\n"
            "Each score is 0.0-1.0. overall should be the average.\n"
        )
        user = _json_dumps({"question": question, "answer": answer, "sources": sources[:12]})
        resp = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config=GenerateContentConfig(system_instruction=system, temperature=0.0),
        )
        try:
            return _safe_json_loads(resp.text or "")
        except Exception:
            return {"overall": 0.0}

    def _run_tool(self, tool: str, tool_input: dict[str, Any]) -> tuple[ToolResult, list[dict]]:
        tool = (tool or "").strip()
        tool_input = tool_input or {}

        if tool == "knowledge_search":
            q = str(tool_input.get("question", "")).strip()
            k = tool_input.get("top_k", self.top_k)
            try:
                k_int = int(k)
            except Exception:
                k_int = self.top_k
            r = knowledge_search(db_dir=self.db_dir, collection=self.collection, question=q, top_k=k_int)
            return r, (r.meta or {}).get("sources", []) if r.meta else []

        if tool == "list_directory":
            p = str(tool_input.get("path", ".")).strip() or "."
            r = list_directory(workspace_root=self.workspace_root, rel_path=p)
            return r, []

        if tool == "file_inspector":
            p = str(tool_input.get("path", "")).strip()
            r = file_inspector(workspace_root=self.workspace_root, rel_path=p)
            return r, []

        if tool == "code_analyzer":
            p = str(tool_input.get("path", "")).strip()
            r = code_analyzer(workspace_root=self.workspace_root, rel_path=p)
            return r, []

        if tool == "summarize_text":
            t = str(tool_input.get("text", ""))
            mb = tool_input.get("max_bullets", 10)
            try:
                mb_int = int(mb)
            except Exception:
                mb_int = 10
            r = summarize_text(text=t, max_bullets=mb_int)
            return r, []

        return ToolResult(ok=False, output=f"Unknown tool: {tool}"), []

    def answer(self, *, question: str, max_steps: int = 6) -> dict[str, Any]:
        started = time.time()
        steps: list[AgentStep] = []
        all_sources: list[dict] = []
        plan_preview: str | None = None

        for _ in range(max_steps):
            decision = self._decide_next(question=question, steps=steps)
            tool = decision.get("tool", "knowledge_search")
            tool_input = decision.get("tool_input", {}) or {}
            plan_preview = decision.get("short_plan") or plan_preview

            if tool == "final":
                break

            res, sources = self._run_tool(str(tool), tool_input if isinstance(tool_input, dict) else {})
            if sources:
                all_sources.extend([s for s in sources if isinstance(s, dict)])
            steps.append(
                AgentStep(
                    tool=str(tool),  # type: ignore[arg-type]
                    tool_input=tool_input if isinstance(tool_input, dict) else {},
                    observation=res.output,
                    ok=res.ok,
                )
            )

            # If we already have KB context, usually we can finalize soon.
            if tool == "knowledge_search" and res.ok:
                break

        # Final synthesis from retrieved context (if any) + tool observations
        kb_context = ""
        for s in steps:
            if s.tool == "knowledge_search" and s.ok and "No relevant context" not in s.observation:
                kb_context = s.observation
                break

        synthesis_system = (
            "You are a helpful assistant. Answer the user's question.\n"
            "Use the provided context when available. If insufficient, say you don't know.\n"
            "Do not mention hidden reasoning. Keep it direct.\n"
        )
        synthesis_user = _json_dumps(
            {
                "question": question,
                "kb_context": kb_context,
                "tool_observations": [
                    {"tool": s.tool, "ok": s.ok, "observation": s.observation[:2500]} for s in steps if s.tool != "knowledge_search"
                ],
            }
        )

        synthesis = self.client.models.generate_content(
            model=self.model,
            contents=synthesis_user,
            config=GenerateContentConfig(system_instruction=synthesis_system, temperature=0.2),
        )
        answer = (synthesis.text or "").strip()

        # Judge score
        judge = self._judge(question=question, answer=answer, sources=all_sources)

        elapsed_ms = int((time.time() - started) * 1000)
        return {
            "answer": answer,
            "plan": plan_preview,
            "tools_used": [s.tool for s in steps],
            "sources": all_sources,
            "judge": judge,
            "metrics": {"steps": len(steps), "elapsed_ms": elapsed_ms},
        }

