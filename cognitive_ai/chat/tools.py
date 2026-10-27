"""Local tools Complete AI can invoke from ChatGPT-class chat.

The LLM (or a symbolic planner) may request a tool with a line like:

  TOOL finderbay {"text": "Show unpaid invoices for Project A"}
  TOOL vision {"task": "count_people", "image": "./crowd_people.jpg"}
  TOOL coding {"text": "generate function add(a, b)"}
  TOOL status {}
  TOOL alerts {"project_id": "project_a"}

Complete AI executes the tool locally and feeds the result back.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional


TOOL_LINE = re.compile(
    r"^\s*TOOL\s+(finderbay|vision|coding|status|alerts|memory|voice|computer|learn|plan)\s+(\{.*\})\s*$",
    re.I | re.M,
)


def extract_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for m in TOOL_LINE.finditer(text or ""):
        name = m.group(1).lower()
        try:
            args = json.loads(m.group(2))
        except json.JSONDecodeError:
            args = {"raw": m.group(2)}
        calls.append({"name": name, "args": args, "span": m.group(0)})
    return calls


def strip_tool_lines(text: str) -> str:
    return TOOL_LINE.sub("", text or "").strip()


class ToolRouter:
    """Dispatch named tools against a CompleteAI instance."""

    def __init__(self, ai: Any) -> None:
        self.ai = ai
        self.handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "finderbay": self._finderbay,
            "vision": self._vision,
            "coding": self._coding,
            "status": self._status,
            "alerts": self._alerts,
            "memory": self._memory,
        }

    def run(self, name: str, args: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        args = args or {}
        fn = self.handlers.get(name)
        if not fn:
            return {"ok": False, "error": f"unknown_tool:{name}"}
        try:
            out = fn(args)
            out.setdefault("ok", True)
            out.setdefault("tool", name)
            return out
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "tool": name, "error": str(exc)}

    def _finderbay(self, args: dict[str, Any]) -> dict[str, Any]:
        text = args.get("text") or args.get("query") or args.get("ask") or ""
        if not text:
            return {"ok": False, "error": "text_required"}
        result = self.ai.finderbay.handle(str(text))
        return {
            "ok": True,
            "text": result.get("text"),
            "operation": (result.get("frame") or {}).get("operation"),
            "structured": result.get("structured"),
        }

    def _vision(self, args: dict[str, Any]) -> dict[str, Any]:
        task = args.get("task") or "understand_scene"
        image = args.get("image") or args.get("image_path") or args.get("path")
        if task == "vision_status" or task == "status":
            return {"ok": True, **self.ai.vision.status()}
        if task in ("explain_object", "understand_object"):
            label = str(args.get("label") or args.get("object") or "")
            if not label:
                return {"ok": False, "error": "label_required"}
            return {"ok": True, **self.ai.vision.explain_object(label, image=image)}
        if not image:
            return {"ok": False, "error": "image_path_required"}
        search_unknown = args.get("search_unknown", True)
        modality = args.get("modality")
        return {
            "ok": True,
            **self.ai.vision.analyze(
                image,
                task=task,
                search_unknown=bool(search_unknown),
                modality=modality,
            ),
        }

    def _coding(self, args: dict[str, Any]) -> dict[str, Any]:
        text = args.get("text") or args.get("instruction") or ""
        if not text:
            return {"ok": False, "error": "instruction_required"}
        return dict(self.ai.coding.handle(str(text)))

    def _status(self, _args: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, **self.ai.status()}

    def _alerts(self, args: dict[str, Any]) -> dict[str, Any]:
        project = args.get("project_id")
        return {"ok": True, **self.ai.finderbay.alerts.summary(project_id=project)}

    def _memory(self, args: dict[str, Any]) -> dict[str, Any]:
        action = (args.get("action") or "list").lower()
        if action == "remember":
            key = str(args.get("key") or "note")
            val = str(args.get("value") or "")
            self.ai.remember(key, val)
            return {"ok": True, "remembered": {key: val}}
        if action == "get":
            key = str(args.get("key") or "")
            return {"ok": True, "value": self.ai.recall(key)}
        return {"ok": True, "memory": self.ai.memory_snapshot()}
