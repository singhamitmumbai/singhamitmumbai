"""Complete AI — unified product (v1.5).

Product name: **Complete AI**

Upgrades in v1.5:
  • Multi-turn session memory
  • Tool-using chat (LLM can call finderbay/vision/coding/alerts/memory)
  • Built-in diagnostics
  • Stronger ChatGPT-class routing when an API key is present

human_level_agi is always false.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .agent import FinderbayAgentSystem
from .chat.llm_bridge import DEFAULT_SYSTEM, LLMChatBridge, llm_bridge_from_env
from .chat.tools import ToolRouter, extract_tool_calls, strip_tool_lines
from .config import SystemConfig
from .general.agent import GeneralNonLLMAI
from .runtime.persist import StateStore, default_store
from .version import __version__
from .vision import VisionSystem


TOOL_AWARE_SYSTEM = (
    DEFAULT_SYSTEM
    + "\n\nYou may call local Complete AI / Finderbay Assistant tools by emitting lines like:\n"
    'TOOL finderbay {"text": "Show unpaid invoices for Project A"}\n'
    'TOOL vision {"task": "understand_scene", "image": "./crowd_people.jpg"}\n'
    'TOOL vision {"task": "count_people", "image": "./crowd_people.jpg"}\n'
    'TOOL vision {"task": "explain_object", "label": "rebar"}\n'
    'TOOL coding {"text": "generate function add(a, b)"}\n'
    'TOOL alerts {"project_id": "project_a"}\n'
    'TOOL memory {"action": "remember", "key": "user_name", "value": "Alex"}\n'
    'TOOL voice {"action": "speak", "text": "Hello"}\n'
    'TOOL plan {"text": "reserve 10 cement"}\n'
    'TOOL computer {"action": "open_url", "url": "https://example.com"}\n'
    'TOOL learn {"action": "alias", "former": "Warehouse B", "alias": "Central Store"}\n'
    'TOOL jarvis {"text": "Prepare tomorrow material shortage report for Site-7"}\n'
    'TOOL status {}\n'
    "After tool results are provided, give the user a clear final answer. "
    "Do not invent Finderbay numbers — use TOOL finderbay / alerts. "
    "Computer control may be disabled unless the user enabled it."
)


@dataclass
class SessionTurn:
    role: str
    text: str
    route: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CompleteAI:
    """Complete AI product entrypoint."""

    PRODUCT_NAME = "Complete AI"

    def __init__(
        self,
        *,
        seed_demo: bool = True,
        config: Optional[SystemConfig] = None,
        enable_llm: Optional[bool] = None,
        llm: Optional[LLMChatBridge] = None,
        max_tool_rounds: int = 3,
        session_limit: int = 100,
        persist_path: Optional[str] = None,
        autosave: bool = True,
    ) -> None:
        self.version = __version__
        self.general = GeneralNonLLMAI(config=config, seed_demo=seed_demo)
        self.finderbay = self.general.task_ai
        self.vision = self.finderbay.vision if hasattr(self.finderbay, "vision") else VisionSystem()
        self.agent = FinderbayAgentSystem(seed_demo=seed_demo)
        self.coding = self.agent.coding
        self.max_tool_rounds = max_tool_rounds
        self.session_limit = session_limit
        self.session: list[SessionTurn] = []
        self.user_memory: dict[str, str] = {}
        self.autosave = autosave
        self.store: StateStore = default_store(persist_path)
        if llm is not None:
            self.llm = llm
            if enable_llm is not None:
                self.llm.enabled = enable_llm
        else:
            self.llm = llm_bridge_from_env()
            if enable_llm is not None:
                self.llm.enabled = enable_llm
        # Prefer tool-aware system prompt for ChatGPT-class chat
        if not getattr(self.llm, "_complete_ai_tools", False):
            self.llm.system_prompt = TOOL_AWARE_SYSTEM
            self.llm._complete_ai_tools = True  # type: ignore[attr-defined]
        self.tools = ToolRouter(self)
        self.assistant = None  # optional FinderbayAssistant attached later
        self._restore()

    def attach_assistant(self, assistant) -> None:
        """Wire Finderbay Assistant tools (voice/memory/computer/learn/plan)."""
        from .assistant.tools_ext import attach_assistant_tools

        self.assistant = assistant
        attach_assistant_tools(self, assistant)

    # --- primary API ----------------------------------------------------------

    def ask(self, text: str, *, mode: str = "auto") -> str:
        return self.handle(text, mode=mode)["text"]

    def handle(
        self,
        payload: Any,
        *,
        mode: str = "auto",
        modality: str = "text",
        context: Optional[dict] = None,
    ) -> dict[str, Any]:
        text = payload if isinstance(payload, str) else str(payload or "")
        mode = (mode or "auto").lower().strip()
        self._push("user", text, route=mode)

        if mode == "llm":
            out = self._llm_with_tools(text, default_route="llm")
            self._push("assistant", out.get("text") or "", route=out.get("route") or "llm", meta=out)
            return out

        if mode in ("auto", "coding") and (mode == "coding" or self._looks_coding(text)):
            raw = self.coding.handle(text)
            out = self._tag(raw, route="coding", llm=False)
            self._push("assistant", out.get("text") or "", route="coding", meta=out)
            return out

        if mode in ("auto", "vision") and (mode == "vision" or self._looks_vision(text)):
            raw = self.finderbay.handle(text, modality=modality, context=context)
            out = self._tag(raw, route="vision", llm=False)
            self._push("assistant", out.get("text") or "", route="vision", meta=out)
            return out

        if mode == "agent":
            report = self.agent.auto([text])
            out = self._tag(
                {
                    "text": (
                        f"Agent run finished: succeeded={report.get('succeeded')} "
                        f"failed={report.get('failed')}"
                    ),
                    "structured": report,
                    "faithful": True,
                },
                route="agent",
                llm=False,
            )
            self._push("assistant", out["text"], route="agent", meta=out)
            return out

        if mode == "finderbay" or (mode == "auto" and self._looks_finderbay(text)):
            raw = self.finderbay.handle(payload, modality=modality, context=context)
            if (raw.get("frame") or {}).get("operation") != "unsupported":
                out = self._tag(raw, route="finderbay", llm=False)
                self._push("assistant", out.get("text") or "", route="finderbay", meta=out)
                return out

        if mode == "symbolic":
            raw = self.general.handle(payload, modality=modality, context=context, force="auto")
            out = self._tag(raw, route=raw.get("route") or "symbolic", llm=False)
            self._push("assistant", out.get("text") or "", route=out["route"], meta=out)
            return out

        # General chat: ChatGPT-class + tools when available
        if mode == "auto" and self.llm.available and not self._looks_finderbay(text):
            out = self._llm_with_tools(text, default_route="llm_chat")
            self._push("assistant", out.get("text") or "", route=out.get("route") or "llm_chat", meta=out)
            return out

        raw = self.general.handle(payload, modality=modality, context=context, force="auto")
        route = raw.get("route") or "general"
        if self._is_unsupported(raw) and self.llm.available:
            out = self._llm_with_tools(text, default_route="llm_fallback")
            out.setdefault("structured", {})["prior_route"] = route
            self._push("assistant", out.get("text") or "", route=out.get("route") or "llm_fallback", meta=out)
            return out

        out = self._tag(raw, route=route, llm=False)
        self._push("assistant", out.get("text") or "", route=route, meta=out)
        return out

    def _llm_with_tools(self, text: str, *, default_route: str = "llm_chat") -> dict[str, Any]:
        """ChatGPT-class chat with optional local TOOL round-trips."""
        memory_hint = ""
        if self.user_memory:
            memory_hint = "\nKnown user memory: " + json.dumps(self.user_memory)
        prompt = text + memory_hint
        tool_trace: list[dict[str, Any]] = []
        llm_out = self.llm.chat(prompt, system=TOOL_AWARE_SYSTEM)
        if not llm_out.get("ok"):
            return self._finish_llm(llm_out, route=default_route, tools=tool_trace)

        reply = llm_out.get("text") or ""
        for _ in range(self.max_tool_rounds):
            calls = extract_tool_calls(reply)
            if not calls:
                break
            observations: list[str] = []
            for call in calls:
                result = self.tools.run(call["name"], call.get("args") or {})
                tool_trace.append({"call": call, "result": _shrink(result)})
                observations.append(
                    f"TOOL_RESULT {call['name']} "
                    + json.dumps(_shrink(result), default=str)[:4000]
                )
            follow = (
                "Tool results:\n"
                + "\n".join(observations)
                + "\n\nUsing only these results plus the user question, give the final answer. "
                "Do not emit more TOOL lines unless absolutely required."
            )
            llm_out = self.llm.chat(follow, system=TOOL_AWARE_SYSTEM)
            if not llm_out.get("ok"):
                break
            reply = llm_out.get("text") or ""

        final_text = strip_tool_lines(reply) or reply
        finished = self._finish_llm(
            {**llm_out, "text": final_text},
            route="llm_tools" if tool_trace else default_route,
            tools=tool_trace,
        )
        return finished

    # --- session / memory -----------------------------------------------------

    def remember(self, key: str, value: str) -> None:
        self.user_memory[str(key)] = str(value)
        self._save()

    def recall(self, key: str) -> Optional[str]:
        return self.user_memory.get(str(key))

    def memory_snapshot(self) -> dict[str, str]:
        return dict(self.user_memory)

    def reset_session(self) -> None:
        self.session.clear()
        if hasattr(self.llm, "history"):
            self.llm.history.clear()
        self._save()

    def session_transcript(self) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self.session]

    def export_session(self, *, fmt: str = "markdown") -> str:
        if fmt == "json":
            return json.dumps(
                {
                    "product": self.PRODUCT_NAME,
                    "version": self.version,
                    "memory": self.user_memory,
                    "session": self.session_transcript(),
                },
                indent=2,
                default=str,
            )
        lines = [f"# {self.PRODUCT_NAME} session (v{self.version})", ""]
        if self.user_memory:
            lines.append("## Memory")
            for k, v in self.user_memory.items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")
        lines.append("## Transcript")
        for t in self.session:
            lines.append(f"**{t.role}** ({t.route}): {t.text}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def completeness(self) -> dict[str, Any]:
        from .runtime.completeness import run_completeness_check

        return run_completeness_check()

    def _restore(self) -> None:
        data = self.store.load()
        self.user_memory = dict(data.get("user_memory") or {})
        # Do not auto-replay full session into LLM history; keep transcript only
        restored: list[SessionTurn] = []
        for row in data.get("session") or []:
            try:
                restored.append(
                    SessionTurn(
                        role=str(row.get("role") or "user"),
                        text=str(row.get("text") or ""),
                        route=str(row.get("route") or ""),
                        meta=dict(row.get("meta") or {}),
                        ts=float(row.get("ts") or time.time()),
                    )
                )
            except Exception:  # noqa: BLE001
                continue
        self.session = restored[-self.session_limit :]

    def _save(self) -> None:
        if not self.autosave:
            return
        self.store.save(
            user_memory=self.user_memory,
            session=self.session_transcript(),
        )

    def _push(self, role: str, text: str, *, route: str = "", meta: Optional[dict] = None) -> None:
        # Avoid persisting huge nested action payloads
        slim_meta = {}
        if meta:
            for k in ("route", "llm_at_runtime", "faithful", "method"):
                if k in meta:
                    slim_meta[k] = meta[k]
            if meta.get("tools"):
                slim_meta["tools_count"] = len(meta["tools"])
        self.session.append(SessionTurn(role=role, text=text, route=route, meta=slim_meta))
        if len(self.session) > self.session_limit:
            self.session = self.session[-self.session_limit :]
        self._save()

    # --- diagnostics ----------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []

        def check(name: str, fn) -> None:
            t0 = time.perf_counter()
            try:
                detail = fn()
                checks.append(
                    {
                        "name": name,
                        "ok": True,
                        "detail": detail,
                        "ms": round((time.perf_counter() - t0) * 1000, 2),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                checks.append(
                    {
                        "name": name,
                        "ok": False,
                        "error": str(exc),
                        "ms": round((time.perf_counter() - t0) * 1000, 2),
                    }
                )

        check("version", lambda: self.version)
        check("finderbay_status", lambda: self.finderbay.handle("status")["frame"]["operation"])
        check("vision_status", lambda: self.vision.status().get("engine"))
        check("coding_status", lambda: self.coding.status().get("name") or "CodingSystem")
        check("llm_bridge", lambda: self.llm.status())
        check(
            "open_domain",
            lambda: self.general.chat.respond("Can penguins fly?").get("supported"),
        )
        check(
            "tool_finderbay",
            lambda: self.tools.run("finderbay", {"text": "status"}).get("ok"),
        )
        check("persist_store", lambda: str(self.store.path))
        check("export_markdown", lambda: "Complete AI" in self.export_session())
        check(
            "intelligence_self_check",
            lambda: __import__(
                "cognitive_ai.intelligence", fromlist=["SelfCheck"]
            ).SelfCheck().run_core_imports().run_invariants().summary()["ok"],
        )
        check(
            "smart_router",
            lambda: __import__(
                "cognitive_ai.intelligence", fromlist=["SmartRouter"]
            ).SmartRouter().decide("shortage report for Site-7").route
            == "jarvis",
        )
        passed = sum(1 for c in checks if c["ok"])
        return {
            "product": self.PRODUCT_NAME,
            "version": self.version,
            "ok": passed == len(checks),
            "passed": passed,
            "total": len(checks),
            "checks": checks,
            "completeness": self.completeness(),
            "llm_at_runtime": bool(self.llm.available),
            "human_level_agi": False,
        }

    def status(self) -> dict[str, Any]:
        return {
            "name": self.PRODUCT_NAME,
            "version": self.version,
            "tagline": "Complete AI",
            "chatgpt_class_general_chat": self.llm.available,
            "llm_at_runtime": bool(self.llm.available),
            "human_level_agi": False,
            "session_turns": len(self.session),
            "user_memory_keys": list(self.user_memory.keys()),
            "tools": list(self.tools.handlers.keys()),
            "modes": {
                "general_chat": "llm+tools" if self.llm.available else "symbolic",
                "finderbay": True,
                "vision": True,
                "coding": True,
                "agent": True,
                "llm": self.llm.status(),
            },
            "setup_general_chat": "export OPENAI_API_KEY=sk-...",
            "limits": [
                "human_level_agi=false",
                "General chat is ChatGPT-class via your API when a key is set",
                "Tools (finderbay/vision/coding) execute locally",
            ],
        }

    def greeting(self) -> str:
        chat = (
            "ChatGPT-class + local tools"
            if self.llm.available
            else "symbolic (set OPENAI_API_KEY for ChatGPT-class chat)"
        )
        return (
            f"{self.PRODUCT_NAME} v{self.version}\n"
            f"General chat: {chat}\n"
            "Tools: finderbay · vision · coding · alerts · memory · status\n"
            "Also: autonomous agent\n\n"
            "Try:\n"
            "  • Write a short poem about rain\n"
            "  • Show unpaid invoices for Project A\n"
            "  • Count people in image ./crowd_people.jpg\n"
            "  • generate function add(a, b)\n"
        )

    # --- helpers --------------------------------------------------------------

    def _tag(self, out: dict[str, Any], *, route: str, llm: bool) -> dict[str, Any]:
        result = dict(out)
        result["product"] = self.PRODUCT_NAME
        result["version"] = self.version
        result["route"] = route
        result["llm_at_runtime"] = bool(llm)
        result["human_level_agi"] = False
        result.setdefault("faithful", True)
        if "text" in result and result["text"] is not None:
            result["text"] = str(result["text"])
        elif result.get("code"):
            result["text"] = str(result.get("code"))
        else:
            result.setdefault("text", result.get("error") or "Done.")
        return result

    def _finish_llm(
        self,
        llm_out: dict[str, Any],
        *,
        route: str,
        prior: Optional[dict] = None,
        tools: Optional[list] = None,
    ) -> dict[str, Any]:
        return {
            "text": llm_out.get("text"),
            "structured": {
                "llm": True,
                "model": llm_out.get("model"),
                "ok": llm_out.get("ok"),
                "error": llm_out.get("error"),
                "prior_route": (prior or {}).get("route"),
                "tools": tools or [],
            },
            "faithful": bool(llm_out.get("ok")),
            "frame": {
                "operation": "llm_chat",
                "supported": bool(llm_out.get("ok")),
                "confidence": 0.85 if llm_out.get("ok") else 0.0,
            },
            "product": self.PRODUCT_NAME,
            "version": self.version,
            "route": route,
            "llm_at_runtime": bool(llm_out.get("ok") and self.llm.available),
            "human_level_agi": False,
            "method": llm_out.get("method"),
            "tools": tools or [],
        }

    @staticmethod
    def _is_unsupported(out: dict[str, Any]) -> bool:
        frame = out.get("frame") or {}
        if frame.get("operation") == "unsupported":
            return True
        if frame.get("supported") is False:
            return True
        text = (out.get("text") or "").lower()
        return "could not map that" in text

    @staticmethod
    def _looks_coding(text: str) -> bool:
        lower = text.lower()
        return any(
            h in lower
            for h in (
                "generate function",
                "generate class",
                "write code",
                "refactor",
                "run tests",
                "find symbol",
                "analyze code",
                "create module",
                "unit test",
                "pytest",
                "implement function",
                "bugfix",
                "write a class",
            )
        )

    @staticmethod
    def _looks_vision(text: str) -> bool:
        lower = text.lower()
        return any(
            h in lower
            for h in (
                "detect objects",
                "detect vehicles",
                "count people",
                "detect fire",
                "vision status",
                "ppe compliance",
                "in image ",
                "computer vision",
                "understand the scene",
                "understand this image",
                "what do you see",
                "thermal",
                "thermography",
                "flir",
                "night vision",
                "infrared",
                ".jpg",
                ".png",
                ".jpeg",
                ".webp",
            )
        )

    @staticmethod
    def _looks_finderbay(text: str) -> bool:
        lower = text.lower()
        hooks = (
            "invoice",
            "delivery",
            "deliveries",
            "follow-up",
            "followup",
            "reserve",
            "project a",
            "project b",
            "purchase order",
            "create po",
            "alert",
            "payment",
            "contract",
            "vendor",
            "warehouse",
            "create task",
            "list tasks",
            "export report",
            "unpaid",
            "cement",
        )
        return any(h in lower for h in hooks)


def _shrink(obj: Any, limit: int = 2500) -> Any:
    try:
        blob = json.dumps(obj, default=str)
    except TypeError:
        return str(obj)[:limit]
    if len(blob) <= limit:
        return obj
    if isinstance(obj, dict):
        keep = {k: obj[k] for k in list(obj)[:12]}
        keep["_truncated"] = True
        return keep
    return blob[:limit]


def create_complete_ai(**kwargs: Any) -> CompleteAI:
    return CompleteAI(**kwargs)
