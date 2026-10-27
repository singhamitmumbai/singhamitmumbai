"""Complete AI — unified product.

Product name: **Complete AI** (no subtitle required).

Capabilities:
  • ChatGPT-class general chat when OPENAI_API_KEY is set (primary open chat)
  • Finderbay business tasks, YOLO vision, coding, SelfBrain agent (always)
  • Symbolic open-domain fallback when no LLM key

Modes:
  auto      — tools for tasks/vision/coding; else LLM chat if available; else symbolic
  llm       — force ChatGPT-class chat
  symbolic  — never call LLM
  finderbay / vision / coding / agent — force subsystem

human_level_agi is always false (Complete AI ≠ claimed AGI).
"""

from __future__ import annotations

import os
from typing import Any, Optional

from .agent import FinderbayAgentSystem
from .chat.llm_bridge import LLMChatBridge, llm_bridge_from_env
from .config import SystemConfig
from .general.agent import GeneralNonLLMAI
from .version import __version__
from .vision import VisionSystem


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
    ) -> None:
        self.version = __version__
        self.general = GeneralNonLLMAI(config=config, seed_demo=seed_demo)
        self.finderbay = self.general.task_ai
        self.vision = self.finderbay.vision if hasattr(self.finderbay, "vision") else VisionSystem()
        self.agent = FinderbayAgentSystem(seed_demo=seed_demo)
        self.coding = self.agent.coding
        if llm is not None:
            self.llm = llm
            if enable_llm is not None:
                self.llm.enabled = enable_llm
        else:
            self.llm = llm_bridge_from_env()
            if enable_llm is not None:
                self.llm.enabled = enable_llm

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

        if mode == "llm":
            return self._finish_llm(self.llm.chat(text), route="llm")

        if mode in ("auto", "coding") and (mode == "coding" or self._looks_coding(text)):
            out = self.coding.handle(text)
            return self._tag(out, route="coding", llm=False)

        if mode in ("auto", "vision") and (mode == "vision" or self._looks_vision(text)):
            out = self.finderbay.handle(text, modality=modality, context=context)
            return self._tag(out, route="vision", llm=False)

        if mode == "agent":
            report = self.agent.auto([text])
            return self._tag(
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

        if mode == "finderbay" or (mode == "auto" and self._looks_finderbay(text)):
            out = self.finderbay.handle(payload, modality=modality, context=context)
            # If Finderbay unsupported and LLM available, fall through to general chat
            if mode == "finderbay" or (out.get("frame") or {}).get("operation") != "unsupported":
                if (out.get("frame") or {}).get("operation") != "unsupported":
                    return self._tag(out, route="finderbay", llm=False)

        if mode == "symbolic":
            out = self.general.handle(payload, modality=modality, context=context, force="auto")
            return self._tag(out, route=out.get("route") or "symbolic", llm=False)

        # --- General chat: ChatGPT-class first when available -----------------
        if mode == "auto" and self.llm.available and not self._looks_finderbay(text):
            llm_out = self.llm.chat(text)
            if llm_out.get("ok"):
                return self._finish_llm(llm_out, route="llm_chat")

        # Symbolic / Finderbay general path
        out = self.general.handle(payload, modality=modality, context=context, force="auto")
        route = out.get("route") or "general"
        unsupported = self._is_unsupported(out)

        if unsupported and self.llm.available:
            llm_out = self.llm.chat(text)
            if llm_out.get("ok"):
                return self._finish_llm(llm_out, route="llm_fallback", prior=out)

        return self._tag(out, route=route, llm=False)

    def status(self) -> dict[str, Any]:
        return {
            "name": self.PRODUCT_NAME,
            "version": self.version,
            "tagline": "Complete AI",
            "chatgpt_class_general_chat": self.llm.available,
            "llm_at_runtime": bool(self.llm.available),
            "human_level_agi": False,
            "modes": {
                "general_chat": "llm" if self.llm.available else "symbolic",
                "finderbay": True,
                "vision": True,
                "coding": True,
                "agent": True,
                "llm": self.llm.status(),
            },
            "setup_general_chat": (
                "export OPENAI_API_KEY=sk-...  # enables ChatGPT-class chat automatically"
            ),
            "limits": [
                "human_level_agi=false",
                "General chat is ChatGPT-class via your API when a key is set",
                "Specialist engines (Finderbay/vision/coding) stay local/symbolic",
            ],
        }

    def greeting(self) -> str:
        chat = "ChatGPT-class (API key detected)" if self.llm.available else "symbolic (set OPENAI_API_KEY for ChatGPT-class chat)"
        return (
            f"{self.PRODUCT_NAME} v{self.version}\n"
            f"General chat: {chat}\n"
            "Also: Finderbay tasks · computer vision · coding · autonomous agent\n\n"
            "Try anything — casual chat, writing, explanations — or:\n"
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
    ) -> dict[str, Any]:
        return {
            "text": llm_out.get("text"),
            "structured": {
                "llm": True,
                "model": llm_out.get("model"),
                "ok": llm_out.get("ok"),
                "error": llm_out.get("error"),
                "prior_route": (prior or {}).get("route"),
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
        }

    @staticmethod
    def _is_unsupported(out: dict[str, Any]) -> bool:
        frame = out.get("frame") or {}
        if frame.get("operation") == "unsupported":
            return True
        if frame.get("supported") is False:
            return True
        text = (out.get("text") or "").lower()
        if "could not map that" in text:
            return True
        return False

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


def create_complete_ai(**kwargs: Any) -> CompleteAI:
    return CompleteAI(**kwargs)
