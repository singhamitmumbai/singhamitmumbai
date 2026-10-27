"""Complete AI — unified product facade.

Branding: **Complete AI** (full stack). Honest limits remain:
- human_level_agi = false (never claimed)
- Core engines are non-LLM (Finderbay, vision, coding, common-sense)
- Optional ChatGPT-class general chat via LLM bridge when enabled

Modes for a message:
  auto     — Finderbay/vision/coding if it looks like a task, else open-domain;
             if open-domain fails and LLM bridge is available, use LLM
  symbolic — never call LLM
  llm      — force optional LLM bridge (ChatGPT-class)
  finderbay / vision / coding / agent — force subsystem
"""

from __future__ import annotations

import os
from typing import Any, Optional

from .agent import FinderbayAgentSystem
from .chat.llm_bridge import LLMChatBridge, llm_bridge_from_env
from .coding import CodingSystem
from .config import SystemConfig
from .general.agent import GeneralNonLLMAI
from .version import __version__
from .vision import VisionSystem


class CompleteAI:
    """One entrypoint for the full Complete AI product."""

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
        else:
            self.llm = llm_bridge_from_env()
            if enable_llm is not None:
                self.llm.enabled = enable_llm
            elif os.environ.get("COMPLETE_AI_ENABLE_LLM", "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            ):
                self.llm.enabled = True

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

        if mode == "llm":
            llm_out = self.llm.chat(text)
            wrapped = self._wrap_llm(llm_out, route="llm_forced")
            # Only mark llm_at_runtime when a model was actually reachable/configured
            wrapped["llm_at_runtime"] = bool(llm_out.get("ok") and self.llm.available)
            return wrapped

        if mode == "coding" or self._looks_coding(text):
            if mode in ("auto", "coding"):
                out = self.coding.handle(text)
                return self._tag(out, route="coding", llm=False)

        if mode == "vision" or self._looks_vision(text):
            if mode in ("auto", "vision"):
                out = self.finderbay.handle(text, modality=modality, context=context)
                return self._tag(out, route="vision", llm=False)

        if mode == "agent":
            report = self.agent.auto([text])
            return self._tag(
                {
                    "text": f"Agent run finished: succeeded={report.get('succeeded')} failed={report.get('failed')}",
                    "structured": report,
                    "faithful": True,
                },
                route="agent",
                llm=False,
            )

        if mode == "finderbay":
            out = self.finderbay.handle(payload, modality=modality, context=context)
            return self._tag(out, route="finderbay", llm=False)

        if mode == "symbolic":
            out = self.general.handle(payload, modality=modality, context=context, force="auto")
            return self._tag(out, route=out.get("route") or "symbolic", llm=False)

        # auto: prefer symbolic/general; escalate to LLM only if unsupported + bridge ready
        out = self.general.handle(payload, modality=modality, context=context, force="auto")
        route = out.get("route") or "general"
        supported = True
        if route == "open_domain":
            supported = bool((out.get("structured") or {}).get("method") not in ("fallback",))
            # OpenDomainChat puts supported on frame / structured
            frame = out.get("frame") or {}
            if frame.get("supported") is False:
                supported = False
            if out.get("structured", {}).get("open_domain") and (
                (out.get("frame") or {}).get("supported") is False
                or (out.get("method") == "fallback")
            ):
                supported = False
            # General wrap uses frame.supported
            if (out.get("frame") or {}).get("operation") == "open_domain_chat":
                # check method via asking chat history — use confidence/text heuristic
                if (out.get("frame") or {}).get("supported") is False:
                    supported = False
        if (out.get("frame") or {}).get("operation") == "unsupported":
            supported = False
        # Detect open-domain fallback text
        if route == "open_domain" and "could not map that" in (out.get("text") or "").lower():
            supported = False

        if not supported and self.llm.available and mode == "auto":
            llm_out = self.llm.chat(text)
            if llm_out.get("ok"):
                return self._wrap_llm(llm_out, route="llm_fallback", prior=out)

        return self._tag(out, route=route, llm=False)

    # --- status / identity ----------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "name": self.PRODUCT_NAME,
            "version": self.version,
            "tagline": "Complete AI — full stack: tasks, vision, coding, agent, general chat",
            "llm_at_runtime": bool(self.llm.available),
            "human_level_agi": False,
            "modes": {
                "symbolic_core": True,
                "finderbay": True,
                "vision": True,
                "coding": True,
                "agent": True,
                "open_domain_chat": True,
                "llm_general_chat": self.llm.status(),
            },
            "general": self.general.status(),
            "how_to_enable_chatgpt_class_chat": (
                "export COMPLETE_AI_ENABLE_LLM=1 OPENAI_API_KEY=sk-... "
                "then ask general questions; or force mode=llm"
            ),
            "limits": [
                "Not human-level AGI",
                "Without LLM bridge: curated open-domain + Finderbay (not unrestricted ChatGPT)",
                "With LLM bridge: ChatGPT-class general chat via your API key (llm_at_runtime=true)",
            ],
        }

    def greeting(self) -> str:
        llm = "ON" if self.llm.available else "OFF"
        return (
            f"{self.PRODUCT_NAME} v{self.version} ready.\n"
            f"LLM general chat: {llm} (ChatGPT-class when ON).\n"
            "I handle Finderbay business tasks, computer vision, coding, autonomous agent jobs,\n"
            "and open-domain chat. Not human-level AGI.\n\n"
            "Try:\n"
            "  • Show unpaid invoices for Project A\n"
            "  • Count people in image ./crowd_people.jpg\n"
            "  • generate function add(a, b)\n"
            "  • Can penguins fly?\n"
            "  • Explain gravity simply\n"
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
        return result

    def _wrap_llm(
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
                "confidence": 0.7 if llm_out.get("ok") else 0.0,
            },
            "product": self.PRODUCT_NAME,
            "version": self.version,
            "route": route,
            "llm_at_runtime": True,
            "human_level_agi": False,
            "method": llm_out.get("method"),
        }

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


def create_complete_ai(**kwargs: Any) -> CompleteAI:
    return CompleteAI(**kwargs)
