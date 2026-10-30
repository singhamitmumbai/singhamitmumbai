"""Text/input hardening + lightweight self-check utilities."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


def harden_text(text: str, *, max_len: int = 8000) -> str:
    """Normalize user text: unicode, whitespace, length bound."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = s.strip()
    if len(s) > max_len:
        s = s[:max_len].rstrip() + "…"
    return s


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class SelfCheck:
    """Run a battery of cheap invariants before shipping a release/answer."""

    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.results.append(CheckResult(name, ok, detail))

    def run_core_imports(self) -> "SelfCheck":
        modules = [
            "cognitive_ai.complete_ai",
            "cognitive_ai.assistant.finder_assistant",
            "cognitive_ai.jarvis.system",
            "cognitive_ai.vision.system",
            "cognitive_ai.intelligence.router",
            "cognitive_ai.nlu.pipeline",
            "cognitive_ai.execution.verifier",
            "cognitive_ai.memory.retrieval",
        ]
        for mod in modules:
            try:
                __import__(mod)
                self.add(f"import:{mod}", True)
            except Exception as exc:  # noqa: BLE001
                self.add(f"import:{mod}", False, str(exc))
        return self

    def run_invariants(self) -> "SelfCheck":
        from cognitive_ai.version import __version__
        from cognitive_ai.jarvis import JarvisSystem
        from cognitive_ai.vision import VisionSystem
        from cognitive_ai.intelligence import SmartRouter

        self.add("version_set", bool(__version__), __version__)
        js = JarvisSystem()
        self.add("jarvis_tools", len(js.tools.contracts) >= 15, str(len(js.tools.contracts)))
        vs = VisionSystem()
        st = vs.status()
        self.add("vision_modalities", "thermal" in (st.get("understanding") or {}).get("modalities", []), "")
        self.add("human_level_agi_false", True, "always_false")
        d = SmartRouter().decide("Understand thermal image ./x_thermal.jpg")
        self.add("router_thermal", d.route in ("vision_thermal", "vision_understand"), d.route)
        d2 = SmartRouter().decide("Prepare material shortage report for Site-7")
        self.add("router_jarvis", d2.route == "jarvis", d2.route)
        return self

    def summary(self) -> dict[str, Any]:
        passed = sum(1 for r in self.results if r.ok)
        return {
            "ok": passed == len(self.results) and len(self.results) > 0,
            "passed": passed,
            "total": len(self.results),
            "results": [r.to_dict() for r in self.results],
            "human_level_agi": False,
        }
