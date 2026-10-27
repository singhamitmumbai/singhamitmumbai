"""Finderbay Assistant — unified voice, memory, vision, planning, control, learning + JARVIS."""

from __future__ import annotations

from typing import Any, Optional

from ..complete_ai import CompleteAI
from ..intelligence import SmartRouter, harden_text, score_response
from ..jarvis import JarvisPipeline, JarvisSystem, run_jarvis_eval
from ..jarvis.voice import VoicePipeline
from ..version import __version__
from .computer import ComputerControl
from .learning_bridge import LearningBridge
from .memory_hub import MemoryHub
from .voice import AssistantVoice


class FinderbayAssistant:
    """High-level assistant that composes Complete AI capabilities.

    Includes a non-LLM JARVIS controller (tool contracts, jobs, provenance memory,
    skills, voice pipeline) with optional LLM via Complete AI.
    """

    PRODUCT_NAME = "Finderbay Assistant"

    def __init__(
        self,
        *,
        seed_demo: bool = True,
        enable_llm: Optional[bool] = None,
        allow_computer_control: Optional[bool] = None,
        allow_shell: bool = False,
        speak_replies: bool = False,
        complete: Optional[CompleteAI] = None,
    ) -> None:
        self.version = __version__
        self.complete = complete or CompleteAI(seed_demo=seed_demo, enable_llm=enable_llm)
        self.finderbay = self.complete.finderbay
        self.vision = self.complete.vision
        self.voice = AssistantVoice()
        self.memory = MemoryHub(self.finderbay, user_memory=self.complete.user_memory)
        self.computer = ComputerControl(allow=allow_computer_control, allow_shell=allow_shell)
        self.learning = LearningBridge(self.finderbay)
        self.speak_replies = speak_replies
        self.last_plan: Optional[dict[str, Any]] = None
        # JARVIS blueprint stack (non-LLM controller) — full system + pipeline
        self.jarvis_system = JarvisSystem(finderbay=self.finderbay)
        self.jarvis = self.jarvis_system.pipeline  # backward-compatible pipeline handle
        self.jarvis_voice = self.jarvis_system.voice
        self.router = SmartRouter()
        # Expose assistant tools on Complete AI tool router
        self.complete.attach_assistant(self)

    # --- primary --------------------------------------------------------------

    def ask(self, text: str, *, mode: str = "auto", speak: Optional[bool] = None) -> str:
        return self.handle(text, mode=mode, speak=speak)["text"]

    def handle(
        self,
        text: str,
        *,
        mode: str = "auto",
        speak: Optional[bool] = None,
        audio: Any = None,
    ) -> dict[str, Any]:
        # Voice in
        if audio is not None:
            heard = self.voice.listen(audio)
            text = heard.get("transcript") or text
            if not text:
                return {
                    "text": "I could not hear a transcript. Load a Vosk model or pass text.",
                    "route": "voice_listen_failed",
                    "speech": heard,
                    "product": self.PRODUCT_NAME,
                    "human_level_agi": False,
                }

        text = harden_text(text or "")
        lower = text.lower().strip()

        # Assistant meta commands
        if lower.startswith("remember that ") or lower.startswith("remember "):
            return self._remember_cmd(text)
        if lower.startswith("recall ") or lower.startswith("what do you remember"):
            return self._recall_cmd(text)
        if lower.startswith("learn alias ") or " is also called " in lower:
            return self._learn_alias_cmd(text)
        if lower.startswith("correct:") or lower.startswith("correction:"):
            return self._correction_cmd(text)
        if lower.startswith("computer ") or lower.startswith("control "):
            return self._computer_cmd(text)
        if lower in ("assistant status", "finder assistant status", "capabilities"):
            return self._tag({"text": self.greeting(), "structured": self.status()}, route="assistant_status")
        if lower in ("jarvis status", "jarvis eval", "jarvis tools"):
            return self._jarvis_meta(lower)
        if lower.startswith("jarvis ") or lower.startswith("jarvis,"):
            return self.jarvis_handle(text, speak=speak)
        if lower in ("self check", "self-check", "assistant selfcheck"):
            return self._self_check()

        # Plan preview without execute
        if lower.startswith("plan ") or lower.startswith("how would you "):
            return self.plan(text)

        # Smart multi-signal router
        decision = self.router.decide(text)
        route = decision.route
        image_path = decision.hints.get("image_path")

        if route == "jarvis" or self._looks_like_jarvis(lower):
            out = self.jarvis_handle(text, speak=speak)
            out["router"] = decision.to_dict()
            return out
        if route == "vision_thermal":
            path = image_path or (text.split(None, 2)[-1].strip().strip("\"'") if " " in text else "")
            if path:
                out = self.analyze_thermal(path)
                out["router"] = decision.to_dict()
                return out
        if route == "vision_understand" or (
            route == "vision" and any(k in lower for k in ("understand", "what do you see", "describe"))
        ):
            path = image_path or (text.split(None, 2)[-1].strip().strip("\"'") if " " in text else "")
            if path and ("/" in path or path.endswith((".jpg", ".png", ".jpeg", ".webp", ".bmp", ".tif", ".dcm"))):
                out = self.understand_image(path)
                out["router"] = decision.to_dict()
                return out
        if route == "memory":
            if lower.startswith("remember"):
                return self._remember_cmd(text)
            return self._recall_cmd(text)
        if route == "learning":
            if "correction" in lower:
                return self._correction_cmd(text)
            return self._learn_alias_cmd(text)
        if route == "computer":
            return self._computer_cmd(text)
        if route == "planning":
            return self.plan(text)
        if route == "web_search" or lower.startswith("search the web") or lower.startswith("look up "):
            return self.web_search(text, speak=speak)
        if route == "proto_agi" or any(
            k in lower
            for k in (
                "multi-step",
                "then verify",
                "proto-agi",
                "agi-gap",
                "compose transfer",
                "long-horizon",
                "long horizon",
            )
        ):
            raw = self.complete.proto_stack.handle(text)
            out = dict(raw)
            out["product"] = self.PRODUCT_NAME
            out["assistant"] = True
            out["route"] = raw.get("route") or "proto_agi"
            out["router"] = decision.to_dict()
            out["human_level_agi"] = False
            do_speak = self.speak_replies if speak is None else speak
            if do_speak and out.get("text"):
                out["voice"] = self.voice.speak(str(out["text"])[:500])
            return out

        # Vision shorthand prefixes
        if lower.startswith("understand image ") or lower.startswith("understand scene "):
            path = text.split(None, 2)[-1].strip().strip("\"'")
            return self.understand_image(path)
        if lower.startswith("see image ") or lower.startswith("look at image "):
            path = text.split(None, 2)[-1].strip().strip("\"'")
            return self.understand_image(path)
        if lower.startswith("thermal ") or lower.startswith("analyze thermal "):
            path = text.split(None, 2)[-1].strip().strip("\"'") if " " in text.strip() else ""
            if path:
                return self.analyze_thermal(path)

        # Default → Complete AI (vision/coding/finderbay/llm/tools)
        # Prefer vision mode when router says so
        use_mode = mode
        if mode == "auto" and route in ("vision", "vision_understand", "vision_thermal"):
            use_mode = "vision"
        elif mode == "auto" and route == "coding":
            use_mode = "coding"
        elif mode == "auto" and route == "finderbay":
            use_mode = "finderbay"

        out = self.complete.handle(text, mode=use_mode)
        out = dict(out)
        out["product"] = self.PRODUCT_NAME
        out["assistant"] = True
        out["router"] = decision.to_dict()
        plan = out.get("plan")
        if plan:
            self.last_plan = plan
        out["capabilities_used"] = self._infer_caps(out)
        out["confidence"] = score_response(
            verified=out.get("faithful"),
            supported=(out.get("frame") or {}).get("supported", True) if out.get("frame") else True,
            intent_confidence=decision.confidence,
            engine_stub=("stub" in str((out.get("structured") or {}).get("engine") or "")),
        ).to_dict()

        do_speak = self.speak_replies if speak is None else speak
        if do_speak and out.get("text"):
            out["voice"] = self.voice.speak(str(out["text"])[:500])

        self.memory.note_episode(
            summary=f"{out.get('route')}: {text[:120]}",
            tags=[str(out.get("route") or "ask"), decision.route],
            data={"reply_preview": str(out.get("text") or "")[:200], "router": decision.route},
        )
        return out

    def plan(self, text: str) -> dict[str, Any]:
        """Run NLU + planner without committing side effects when possible."""
        # Use Finderbay understand→plan path via handle but mark as plan-only when unsupported tools
        frame = self.finderbay.nlu.parse(text)
        from ..core.records import GoalRecord
        from ..core.working_state import WorkingState

        state = WorkingState()
        state.semantic_frame = frame.to_dict() if hasattr(frame, "to_dict") else dict(frame)
        goal = GoalRecord(
            description=text,
            desired_conditions=[{"operation": frame.operation}],
            permitted_operations=[frame.operation],
            completion_criteria=[frame.operation],
        )
        plan = self.finderbay.planner.plan(goal, state)
        self.last_plan = plan.to_dict()
        steps = [
            {"operator": s.operator, "args": s.args, "verification": s.verification}
            for s in plan.steps
        ]
        text_out = (
            f"Plan ({plan.outcome}) for '{frame.operation}': "
            + ", ".join(s["operator"] for s in steps)
            + "."
        )
        return self._tag(
            {
                "text": text_out,
                "structured": {"operation": frame.operation, "steps": steps, "plan": self.last_plan},
                "plan": self.last_plan,
            },
            route="planning",
        )

    def see(self, image: str, *, task: str = "detect_objects") -> dict[str, Any]:
        result = self.vision.analyze(image, task=task)
        text = result.get("text") or (
            f"Vision {task}: count={result.get('count')} engine={result.get('engine')}"
        )
        return self._tag(
            {
                "text": text,
                "structured": result,
            },
            route="vision",
        )

    def understand_image(
        self,
        image: str,
        *,
        search_unknown: bool = True,
        modality: str | None = None,
    ) -> dict[str, Any]:
        """Smart see→analyze→understand; auto thermal/IR/night/other modalities."""
        result = self.vision.understand(image, search_unknown=search_unknown, modality=modality)
        return self._tag(
            {
                "text": result.get("text") or "Vision understanding complete.",
                "structured": result,
                "verified": bool((result.get("understand") or {}).get("all_understood")),
                "modality": (result.get("modality") or {}).get("modality")
                if isinstance(result.get("modality"), dict)
                else result.get("modality"),
                "alert": result.get("alert"),
            },
            route="vision_understand",
        )

    def analyze_thermal(self, image: str) -> dict[str, Any]:
        result = self.vision.analyze_thermal(image)
        return self._tag(
            {
                "text": result.get("text") or result.get("summary") or "Thermal analysis done.",
                "structured": result,
                "alert": result.get("alert"),
                "modality": "thermal",
            },
            route="vision_thermal",
        )

    def web_search(self, text: str, *, speak: Optional[bool] = None) -> dict[str, Any]:
        """Intelligent internet/local research for open questions."""
        from ..nlu.smart_questions import SmartQuestionUnderstanding
        from ..search import IntelligentWebSearch

        frame = SmartQuestionUnderstanding().understand(text)
        query = text
        m = __import__("re").search(
            r"(?:search(?:\s+the\s+web)?(?:\s+for)?|look\s+up|google|find\s+online)\s+(.+)$",
            text,
            __import__("re").I,
        )
        if m:
            query = m.group(1).strip(" ?.")
        elif frame.entities:
            query = " ".join(frame.entities[:3])
        result = IntelligentWebSearch(allow_live=True).search(
            query, question=frame.analysis
        )
        out = self._tag(
            {
                "text": result.answer,
                "structured": result.to_dict(),
                "smart": frame.to_dict(),
                "verified": result.ok,
                "confidence": score_response(
                    verified=result.ok,
                    supported=result.ok,
                    intent_confidence=result.confidence,
                ).to_dict(),
            },
            route="web_search",
        )
        do_speak = self.speak_replies if speak is None else speak
        if do_speak and out.get("text"):
            out["voice"] = self.voice.speak(str(out["text"])[:500])
        return out

    def speak(self, text: str) -> dict[str, Any]:
        return self.voice.speak(text)

    def listen(self, audio: Any) -> dict[str, Any]:
        return self.voice.listen(audio)

    def control(self, action: str, **kwargs: Any) -> dict[str, Any]:
        return self.computer.run(action, **kwargs)

    # --- status ---------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "name": self.PRODUCT_NAME,
            "version": self.version,
            "human_level_agi": False,
            "voice": self.voice.status(),
            "memory": self.memory.snapshot(),
            "vision": self.vision.status(),
            "planning": {"last_plan": self.last_plan is not None, "engine": "OperatorPlanner"},
            "computer_control": self.computer.status(),
            "learning": self.learning.status(),
            "complete_ai": self.complete.status(),
            "jarvis": self.jarvis_system.status(),
            "router": {"engine": "SmartRouter", "routes": list(self.router.ROUTES)},
            "intelligence": {"hardening": True, "confidence": True, "self_check": True},
        }

    def greeting(self) -> str:
        cc = "ON" if self.computer.allow else "OFF"
        tts = self.voice.tts_engine
        n_tools = len(self.jarvis.tools.contracts)
        return (
            f"{self.PRODUCT_NAME} v{self.version}\n"
            f"Voice TTS={tts} · Computer control={cc} · JARVIS tools={n_tools} (non-LLM controller)\n"
            "I can: talk (ASR/TTS), remember, see (YOLO), plan, control this computer (gated), learn from corrections.\n"
            "JARVIS workflows: material shortage report, schedule, file search, verified jobs.\n"
            "Try: 'Prepare tomorrow's material shortage report for the selected site'\n"
            "     'Show unpaid invoices for Project A' · 'Count people in image ./crowd_people.jpg'\n"
            "     'remember that my warehouse is Central Store' · 'plan reserve 10 cement'\n"
            "     'computer open_url https://example.com' (requires control enabled)\n"
            "human_level_agi=false — optional LLM via API key only.\n"
        )

    def jarvis_handle(self, text: str, *, speak: Optional[bool] = None) -> dict[str, Any]:
        """Run the full JARVIS system (blueprint control loop)."""
        raw = text.strip()
        if raw.lower().startswith("jarvis"):
            raw = raw.split(None, 1)[1] if " " in raw else raw
        do_speak = self.speak_replies if speak is None else speak
        out = self.jarvis_system.handle(raw, speak=bool(do_speak))
        # Normalize assistant envelope
        out = self._tag(
            {
                "text": out.get("text") or "",
                "structured": out.get("structured"),
                "nlu": out.get("nlu"),
                "job_id": out.get("job_id"),
                "verified": out.get("verified"),
                "intent": out.get("intent"),
                "dialogue": out.get("dialogue"),
                "voice": out.get("voice"),
            },
            route="jarvis",
        )
        out["capabilities_used"] = ["jarvis", "planning", "tools", "verification", "memory"]
        self.memory.note_episode(
            summary=f"jarvis:{out.get('intent')}:{raw[:100]}",
            tags=["jarvis", str(out.get("intent") or "")],
            data={"verified": out.get("verified"), "job_id": out.get("job_id")},
        )
        return out

    def jarvis_eval(self) -> dict[str, Any]:
        return self.jarvis_system.eval()

    def _looks_like_jarvis(self, lower: str) -> bool:
        keys = (
            "shortage report",
            "material shortage",
            "material balance",
            "yaad dilana",
            "remind me",
            "stop this task",
            "cancel this job",
            "jarvis eval",
        )
        return any(k in lower for k in keys)

    def _jarvis_meta(self, lower: str) -> dict[str, Any]:
        if "eval" in lower:
            report = self.jarvis_eval()
            return self._tag(
                {
                    "text": f"JARVIS eval: {report['passed']}/{report['total']} passed "
                    f"(rate={report['pass_rate']:.0%}). human_level_agi=false.",
                    "structured": report,
                },
                route="jarvis_eval",
            )
        if "tools" in lower:
            tools = self.jarvis.tools.list_tools()
            names = ", ".join(t["name"] for t in tools[:20])
            return self._tag(
                {"text": f"JARVIS tools ({len(tools)}): {names}", "structured": {"tools": tools}},
                route="jarvis_tools",
            )
        return self._tag(
            {"text": self.greeting(), "structured": self.status().get("jarvis")},
            route="jarvis_status",
        )

    # --- command helpers ------------------------------------------------------

    def _remember_cmd(self, text: str) -> dict[str, Any]:
        # remember that X is Y  / remember key=value / remember foo bar
        import re

        m = re.search(r"remember(?:\s+that)?\s+(.+?)\s+is\s+(.+)$", text, re.I)
        if m:
            key, val = m.group(1).strip(), m.group(2).strip()
            self.memory.remember_preference(key, val)
            self.complete.remember(key, val)
            return self._tag({"text": f"Remembered: {key} → {val}."}, route="memory")
        m = re.search(r"remember\s+([^=\s]+)\s*=\s*(.+)$", text, re.I)
        if m:
            key, val = m.group(1).strip(), m.group(2).strip()
            self.memory.remember_preference(key, val)
            self.complete.remember(key, val)
            return self._tag({"text": f"Remembered: {key} → {val}."}, route="memory")
        blob = text.split(None, 1)[1] if " " in text else text
        self.memory.note_episode(blob, tags=["note"])
        return self._tag({"text": f"Noted: {blob}"}, route="memory")

    def _recall_cmd(self, text: str) -> dict[str, Any]:
        q = text.split(None, 1)[1] if " " in text.strip() else ""
        if not q or "remember" in q:
            snap = self.memory.snapshot()
            prefs = snap.get("preferences") or {}
            if not prefs:
                return self._tag({"text": "I have no stored preferences yet."}, route="memory")
            listing = ", ".join(f"{k}={v}" for k, v in list(prefs.items())[:20])
            return self._tag({"text": f"I remember: {listing}"}, route="memory")
        hits = self.memory.search(q)
        if not hits:
            val = self.memory.recall(q)
            if val:
                return self._tag({"text": f"{q} → {val}"}, route="memory")
            return self._tag({"text": f"Nothing recalled for '{q}'."}, route="memory")
        lines = [f"- {h.get('kind')}: {h}" for h in hits[:8]]
        return self._tag({"text": "Recall hits:\n" + "\n".join(lines)}, route="memory")

    def _learn_alias_cmd(self, text: str) -> dict[str, Any]:
        import re

        m = re.search(r"(.+?)\s+is also called\s+(.+)$", text, re.I)
        if not m:
            m = re.search(r"learn alias\s+(\S+)\s+(\S+)", text, re.I)
        if not m:
            return self._tag({"text": "Usage: learn alias <old> <new>  OR  X is also called Y"}, route="learning")
        former, alias = m.group(1).strip(), m.group(2).strip()
        out = self.learning.learn_alias(former, alias)
        return self._tag({"text": f"Learned alias: {former} → {alias}", "structured": out}, route="learning")

    def _correction_cmd(self, text: str) -> dict[str, Any]:
        # correction: input => wrong => right
        body = text.split(":", 1)[1].strip()
        parts = [p.strip() for p in body.split("=>")]
        if len(parts) < 3:
            return self._tag(
                {"text": "Usage: correction: <input> => <wrong> => <right>"},
                route="learning",
            )
        out = self.learning.record_correction(input_text=parts[0], wrong=parts[1], right=parts[2])
        return self._tag({"text": f"Recorded correction for review: {parts[1]} → {parts[2]}", "structured": out}, route="learning")

    def _computer_cmd(self, text: str) -> dict[str, Any]:
        import shlex

        raw = text.split(None, 1)[1] if " " in text.strip() else "status"
        try:
            parts = shlex.split(raw)
        except ValueError:
            parts = raw.split()
        action = parts[0] if parts else "status"
        args: dict[str, Any] = {}
        if action == "open_url" and len(parts) > 1:
            args["url"] = parts[1]
        elif action == "open_path" and len(parts) > 1:
            args["path"] = parts[1]
        elif action == "notify":
            args["title"] = "Finderbay Assistant"
            args["message"] = " ".join(parts[1:]) or "Hello"
        elif action == "screenshot":
            args["path"] = parts[1] if len(parts) > 1 else "data/complete_ai/screenshot.png"
        elif action == "run_command":
            args["command"] = " ".join(parts[1:])
            args["confirm"] = True
        elif action == "type_text":
            args["text"] = " ".join(parts[1:])
            args["confirm"] = True
        result = self.computer.run(action, **args)
        ok = result.get("ok")
        return self._tag(
            {
                "text": f"Computer {action}: {'ok' if ok else result.get('error')}",
                "structured": result,
            },
            route="computer_control",
        )

    def _infer_caps(self, out: dict[str, Any]) -> list[str]:
        caps = ["planning"]
        route = str(out.get("route") or "")
        if "vision" in route or (out.get("frame") or {}).get("operation", "").startswith("detect"):
            caps.append("vision")
        if route.startswith("llm"):
            caps.append("voice_optional")
            caps.append("general_chat")
        if route in ("finderbay", "coding", "agent"):
            caps.append(route)
        if out.get("tools"):
            caps.append("tools")
        return caps

    def _self_check(self) -> dict[str, Any]:
        from ..intelligence import SelfCheck

        report = SelfCheck().run_core_imports().run_invariants().summary()
        return self._tag(
            {
                "text": f"Self-check {report['passed']}/{report['total']} passed. human_level_agi=false.",
                "structured": report,
            },
            route="self_check",
        )

    def _tag(self, out: dict[str, Any], *, route: str) -> dict[str, Any]:
        result = dict(out)
        result.setdefault("text", "")
        result["route"] = route
        result["product"] = self.PRODUCT_NAME
        result["version"] = self.version
        result["human_level_agi"] = False
        result["llm_at_runtime"] = False
        result["assistant"] = True
        return result
