"""Finderbay Assistant — unified voice, memory, vision, planning, control, learning."""

from __future__ import annotations

from typing import Any, Optional

from ..complete_ai import CompleteAI
from ..version import __version__
from .computer import ComputerControl
from .learning_bridge import LearningBridge
from .memory_hub import MemoryHub
from .voice import AssistantVoice


class FinderbayAssistant:
    """High-level assistant that composes Complete AI capabilities."""

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

        lower = (text or "").lower().strip()

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

        # Plan preview without execute
        if lower.startswith("plan ") or lower.startswith("how would you "):
            return self.plan(text)

        # Default → Complete AI (vision/coding/finderbay/llm/tools)
        out = self.complete.handle(text, mode=mode)
        out = dict(out)
        out["product"] = self.PRODUCT_NAME
        out["assistant"] = True
        # Attach plan snapshot when Finderbay produced one
        goal = out.get("goal")
        plan = out.get("plan")
        if plan:
            self.last_plan = plan
        out["capabilities_used"] = self._infer_caps(out)

        do_speak = self.speak_replies if speak is None else speak
        if do_speak and out.get("text"):
            out["voice"] = self.voice.speak(str(out["text"])[:500])

        self.memory.note_episode(
            summary=f"{out.get('route')}: {text[:120]}",
            tags=[str(out.get("route") or "ask")],
            data={"reply_preview": str(out.get("text") or "")[:200]},
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
        return self._tag(
            {
                "text": f"Vision {task}: count={result.get('count')} engine={result.get('engine')}",
                "structured": result,
            },
            route="vision",
        )

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
        }

    def greeting(self) -> str:
        cc = "ON" if self.computer.allow else "OFF"
        tts = self.voice.tts_engine
        return (
            f"{self.PRODUCT_NAME} v{self.version}\n"
            f"Voice TTS={tts} · Computer control={cc}\n"
            "I can: talk (ASR/TTS), remember, see (YOLO), plan, control this computer (gated), learn from corrections.\n"
            "Try: 'Show unpaid invoices for Project A' · 'Count people in image ./crowd_people.jpg'\n"
            "     'remember that my warehouse is Central Store' · 'plan reserve 10 cement'\n"
            "     'computer open_url https://example.com' (requires control enabled)\n"
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
