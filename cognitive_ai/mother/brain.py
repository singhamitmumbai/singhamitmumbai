"""Finderbay Mother Brain — coordinator over existing cognitive subsystems.

Implements the unified cognitive loop using real existing modules:
CompleteAI, symbolic Brain/dialogue, thinking, lifelong memory, skills, coding.
Does NOT claim AGI. human_level_agi=false always.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..brain import Brain
from ..core.ids import new_id
from ..core.records import utc_now
from ..version import __version__
from .beliefs import BeliefStore
from .events import EventBus
from .gateway import InputGateway, InputInterpretation
from .goals import GoalManager
from .workspace import GlobalWorkspace


FAILURE_TYPES = (
    "LANGUAGE_UNDERSTANDING",
    "REFERENCE_RESOLUTION",
    "KNOWLEDGE_MISSING",
    "MEMORY_RETRIEVAL",
    "PLANNING",
    "TOOL_SELECTION",
    "TOOL_EXECUTION",
    "WORLD_MODEL",
    "PREDICTION",
    "CODING",
    "PERCEPTION",
    "SKILL_MISSING",
    "CONFIDENCE_ERROR",
    "VERIFICATION",
)


@dataclass
class CycleTrace:
    cycle_id: str
    stages: list[dict[str, Any]] = field(default_factory=list)
    interpretation: dict[str, Any] = field(default_factory=dict)
    goal_id: Optional[str] = None
    plan: list[str] = field(default_factory=list)
    action: str = ""
    observation: str = ""
    prediction_error: float = 0.0
    learned: list[str] = field(default_factory=list)
    text: str = ""
    ok: bool = True
    confidence: float = 0.5
    failure_type: Optional[str] = None
    duration_ms: float = 0.0
    human_level_agi: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MotherBrain:
    """Integrated Finderbay Mother AI runtime."""

    version = "mother_brain_v1"

    def __init__(
        self,
        *,
        store_dir: Path | str = "data/complete_ai/mother",
        enable_llm: bool = False,
        seed_demo: bool = True,
    ) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.product_version = __version__
        self.events = EventBus()
        self.gateway = InputGateway()
        self.goals = GoalManager(self.store_dir / "goals.json")
        self.beliefs = BeliefStore(self.store_dir / "beliefs.json")
        self.workspace = GlobalWorkspace()
        self.symbolic = Brain(name="FinderbayMother")
        try:
            self.symbolic.load_domain("animals")
        except Exception:  # noqa: BLE001
            pass
        self._complete = None
        self._enable_llm = enable_llm
        self._seed_demo = seed_demo
        self.learning_log: list[dict[str, Any]] = []
        self.last_cycle: Optional[CycleTrace] = None
        self.cycles: list[dict[str, Any]] = []
        self.started_at = utc_now()
        self.events.emit("MotherBoot", version=self.product_version, mother=self.version)

    # --- lazy backends --------------------------------------------------------

    @property
    def complete(self):
        if self._complete is None:
            from ..complete_ai import CompleteAI

            self._complete = CompleteAI(
                seed_demo=self._seed_demo,
                enable_llm=self._enable_llm,
                autosave=True,
                persist_path=str(self.store_dir / "complete_session.json"),
            )
        return self._complete

    # --- primary API ----------------------------------------------------------

    def handle(self, text: str, *, context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Run one Mother AI cognitive cycle on natural language (or tasks)."""
        t0 = time.perf_counter()
        trace = CycleTrace(cycle_id=new_id("cyc"))
        raw = (text or "").strip()
        self.events.emit("InputReceived", text=raw[:300])

        # PERCEIVE / UNDERSTAND
        interp = self.gateway.interpret(raw, context=context)
        trace.interpretation = interp.to_dict()
        self._stage(trace, "understand", ok=True, detail=interp.intent)
        self.workspace.set_input(raw)
        self.events.emit("IntentResolved", intent=interp.intent, confidence=interp.confidence)

        # BUILD CONTEXT + RECALL
        recalled = self._recall(raw, interp)
        for r in recalled:
            self.workspace.push("memory", r, score=0.6, source="recall")
        self._stage(trace, "recall", ok=True, detail=f"n={len(recalled)}")

        # IDENTIFY NEED / FORM OBJECTIVE
        goal = None
        if interp.intent in ("inspect", "task", "coding", "planning", "learning") or (
            interp.objective_hint and interp.intent not in ("conversation", "meta", "system_control")
        ):
            goal = self.goals.create(
                interp.objective_hint or raw,
                origin="user",
                priority=0.7 if interp.intent == "inspect" else 0.55,
                success_conditions=["produce_useful_response", "no_crash"],
            )
            trace.goal_id = goal.goal_id
            self.workspace.set_objective(goal.description)
            self.events.emit("GoalCreated", goal_id=goal.goal_id, description=goal.description[:200])
            self._stage(trace, "objective", ok=True, detail=goal.goal_id)
        else:
            self._stage(trace, "objective", ok=True, detail="none")

        # METACOGNITION budget
        meta = self._metacognition(interp)
        self._stage(trace, "metacognition", ok=True, detail=meta.get("decision"))
        if meta.get("abstain"):
            text_out = meta.get("message") or "I need a clearer request."
            trace.text = text_out
            trace.ok = False
            trace.confidence = 0.3
            trace.failure_type = "LANGUAGE_UNDERSTANDING"
            return self._finish(trace, t0)

        # GENERATE CANDIDATE PLANS + PROPOSER / CRITIC / VERIFIER
        proposals = self._propose(interp, goal)
        critique = self._critique(proposals, interp)
        chosen = self._verify_select(proposals, critique, interp)
        trace.plan = list(chosen.get("steps") or [])
        trace.action = str(chosen.get("action") or "respond")
        self.events.emit("PlanCreated", action=trace.action, steps=trace.plan)
        self._stage(trace, "plan", ok=True, detail=trace.action)

        # PREDICT
        predicted = chosen.get("expected") or f"Respond via {trace.action}"
        self.workspace.push("prediction", predicted, score=0.65, source="world_model")

        # EXECUTE
        result = self._execute(chosen, interp, raw)
        observation = str(result.get("text") or result.get("observation") or "")
        trace.observation = observation[:1000]
        self.events.emit("ActionExecuted", action=trace.action, ok=bool(result.get("ok", True)))
        self._stage(trace, "execute", ok=bool(result.get("ok", True)), detail=trace.action)

        # COMPARE PREDICTION VS RESULT
        err = self.workspace.record_prediction(predicted, observation[:240])
        trace.prediction_error = err
        self.events.emit("PredictionError", error=err)
        self._stage(trace, "compare", ok=True, detail=f"err={err}")

        # LEARN / UPDATE BELIEFS / MEMORY
        learned = self._learn(interp, result, err, goal)
        trace.learned = learned
        self._stage(trace, "learn", ok=True, detail=f"n={len(learned)}")

        if goal is not None:
            if result.get("ok", True) and observation:
                self.goals.complete(goal.goal_id, summary=observation[:240])
                self.events.emit("GoalCompleted", goal_id=goal.goal_id)
            else:
                self.goals.fail(goal.goal_id, summary=observation[:240] or "no_observation")
                self.events.emit("GoalFailed", goal_id=goal.goal_id)
                trace.failure_type = result.get("failure_type") or "VERIFICATION"

        trace.text = observation or "Done."
        trace.ok = bool(result.get("ok", True))
        trace.confidence = float(result.get("confidence") or chosen.get("confidence") or interp.confidence)
        return self._finish(trace, t0, extra=result)

    def ask(self, text: str) -> str:
        return self.handle(text)["text"]

    # --- roles ----------------------------------------------------------------

    def _propose(self, interp: InputInterpretation, goal) -> list[dict[str, Any]]:
        intent = interp.intent
        proposals: list[dict[str, Any]] = []
        if intent in ("conversation", "meta"):
            proposals.append(
                {
                    "action": "dialogue",
                    "steps": ["greet_or_meta", "use_symbolic_dialogue"],
                    "expected": "friendly local reply",
                    "confidence": 0.85,
                    "cost": 1,
                }
            )
        if intent == "inspect":
            proposals.append(
                {
                    "action": "inspect_repo",
                    "steps": ["status", "diagnostics", "summarize_gaps"],
                    "expected": "diagnostic report",
                    "confidence": 0.8,
                    "cost": 3,
                }
            )
        if intent == "coding":
            proposals.append(
                {
                    "action": "coding",
                    "steps": ["route_coding_system", "verify_if_possible"],
                    "expected": "code result",
                    "confidence": 0.75,
                    "cost": 4,
                }
            )
        if intent in ("question", "task", "planning", "tool", "vision", "learning", "memory"):
            proposals.append(
                {
                    "action": "complete_ai",
                    "steps": ["think_plan", "route_complete_ai", "observe"],
                    "expected": "task answer via CompleteAI",
                    "confidence": 0.78,
                    "cost": 3,
                }
            )
        # Always keep a dialogue fallback
        proposals.append(
            {
                "action": "dialogue",
                "steps": ["fallback_conversation"],
                "expected": "safe conversational reply",
                "confidence": 0.5,
                "cost": 1,
            }
        )
        self.events.emit("ActionProposed", n=len(proposals), intent=intent)
        return proposals

    def _critique(self, proposals: list[dict[str, Any]], interp: InputInterpretation) -> dict[str, Any]:
        risks = []
        if interp.confidence < 0.4:
            risks.append("low_interpretation_confidence")
        if interp.intent == "inspect":
            risks.append("diagnostics_may_be_partial")
        if not proposals:
            risks.append("no_proposals")
        return {"risks": risks, "prefer_low_cost_if_uncertain": interp.confidence < 0.5}

    def _verify_select(
        self,
        proposals: list[dict[str, Any]],
        critique: dict[str, Any],
        interp: InputInterpretation,
    ) -> dict[str, Any]:
        if not proposals:
            return {"action": "dialogue", "steps": ["fallback"], "expected": "reply", "confidence": 0.4}
        ranked = sorted(
            proposals,
            key=lambda p: (
                -float(p.get("confidence") or 0),
                float(p.get("cost") or 9),
            ),
        )
        chosen = ranked[0]
        if critique.get("prefer_low_cost_if_uncertain"):
            cheap = sorted(proposals, key=lambda p: float(p.get("cost") or 9))[0]
            if float(cheap.get("confidence") or 0) >= 0.45:
                chosen = cheap
        # Verifier: reject inspect if input empty
        if chosen.get("action") == "inspect_repo" and not interp.normalized:
            chosen = {"action": "dialogue", "steps": ["clarify"], "expected": "ask clarify", "confidence": 0.4}
        return chosen

    def _execute(self, chosen: dict[str, Any], interp: InputInterpretation, raw: str) -> dict[str, Any]:
        action = chosen.get("action")
        try:
            if action == "dialogue":
                return self._exec_dialogue(raw, interp)
            if action == "inspect_repo":
                return self._exec_inspect(raw)
            if action == "coding":
                out = self.complete.handle(raw, mode="coding")
                return {
                    "ok": True,
                    "text": out.get("text") or "",
                    "route": out.get("route") or "coding",
                    "confidence": 0.7,
                    "structured": out,
                }
            if action == "complete_ai":
                out = self.complete.handle(raw, mode="auto")
                return {
                    "ok": True,
                    "text": out.get("text") or "",
                    "route": out.get("route") or "complete_ai",
                    "confidence": float((out.get("thinking") or {}).get("confidence") or 0.65),
                    "structured": out,
                }
            return self._exec_dialogue(raw, interp)
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "text": f"I hit an execution error while handling that: {exc}",
                "failure_type": "TOOL_EXECUTION",
                "confidence": 0.2,
            }

    def _exec_dialogue(self, raw: str, interp: InputInterpretation) -> dict[str, Any]:
        lower = raw.lower().strip()
        # Wellbeing / Hinglish greetings beyond symbolic defaults
        if re.search(r"\b(?:how are you|kaise ho|kaisa hai)\b", lower):
            text = (
                "I'm functioning normally as Finderbay Mother AI — local symbolic + tool "
                f"stack v{self.product_version}. I don't feel emotions, but I'm ready to help. "
                "(human_level_agi=false)"
            )
            return {"ok": True, "text": text, "route": "dialogue", "confidence": 0.9}
        if interp.dialogue_act == "greet" or re.match(
            r"^(?:hii+|hi+|hello|hey|yo|namaste|good\s+(?:morning|afternoon|evening))\b",
            lower,
        ):
            text = (
                f"Hello. I am Finderbay Mother AI v{self.product_version}. "
                "I am ready — ask a question, give a task, or say “what can you do”. "
                "(human_level_agi=false)"
            )
            return {"ok": True, "text": text, "route": "dialogue", "confidence": 0.92}
        if interp.intent == "meta" or "what can you do" in lower or "tum kya kar sakte" in lower:
            text = self.greeting()
            return {"ok": True, "text": text, "route": "dialogue", "confidence": 0.9}
        # Prefer CompleteAI for richer answers when available; fall back to symbolic dialogue
        try:
            if interp.intent in ("question",) or len(raw.split()) > 6:
                out = self.complete.handle(raw, mode="auto")
                if out.get("text"):
                    return {
                        "ok": True,
                        "text": out["text"],
                        "route": out.get("route") or "complete_ai",
                        "confidence": 0.7,
                        "structured": out,
                    }
        except Exception:  # noqa: BLE001
            pass
        answer = self.symbolic.ask(raw)
        return {"ok": True, "text": answer, "route": "symbolic_dialogue", "confidence": 0.75}

    def _exec_inspect(self, raw: str) -> dict[str, Any]:
        lines = [
            f"**Finderbay Mother AI diagnostics** (v{self.product_version})",
            "",
            f"Request: {raw}",
            "",
        ]
        try:
            diag = self.complete.diagnostics()
            lines.append(
                f"- CompleteAI diagnostics: ok={diag.get('ok')} "
                f"passed={diag.get('passed')}/{diag.get('total')}"
            )
        except Exception as exc:  # noqa: BLE001
            lines.append(f"- CompleteAI diagnostics error: {exc}")
        try:
            st = self.status()
            lines.append(
                f"- Mother status: goals_active={st['goals']['active']} "
                f"beliefs={st['beliefs']['total']} cycles={st['cycles']}"
            )
            lines.append(f"- Integration: {', '.join(st['capabilities'][:8])}…")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"- Status error: {exc}")
        # Lightweight repo signal
        root = Path.cwd()
        py_count = len(list((root / "cognitive_ai").rglob("*.py"))) if (root / "cognitive_ai").exists() else 0
        test_count = len(list((root / "tests").glob("test_*.py"))) if (root / "tests").exists() else 0
        lines.append(f"- Repository: cognitive_ai py modules≈{py_count}, test files≈{test_count}")
        lines.append("")
        lines.append(
            "Known historical gap (fixed in Mother upgrade): bare REPL treated "
            "`hii` / `how are you` as unknown commands instead of conversation."
        )
        lines.append("")
        lines.append("_human_level_agi=false — diagnostics ≠ AGI._")
        return {"ok": True, "text": "\n".join(lines), "route": "inspect", "confidence": 0.8}

    def _recall(self, raw: str, interp: InputInterpretation) -> list[str]:
        out: list[str] = []
        for g in self.goals.active()[:3]:
            out.append(f"active_goal:{g.description[:80]}")
        for b in self.beliefs.active(limit=3):
            out.append(f"belief:{b['proposition'][:80]}")
        # Session memory via CompleteAI if warm
        if self._complete is not None:
            try:
                hits = self._complete.search_messages(raw.split()[0] if raw.split() else raw, limit=3)
                for h in hits.get("hits") or []:
                    out.append(f"session:{str(h.get('text') or '')[:80]}")
            except Exception:  # noqa: BLE001
                pass
        return out[:8]

    def _metacognition(self, interp: InputInterpretation) -> dict[str, Any]:
        if not interp.normalized:
            return {
                "decision": "abstain",
                "abstain": True,
                "message": "I need some text to work with. Try a greeting, question, or task.",
            }
        if interp.needs_clarification and interp.intent == "unknown":
            self.workspace.note_uncertainty("unclear_intent")
            return {
                "decision": "clarify",
                "abstain": False,
                "message": None,
            }
        questions = []
        if interp.confidence < 0.45:
            questions.append("Should I ask the user to clarify?")
            self.workspace.open_questions = questions
        return {"decision": "proceed", "abstain": False, "questions": questions}

    def _learn(
        self,
        interp: InputInterpretation,
        result: dict[str, Any],
        err: float,
        goal,
    ) -> list[str]:
        learned: list[str] = []
        prop = f"intent:{interp.intent}→action_ok={bool(result.get('ok', True))}"
        self.beliefs.assert_belief(
            prop,
            confidence=0.55 if result.get("ok", True) else 0.35,
            source="mother_cycle",
            kind="observation",
            evidence_refs=[result.get("route") or ""],
        )
        learned.append("belief_updated")
        if err > 0.7:
            rec = {
                "failure_type": result.get("failure_type") or "PREDICTION",
                "intent": interp.intent,
                "error": err,
                "ts": utc_now(),
                "human_level_agi": False,
            }
            self.learning_log.append(rec)
            learned.append("failure_recorded")
            self.events.emit(
                "FailureRecorded",
                failure_type=rec["failure_type"],
                intent=rec["intent"],
                error=rec["error"],
            )
        # Persist compact cycle memory into complete session lightly
        try:
            self.complete.remember("last_mother_intent", interp.intent)
            learned.append("session_memory")
        except Exception:  # noqa: BLE001
            pass
        return learned

    # --- status / greeting ----------------------------------------------------

    def greeting(self) -> str:
        return (
            f"Finderbay Mother AI v{self.product_version} ({self.version})\n"
            "I am a local integrated cognitive runtime — dialogue, goals, memory,\n"
            "planning, tools, coding, vision, and learning — coordinated by Mother Brain.\n"
            "I do not claim human-level AGI.\n\n"
            "Try: hi · how are you · who are you · inspect this project · "
            "Show unpaid invoices for Project A · generate function add(a, b)"
        )

    def status(self) -> dict[str, Any]:
        return {
            "name": "Finderbay Mother AI",
            "mother_version": self.version,
            "product_version": self.product_version,
            "started_at": self.started_at,
            "goals": self.goals.summary(),
            "beliefs": self.beliefs.summary(),
            "workspace": {
                "items": len(self.workspace.items),
                "objective": self.workspace.current_objective,
                "prediction_error": self.workspace.prediction_error,
            },
            "events": self.events.summary(),
            "cycles": len(self.cycles),
            "learning_records": len(self.learning_log),
            "capabilities": [
                "dialogue",
                "goals",
                "memory",
                "beliefs",
                "planning",
                "complete_ai_routing",
                "coding",
                "inspect",
                "metacognition",
                "event_bus",
            ],
            "human_level_agi": False,
            "note": "Integrated coordinator over existing modules — not AGI.",
        }

    def diagnose(self) -> dict[str, Any]:
        return self.handle("inspect this project and find what is broken")

    # --- helpers --------------------------------------------------------------

    def _stage(self, trace: CycleTrace, name: str, *, ok: bool, detail: str = "") -> None:
        trace.stages.append({"stage": name, "ok": ok, "detail": detail, "ts": utc_now()})

    def _finish(self, trace: CycleTrace, t0: float, extra: Optional[dict] = None) -> dict[str, Any]:
        trace.duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        self.last_cycle = trace
        self.cycles.append(trace.to_dict())
        if len(self.cycles) > 100:
            self.cycles = self.cycles[-100:]
        out = {
            "text": trace.text,
            "ok": trace.ok,
            "route": "mother",
            "mother_cycle": trace.to_dict(),
            "goal_id": trace.goal_id,
            "confidence": trace.confidence,
            "workspace": self.workspace.snapshot(),
            "human_level_agi": False,
            "product": "Finderbay Mother AI",
            "version": self.product_version,
            "faithful": True,
        }
        if extra:
            if extra.get("structured"):
                out["structured"] = extra["structured"]
            if extra.get("route"):
                out["backend_route"] = extra["route"]
        return out
