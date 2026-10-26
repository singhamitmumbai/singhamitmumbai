"""Complete non-LLM AI system orchestration (operating + improvement loops)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from .capabilities.spec import FINDERBAY_CAPABILITIES
from .core.records import Observation
from .core.working_state import WorkingState
from .domains.finderbay.package import FinderbayPackage
from .domains.finderbay.seed import seed_memory
from .evaluation.harness import EvaluationHarness, EvalReport
from .execution.executor import Executor
from .execution.tools import ToolRegistry
from .execution.verifier import Verifier
from .goals.manager import GoalManager
from .goals.planner import OperatorPlanner
from .learning_sys.learner import LearningSystem
from .memory.attention import AttentionScheduler
from .memory.store import Episode, MemorySystem
from .nlu.dialogue_tracker import DialogueTracker
from .nlu.entities import EntityLinker
from .nlu.pipeline import LanguageUnderstanding
from .perception.gateway import InputGateway
from .reasoning.engine import ReasoningEngine
from .response.generator import ResponseGenerator
from .runtime.capability_model import CapabilityModel
from .runtime.monitor import RuntimeMonitor
from .world.model import WorldModel
from .version import __version__


class CompleteNonLLMAI:
    """Assistant: understand → remember → reason → plan → act → verify → explain.

    No pretrained general-purpose LLM at runtime.
    """

    def __init__(
        self,
        memory: Optional[MemorySystem] = None,
        storage: Optional[Path | str] = None,
        seed_demo: bool = True,
    ) -> None:
        self.version = __version__
        self.package = FinderbayPackage()
        self.memory = memory or MemorySystem(
            ontology=self.package.ontology(),
            path=storage,
        )
        if seed_demo and not self.memory.records:
            seed_memory(self.memory)

        self.state = WorkingState()
        self.gateway = InputGateway()
        self.nlu = LanguageUnderstanding(entity_linker=EntityLinker(self.memory))
        self.dialogue = DialogueTracker(self.nlu)
        self.attention = AttentionScheduler()
        self.reasoner = ReasoningEngine(self.memory)
        self.world = WorldModel()
        self.goals = GoalManager()
        self.planner = OperatorPlanner()
        self.tools = ToolRegistry()
        self.verifier = Verifier(self.memory)
        self.executor = Executor(
            self.memory, self.reasoner, self.world, self.verifier, self.tools
        )
        self.responder = ResponseGenerator()
        self.learner = LearningSystem(self.memory)
        self.capabilities = FINDERBAY_CAPABILITIES
        self.capability_model = CapabilityModel()
        self.monitor = RuntimeMonitor()
        self.history: list[dict[str, Any]] = []

    # --- operating process ----------------------------------------------------

    def handle(
        self,
        payload: Any,
        *,
        modality: str = "text",
        source: str = "user",
        context: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Full operating cycle for one request."""
        t0 = time.perf_counter()
        obs = self.gateway.ingest(
            payload, modality=modality, source=source, context=context, state=self.state
        )
        text = (
            obs.extracted_value
            if isinstance(obs.extracted_value, str)
            else str(obs.extracted_value or "")
        )
        if obs.uncertainty >= 1.0 and modality == "audio" and not text:
            answer = {
                "text": self.capability_model.describe("speech"),
                "structured": {"error": "acoustic_model_not_loaded"},
                "faithful": True,
                "trace": {"observation": obs.to_dict()},
                "llm_dependency": False,
                "version": self.version,
            }
            self.monitor.record("speech", (time.perf_counter() - t0) * 1000, success=False, abstained=True, error="no_acoustic_model")
            self.history.append(answer)
            return answer

        frame = self.dialogue.process(text, self.state)
        context_pack = self.attention.context_pack(self.state, self.memory)
        goal = self.goals.create_from_frame(frame, self.state)

        # Capability gate
        if frame.operation not in (
            "unsupported",
            "cancel",
            "correct",
            "status",
            "clarify",
        ) and not self.capability_model.can(frame.operation):
            answer = {
                "text": self.capability_model.describe(frame.operation),
                "structured": {"supported": False},
                "faithful": True,
                "frame": frame.to_dict(),
                "llm_dependency": False,
                "version": self.version,
            }
            self.monitor.record(frame.operation, (time.perf_counter() - t0) * 1000, success=False, abstained=True)
            self.history.append(answer)
            return answer

        if frame.operation in ("cancel", "correct", "status", "unsupported") or frame.needs_clarification:
            plan = self.planner.plan(goal, self.state)
            if plan.outcome == "missing_info":
                self.executor.run_plan(plan, self.state)
            answer = self.responder.generate(self.state, self.executor)
            abstained = bool(frame.needs_clarification or frame.operation == "unsupported")
            if frame.operation in ("cancel", "correct", "status"):
                self.goals.mark(self.state, "satisfied")
            elif abstained:
                self.goals.mark(self.state, "missing_info" if frame.needs_clarification else "unsupported")
            result = self._package_result(answer, frame, plan, context_pack, obs)
            self._finish(result, goal, t0, abstained=abstained)
            return result

        plan = self.planner.plan(goal, self.state)
        action_results = self.executor.run_plan(plan, self.state)
        answer = self.responder.generate(self.state, self.executor, self.executor.last_reasoning)

        if answer.get("faithful") and plan.outcome in ("feasible", "optimal", "already_satisfied"):
            failed = any(r.status.value == "FAILED" for r in action_results)
            self.goals.mark(self.state, "failed" if failed else "satisfied")
        elif plan.outcome in ("missing_info", "unsupported"):
            self.goals.mark(self.state, plan.outcome)
        else:
            failed = any(r.status.value == "FAILED" for r in action_results)
            self.goals.mark(self.state, "failed" if failed else "satisfied")

        result = self._package_result(answer, frame, plan, context_pack, obs)
        result["actions"] = [r.to_dict() for r in action_results]
        self.memory.add_episode(
            Episode(
                summary=text[:200],
                observations=[obs.to_dict()],
                actions=result.get("actions") or [],
                outcome=goal.status,
            )
        )
        self._finish(result, goal, t0, abstained=False)
        return result

    def _finish(self, result: dict, goal, t0: float, *, abstained: bool) -> None:
        latency = (time.perf_counter() - t0) * 1000
        op = (result.get("frame") or {}).get("operation") or "unknown"
        success = goal.status == "satisfied" or (
            abstained and (result.get("faithful") is True)
        )
        # Clarifications / unsupported are successful abstentions when faithful
        if abstained:
            success = bool(result.get("faithful"))
        self.monitor.record(op, latency, success=success, abstained=abstained)
        self.history.append(result)

    def ask(self, text: str) -> str:
        return self.handle(text, modality="text")["text"]

    def _package_result(self, answer, frame, plan, context_pack, obs: Observation) -> dict[str, Any]:
        return {
            "text": answer.get("text"),
            "structured": answer.get("structured"),
            "faithful": answer.get("faithful"),
            "table": answer.get("table"),
            "frame": frame.to_dict(),
            "plan": plan.to_dict(),
            "goal": self.state.active_goal().to_dict() if self.state.active_goal() else None,
            "state": self.state.summary(),
            "attention": context_pack.get("attention"),
            "observation": obs.to_dict(),
            "capabilities": self.capability_model.to_dict(),
            "monitor": self.monitor.summary(),
            "version": self.version,
            "llm_dependency": False,
        }

    # --- improvement / evaluation ---------------------------------------------

    def evaluate(self) -> EvalReport:
        def run_fn(text: str) -> dict[str, Any]:
            self.state = WorkingState()
            self.executor.last_reasoning = None
            self.executor.last_table = None
            self.executor.last_document = None
            self.executor.last_task = None
            self.executor.last_reservation = None
            return self.handle(text)

        harness = EvaluationHarness(run_fn)
        report = harness.run_suite(self.package.evaluation_cases())
        # Update capability model with overall accuracy
        self.capability_model.update_eval("query_invoices", report.accuracy)
        self.capability_model.update_eval("prepare_followup", report.accuracy)
        return report

    def status(self) -> dict[str, Any]:
        return {
            "name": "CompleteNonLLMAI",
            "version": self.version,
            "domain": self.package.name,
            "llm_at_runtime": False,
            "capabilities": [c.name for c in self.capabilities],
            "capability_model": self.capability_model.to_dict(),
            "monitor": self.monitor.summary(),
            "memory": self.memory.summary(),
            "state": self.state.summary(),
            "package": self.package.manifest(),
            "tools": list(self.tools.tools.keys()),
        }

    def save(self, path: Optional[Path | str] = None) -> Path:
        return self.memory.save(path)

    def reset_session(self) -> None:
        self.state = WorkingState()
        self.executor.last_reasoning = None
        self.executor.last_table = None
        self.executor.last_document = None
        self.executor.last_task = None
        self.executor.last_reservation = None
