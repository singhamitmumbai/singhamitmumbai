"""Complete non-LLM AI system orchestration (operating + improvement loops)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .capabilities.spec import FINDERBAY_CAPABILITIES
from .core.records import Observation
from .core.working_state import WorkingState
from .domains.finderbay.package import FinderbayPackage
from .domains.finderbay.seed import seed_memory
from .evaluation.harness import EvaluationHarness, EvalReport
from .execution.executor import Executor
from .execution.verifier import Verifier
from .goals.manager import GoalManager
from .goals.planner import OperatorPlanner
from .learning_sys.learner import LearningSystem
from .memory.attention import AttentionScheduler
from .memory.store import Episode, MemorySystem
from .nlu.dialogue_tracker import DialogueTracker
from .perception.gateway import InputGateway
from .reasoning.engine import ReasoningEngine
from .response.generator import ResponseGenerator
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
        self.dialogue = DialogueTracker()
        self.attention = AttentionScheduler()
        self.reasoner = ReasoningEngine(self.memory)
        self.world = WorldModel()
        self.goals = GoalManager()
        self.planner = OperatorPlanner()
        self.verifier = Verifier(self.memory)
        self.executor = Executor(self.memory, self.reasoner, self.world, self.verifier)
        self.responder = ResponseGenerator()
        self.learner = LearningSystem(self.memory)
        self.capabilities = FINDERBAY_CAPABILITIES
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
        # 1. Perceive
        obs = self.gateway.ingest(
            payload, modality=modality, source=source, context=context, state=self.state
        )
        text = obs.extracted_value if isinstance(obs.extracted_value, str) else str(obs.extracted_value or "")
        if obs.uncertainty >= 1.0 and modality == "audio" and not text:
            answer = {
                "text": "I can accept audio once an audited non-LLM acoustic model is loaded. "
                "Please provide text, or pass {\"transcript\": \"...\"} for offline tests.",
                "structured": {"error": "acoustic_model_not_loaded"},
                "faithful": True,
                "trace": {"observation": obs.to_dict()},
            }
            self.history.append(answer)
            return answer

        # 2. Language understanding + dialogue state
        frame = self.dialogue.process(text, self.state)

        # 3. Attention / context selection
        context_pack = self.attention.context_pack(self.state, self.memory)

        # Early clarify / cancel / status / correct paths still get goals where useful
        goal = self.goals.create_from_frame(frame, self.state)

        if frame.operation in ("cancel", "correct", "status", "unsupported") or frame.needs_clarification:
            # Lightweight plan
            plan = self.planner.plan(goal, self.state)
            if plan.outcome == "missing_info":
                self.executor.run_plan(plan, self.state)
            answer = self.responder.generate(self.state, self.executor)
            result = self._package_result(answer, frame, plan, context_pack, obs)
            self.history.append(result)
            return result

        # 4. Plan
        plan = self.planner.plan(goal, self.state)

        # 5. Execute + verify
        action_results = self.executor.run_plan(plan, self.state)

        # 6. Respond
        answer = self.responder.generate(self.state, self.executor, self.executor.last_reasoning)

        # 7. Mark goal
        if answer.get("faithful") and plan.outcome in ("feasible", "optimal", "already_satisfied"):
            self.goals.mark(self.state, "satisfied")
        elif plan.outcome in ("missing_info", "unsupported"):
            self.goals.mark(self.state, plan.outcome)
        else:
            failed = any(r.status.value == "FAILED" for r in action_results)
            self.goals.mark(self.state, "failed" if failed else "satisfied")

        result = self._package_result(answer, frame, plan, context_pack, obs)
        result["actions"] = [r.to_dict() for r in action_results]
        self.history.append(result)

        # 8. Reviewed experience hook (improvement process entry)
        self.memory.add_episode(
            Episode(
                summary=text[:200],
                observations=[obs.to_dict()],
                actions=result.get("actions") or [],
                outcome=goal.status,
            )
        )
        return result

    def ask(self, text: str) -> str:
        """Convenience: return natural-language answer text."""
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
            "version": self.version,
            "llm_dependency": False,
        }

    # --- improvement / evaluation ---------------------------------------------

    def evaluate(self) -> EvalReport:
        def run_fn(text: str) -> dict[str, Any]:
            # Fresh dialogue state per case but shared memory
            self.state = WorkingState()
            self.executor.last_reasoning = None
            self.executor.last_table = None
            return self.handle(text)

        harness = EvaluationHarness(run_fn)
        return harness.run_suite(self.package.evaluation_cases())

    def status(self) -> dict[str, Any]:
        return {
            "name": "CompleteNonLLMAI",
            "version": self.version,
            "domain": self.package.name,
            "llm_at_runtime": False,
            "capabilities": [c.name for c in self.capabilities],
            "memory": self.memory.summary(),
            "state": self.state.summary(),
            "package": self.package.manifest(),
        }

    def save(self, path: Optional[Path | str] = None) -> Path:
        return self.memory.save(path)
