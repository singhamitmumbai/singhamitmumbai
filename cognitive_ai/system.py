"""Complete non-LLM AI system orchestration (operating + improvement loops)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from .capabilities.spec import FINDERBAY_CAPABILITIES
from .config import SystemConfig
from .core.records import Observation
from .core.working_state import WorkingState
from .datasets.registry import DatasetRegistry
from .domains.finderbay.auth import AuthService
from .domains.finderbay.contracts import ContractService
from .domains.finderbay.package import FinderbayPackage
from .domains.finderbay.payments import PaymentService
from .domains.finderbay.purchase_orders import PurchaseOrderService
from .domains.finderbay.seed import seed_memory
from .reporting.exporter import ReportExporter
from .runtime.alerts import ProactiveMonitor
from .evaluation.harness import EvaluationHarness, EvalCase, EvalReport
from .execution.executor import Executor
from .execution.recovery import RecoveryManager
from .execution.tools import ToolRegistry
from .execution.verifier import Verifier
from .goals.manager import GoalManager
from .goals.planner import OperatorPlanner
from .blueprint.audit import run_blueprint_audit
from .learning_sys.improvement import ImprovementLoop
from .learning_sys.induction import ProgramInducer
from .learning_sys.learner import LearningSystem
from .memory.attention import AttentionScheduler
from .memory.store import Episode, MemorySystem
from .nlu.dialogue_tracker import DialogueTracker
from .nlu.entities import EntityLinker
from .nlu.pipeline import LanguageUnderstanding
from .perception.gateway import InputGateway
from .reasoning.cases import CaseBasedReasoner
from .reasoning.conflicts import ConflictDetector
from .reasoning.engine import ReasoningEngine
from .response.generator import ResponseGenerator
from .runtime.capability_model import CapabilityModel
from .runtime.cycle import OperatingCycle
from .runtime.monitor import RuntimeMonitor
from .uncertainty.calibration import calibrate_report
from .uncertainty.router import UncertaintyRouter
from .vision.system import VisionSystem
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
        config: Optional[SystemConfig] = None,
    ) -> None:
        self.config = config or SystemConfig(
            seed_demo=seed_demo,
            storage_path=str(storage) if storage else None,
        )
        self.version = __version__
        self.package = FinderbayPackage()
        self.memory = memory or MemorySystem(
            ontology=self.package.ontology(),
            path=self.config.storage(),
        )
        if self.config.seed_demo and not self.memory.records:
            seed_memory(self.memory)

        self.state = WorkingState()
        self.state.resource_budget = {
            "time_ms": self.config.resource_time_ms,
            "steps": self.config.resource_steps,
            "external_calls": self.config.resource_external_calls,
        }
        self.vision = VisionSystem()
        self.gateway = InputGateway(vision=self.vision)
        self.nlu = LanguageUnderstanding(entity_linker=EntityLinker(self.memory))
        self.dialogue = DialogueTracker(self.nlu)
        self.attention = AttentionScheduler()
        self.reasoner = ReasoningEngine(self.memory)
        self.world = WorldModel()
        self.goals = GoalManager()
        self.planner = OperatorPlanner()
        self.tools = ToolRegistry()
        self.verifier = Verifier(self.memory)
        self.recovery = RecoveryManager()
        self.executor = Executor(
            self.memory,
            self.reasoner,
            self.world,
            self.verifier,
            self.tools,
            self.recovery,
            allow_send=self.config.allow_send,
            allow_inventory_write=self.config.allow_inventory_write,
        )
        self.executor.vision = self.vision
        self.auth = AuthService()
        self.payments = PaymentService(self.memory, self.auth)
        self.contracts = ContractService(self.memory, self.auth)
        self.purchase_orders = PurchaseOrderService(self.memory, self.auth)
        self.alerts = ProactiveMonitor(self.memory)
        self.reports = ReportExporter(self.memory, self.alerts)
        self.executor.auth = self.auth
        self.executor.payments = self.payments
        self.executor.contracts = self.contracts
        self.executor.purchase_orders = self.purchase_orders
        self.executor.alerts = self.alerts
        self.executor.reports = self.reports
        self.responder = ResponseGenerator()
        self.learner = LearningSystem(self.memory, dataset_dir=self.config.dataset_dir)
        self.improvement = ImprovementLoop()
        self.inducer = ProgramInducer()
        self.conflicts = ConflictDetector()
        self.cases = CaseBasedReasoner()
        self.uncertainty = UncertaintyRouter()
        self.datasets = DatasetRegistry(self.config.dataset_dir)
        self.capabilities = FINDERBAY_CAPABILITIES
        self.capability_model = CapabilityModel()
        self.monitor = RuntimeMonitor()
        self.history: list[dict[str, Any]] = []
        self.last_cycle = None
        self._bootstrap_datasets()

    def _bootstrap_datasets(self) -> None:
        """Seed reviewed eval examples once (idempotent via source groups)."""
        if self.datasets.summary()["total"] > 0:
            return
        for case in self.package.evaluation_cases():
            try:
                self.datasets.make(
                    task=case.task,
                    input_data=case.input_text,
                    verified_target=case.expected,
                    split="test",
                    source_group=f"eval:{case.case_id}",
                    reviewer="suite",
                )
            except ValueError:
                continue

    # --- operating process ----------------------------------------------------

    def handle(
        self,
        payload: Any,
        *,
        modality: str = "text",
        source: str = "user",
        context: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Full operating cycle: perceive → understand → attend → reason → plan → act → verify → respond."""
        t0 = time.perf_counter()
        cycle = OperatingCycle()

        # 1. Perceive
        t_stage = time.perf_counter()
        ctx = dict(context or {})
        # Allow image path + vision task via context when modality=vision
        if modality == "vision" and "vision_task" not in ctx:
            ctx["vision_task"] = "detect_objects"
        obs = self.gateway.ingest(
            payload, modality=modality, source=source, context=ctx, state=self.state
        )
        text = (
            obs.extracted_value
            if isinstance(obs.extracted_value, str)
            else str(obs.extracted_value or "")
        )
        # If caller passed a vision image but typed a natural-language ask in context
        if modality == "vision" and ctx.get("ask"):
            text = str(ctx["ask"])
        elif modality == "vision" and isinstance(obs.context.get("vision_result"), dict):
            # Keep a usable NL ask so NLU can route; stash result for executor
            self.executor.last_vision = obs.context["vision_result"]
            task = ctx.get("vision_task") or "detect_objects"
            text = ctx.get("ask") or f"{task.replace('_', ' ')} in image {payload}"
        cycle.stage(
            "perceive",
            ok=True,
            summary=f"modality={modality}",
            data={"observation_id": obs.observation_id, "uncertainty": obs.uncertainty},
            duration_ms=(time.perf_counter() - t_stage) * 1000,
            uncertainties=["recognition"] if obs.uncertainty >= 0.5 else [],
        )
        if obs.uncertainty >= 1.0 and modality == "audio" and not text:
            self.uncertainty.note(self.state, "recognition", "Acoustic model produced no transcript.")
            answer = {
                "text": self.capability_model.describe("speech"),
                "structured": {"error": "acoustic_model_not_loaded"},
                "faithful": True,
                "trace": {"observation": obs.to_dict()},
                "cycle": cycle.finish("abstained").to_dict(),
                "uncertainty": self.uncertainty.report(self.state),
                "llm_dependency": False,
                "version": self.version,
                "human_level_agi": False,
            }
            self.monitor.record(
                "speech",
                (time.perf_counter() - t0) * 1000,
                success=False,
                abstained=True,
                error="no_acoustic_model",
            )
            self.history.append(answer)
            self.last_cycle = cycle.trace
            return answer

        # 2. Understand
        t_stage = time.perf_counter()
        frame = self.dialogue.process(text, self.state)
        cycle.stage(
            "understand",
            ok=frame.supported or frame.operation in ("cancel", "correct", "status"),
            summary=f"op={frame.operation} conf={frame.confidence:.2f}",
            data={
                "operation": frame.operation,
                "project_id": frame.project_id,
                "question": frame.question,
                "needs_clarification": list(frame.needs_clarification or []),
            },
            duration_ms=(time.perf_counter() - t_stage) * 1000,
        )

        # 3. Attend + conflicts + cases
        t_stage = time.perf_counter()
        context_pack = self.attention.context_pack(self.state, self.memory)
        conflict_reports = self.conflicts.scan_memory_claims(self.memory)
        conflict_reports.extend(self.conflicts.scan_state(self.state))
        if conflict_reports:
            self.conflicts.apply_to_state(self.state, conflict_reports)
        case_matches = self.cases.retrieve(self.memory, text, operation=frame.operation)
        case_hint = self.cases.suggest_plan_hint(case_matches)
        cycle.stage(
            "attend",
            ok=True,
            summary=f"attention={len(context_pack.get('attention') or [])} conflicts={len(conflict_reports)} cases={len(case_matches)}",
            data={
                "conflicts": [c.to_dict() for c in conflict_reports[:5]],
                "cases": [c.to_dict() for c in case_matches],
                "case_hint": case_hint,
            },
            duration_ms=(time.perf_counter() - t_stage) * 1000,
            uncertainties=["conflicting_evidence"] if conflict_reports else [],
        )

        # 4. Goal
        t_stage = time.perf_counter()
        goal = self.goals.create_from_frame(frame, self.state)
        cycle.stage(
            "goal",
            ok=True,
            summary=goal.description[:120] if goal else "",
            data={"goal_id": goal.goal_id if goal else None, "status": goal.status if goal else None},
            duration_ms=(time.perf_counter() - t_stage) * 1000,
        )

        if frame.operation not in (
            "unsupported",
            "cancel",
            "correct",
            "status",
            "clarify",
            "learn_alias",
            "login",
            "query_payments",
            "query_contracts",
            "record_payment",
            "query_alerts",
            "query_purchase_orders",
            "create_purchase_order",
            "query_vendors",
            "export_report",
        ) and not self.capability_model.can(frame.operation):
            self.uncertainty.note(self.state, "unsupported_task", f"Capability '{frame.operation}' unavailable.")
            answer = {
                "text": self.capability_model.describe(frame.operation),
                "structured": {"supported": False},
                "faithful": True,
                "frame": frame.to_dict(),
                "cycle": cycle.finish("unsupported").to_dict(),
                "uncertainty": self.uncertainty.report(self.state),
                "llm_dependency": False,
                "version": self.version,
                "human_level_agi": False,
            }
            self.improvement.record_failure(
                operation=frame.operation,
                input_text=text,
                reason="unsupported_capability",
                stage="understand",
            )
            self.monitor.record(
                frame.operation,
                (time.perf_counter() - t0) * 1000,
                success=False,
                abstained=True,
            )
            self.history.append(answer)
            self.last_cycle = cycle.trace
            return answer

        if frame.operation in ("cancel", "correct", "status", "unsupported") or frame.needs_clarification:
            if frame.needs_clarification:
                self.uncertainty.note(
                    self.state,
                    "missing_information",
                    "; ".join(frame.needs_clarification),
                )
            plan = self.planner.plan(goal, self.state)
            cycle.stage("plan", ok=True, summary=plan.outcome, data={"plan_id": plan.plan_id})
            if plan.outcome == "missing_info":
                self.executor.run_plan(plan, self.state)
            cycle.stage("execute", ok=True, summary="clarify_or_status")
            cycle.stage("verify", ok=True, summary="n/a")
            answer = self.responder.generate(self.state, self.executor)
            cycle.stage("respond", ok=True, summary=(answer.get("text") or "")[:80])
            abstained = bool(frame.needs_clarification or frame.operation == "unsupported")
            if frame.operation in ("cancel", "correct", "status"):
                self.goals.mark(self.state, "satisfied")
                outcome = "satisfied"
            elif abstained:
                self.goals.mark(
                    self.state,
                    "missing_info" if frame.needs_clarification else "unsupported",
                )
                outcome = "missing_info" if frame.needs_clarification else "unsupported"
            else:
                outcome = goal.status
            if frame.operation == "unsupported":
                self.improvement.record_failure(
                    operation=frame.operation,
                    input_text=text,
                    reason="unsupported_request",
                    stage="understand",
                )
            result = self._package_result(answer, frame, plan, context_pack, obs, cycle, case_hint, conflict_reports)
            cycle.finish(outcome)
            result["cycle"] = cycle.trace.to_dict()
            self._finish(result, goal, t0, abstained=abstained)
            self.last_cycle = cycle.trace
            return result

        # 5–8. Plan → execute → verify → respond
        t_stage = time.perf_counter()
        plan = self.planner.plan(goal, self.state)
        if case_hint and case_hint.get("suggested_operators"):
            plan.explanation = (plan.explanation or "") + f" | case_hint={case_hint['from_case']}"
        cycle.stage(
            "plan",
            ok=plan.outcome in ("feasible", "optimal", "already_satisfied", "missing_info"),
            summary=plan.outcome,
            data={"steps": len(plan.steps), "case_hint": case_hint},
            duration_ms=(time.perf_counter() - t_stage) * 1000,
        )

        t_stage = time.perf_counter()
        action_results = self.executor.run_plan(plan, self.state)
        failed = any(r.status.value == "FAILED" for r in action_results)
        unknown = any(r.status.value == "OUTCOME_UNKNOWN" for r in action_results)
        cycle.stage(
            "execute",
            ok=not failed,
            summary=f"actions={len(action_results)} failed={failed} unknown={unknown}",
            data={"action_ids": [r.attempt_id for r in action_results]},
            duration_ms=(time.perf_counter() - t_stage) * 1000,
        )
        cycle.stage(
            "verify",
            ok=not failed and not unknown,
            summary="OUTCOME_UNKNOWN" if unknown else ("FAILED" if failed else "CONFIRMED"),
            uncertainties=["stale_information"] if unknown else [],
        )
        if unknown:
            self.uncertainty.note(
                self.state,
                "stale_information",
                "At least one action returned OUTCOME_UNKNOWN — reconcile before retry.",
            )

        answer = self.responder.generate(self.state, self.executor, self.executor.last_reasoning)
        # Surface conflict note in text when present (skip unrelated alias noise for vision)
        vision_ops = {
            "understand_scene",
            "detect_objects",
            "detect_vehicles",
            "count_people",
            "detect_fire",
            "scan_document_vision",
            "vision_status",
            "ppe_compliance",
            "estimate_queue",
        }
        if conflict_reports and answer.get("text") and frame.operation not in vision_ops:
            answer["text"] = (
                answer["text"]
                + " Note: "
                + conflict_reports[0].message
            )
        cycle.stage("respond", ok=bool(answer.get("faithful")), summary=(answer.get("text") or "")[:100])

        if answer.get("faithful") and plan.outcome in ("feasible", "optimal", "already_satisfied"):
            self.goals.mark(self.state, "failed" if failed else "satisfied")
        elif plan.outcome in ("missing_info", "unsupported"):
            self.goals.mark(self.state, plan.outcome)
        else:
            self.goals.mark(self.state, "failed" if failed else "satisfied")

        # 9. Learning hook (record episode + failure if needed)
        if failed or goal.status in ("failed", "unsupported"):
            fail = self.improvement.record_failure(
                operation=frame.operation,
                input_text=text,
                reason=f"goal_status={goal.status}",
                stage="execute" if failed else "plan",
            )
            self.improvement.propose_from_failure(fail)
            cycle.stage("learn_hook", ok=True, summary=f"failure={fail.failure_id}")
        else:
            cycle.stage("learn_hook", ok=True, summary="episode_logged" if self.config.log_episodes else "noop")

        result = self._package_result(answer, frame, plan, context_pack, obs, cycle, case_hint, conflict_reports)
        result["actions"] = [r.to_dict() for r in action_results]
        if self.config.log_episodes:
            self.memory.add_episode(
                Episode(
                    summary=text[:200],
                    observations=[obs.to_dict()],
                    actions=result.get("actions") or [],
                    outcome=goal.status,
                )
            )
        cycle.finish(goal.status if goal else "satisfied")
        result["cycle"] = cycle.trace.to_dict()
        self._finish(result, goal, t0, abstained=False)
        self.last_cycle = cycle.trace
        return result

    def _finish(self, result: dict, goal, t0: float, *, abstained: bool) -> None:
        latency = (time.perf_counter() - t0) * 1000
        op = (result.get("frame") or {}).get("operation") or "unknown"
        success = goal.status == "satisfied"
        if abstained:
            success = bool(result.get("faithful"))
        self.monitor.record(op, latency, success=success, abstained=abstained)
        self.history.append(result)

    def ask(self, text: str) -> str:
        return self.handle(text, modality="text")["text"]

    def _package_result(
        self,
        answer,
        frame,
        plan,
        context_pack,
        obs: Observation,
        cycle: Optional[OperatingCycle] = None,
        case_hint: Optional[dict] = None,
        conflict_reports: Optional[list] = None,
    ) -> dict[str, Any]:
        return {
            "text": answer.get("text"),
            "structured": answer.get("structured"),
            "faithful": answer.get("faithful"),
            "table": answer.get("table"),
            "language": answer.get("language") or frame.language,
            "frame": frame.to_dict(),
            "plan": plan.to_dict(),
            "goal": self.state.active_goal().to_dict() if self.state.active_goal() else None,
            "state": self.state.summary(),
            "attention": context_pack.get("attention"),
            "observation": obs.to_dict(),
            "capabilities": self.capability_model.to_dict(),
            "monitor": self.monitor.summary(),
            "uncertainty": self.uncertainty.report(self.state),
            "conflicts": [c.to_dict() for c in (conflict_reports or [])[:5]],
            "case_hint": case_hint,
            "improvement": self.improvement.summary(),
            "cycle": cycle.trace.to_dict() if cycle else None,
            "distinctions": {
                "unknown_vs_false": True,
                "predicted_vs_observed": True,
                "attempted_vs_confirmed": True,
            },
            "version": self.version,
            "llm_dependency": False,
            "human_level_agi": False,
        }

    # --- improvement / evaluation ---------------------------------------------

    def evaluate(self) -> EvalReport:
        def run_fn(text: str) -> dict[str, Any]:
            self.reset_session()
            return self.handle(text)

        harness = EvaluationHarness(run_fn)
        report = harness.run_suite(self.package.evaluation_cases())
        self.capability_model.update_eval("query_invoices", report.accuracy)
        self.capability_model.update_eval("prepare_followup", report.accuracy)
        return report

    def evaluate_with_oracle(self) -> EvalReport:
        """Run suite and attach oracle failure localization for speech-ish cases."""
        report = self.evaluate()
        harness = EvaluationHarness(lambda t: self.handle(t))
        oracles = []
        for case in self.package.evaluation_cases():
            if "speech" not in case.tags and case.modality != "audio":
                continue
            self.reset_session()
            full = self.handle({"transcript": case.input_text, "uncertainty": 0.2}, modality="audio")
            self.reset_session()
            oracle = self.handle(case.input_text, modality="text")
            oracles.append(harness.locate_failure(case, full, oracle))
        report.oracle_comparisons = oracles
        return report

    def calibrate_intent_scores(self) -> dict[str, Any]:
        """Calibration over suite: intent confidence vs case pass/fail."""
        probs: list[float] = []
        outcomes: list[int] = []
        harness = EvaluationHarness(lambda t: {})
        for case in self.package.evaluation_cases():
            self.reset_session()
            out = self.handle(case.input_text)
            conf = float((out.get("frame") or {}).get("confidence") or 0.0)
            passed, _ = harness._check(case, out)
            probs.append(conf)
            outcomes.append(1 if passed else 0)
        return calibrate_report(probs, outcomes).to_dict()

    def status(self) -> dict[str, Any]:
        return {
            "name": "CompleteNonLLMAI",
            "version": self.version,
            "domain": self.package.name,
            "llm_at_runtime": False,
            "human_level_agi": False,
            "config": self.config.to_dict(),
            "capabilities": [c.name for c in self.capabilities],
            "capability_model": self.capability_model.to_dict(),
            "vision": self.vision.status(),
            "monitor": self.monitor.summary(),
            "memory": self.memory.summary(),
            "datasets": self.datasets.summary(),
            "recovery_unknowns": len(self.recovery.pending_unknowns()),
            "state": self.state.summary(),
            "package": self.package.manifest(),
            "tools": list(self.tools.tools.keys()),
            "htn_tasks": list(self.planner.htn.methods.keys()),
            "improvement": self.improvement.summary(),
            "world_transitions": list(self.world.transitions.keys()),
            "last_cycle_id": self.last_cycle.cycle_id if self.last_cycle else None,
        }

    def blueprint_audit(self) -> dict[str, Any]:
        """Honest inventory of blueprint components vs implementation."""
        return run_blueprint_audit()

    def save(self, path: Optional[Path | str] = None) -> Path:
        return self.memory.save(path)

    def reconcile(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.recovery.reconcile_all(self.verifier)]

    def reset_session(self) -> None:
        self.state = WorkingState()
        self.state.resource_budget = {
            "time_ms": self.config.resource_time_ms,
            "steps": self.config.resource_steps,
            "external_calls": self.config.resource_external_calls,
        }
        self.executor.last_reasoning = None
        self.executor.last_table = None
        self.executor.last_document = None
        self.executor.last_task = None
        self.executor.last_reservation = None
        self.executor.last_alias = None
