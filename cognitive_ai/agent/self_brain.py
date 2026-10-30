"""SelfBrain: autonomous multi-job executor driven by Finderbay AI goals.

Runs a bounded perceive→plan→act→verify→reflect loop over a task queue.
Non-LLM: uses CompleteNonLLMAI operators, world model, verifier, and coding tools.
Not human-level AGI — scoped automation under declared capabilities.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from ..core.ids import new_id
from ..core.records import utc_now
from ..system import CompleteNonLLMAI


@dataclass
class AgentJob:
    """A unit of work issued to the SelfBrain (typically from Finderbay AI)."""

    instruction: str
    job_id: str = field(default_factory=lambda: new_id("job"))
    priority: float = 1.0
    source: str = "finderbay"
    status: str = "queued"  # queued|running|succeeded|failed|skipped|cancelled
    max_attempts: int = 2
    attempts: int = 0
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=utc_now)
    finished_at: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentReport:
    run_id: str
    jobs: list[dict[str, Any]] = field(default_factory=list)
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    reflections: list[str] = field(default_factory=list)
    stopped_reason: str = ""
    duration_ms: float = 0.0
    llm_at_runtime: bool = False
    human_level_agi: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SelfBrainAgent:
    """Autonomous agent that drains a Finderbay task queue with verify/retry."""

    def __init__(
        self,
        ai: Optional[CompleteNonLLMAI] = None,
        *,
        seed_demo: bool = True,
        coding=None,
    ) -> None:
        self.ai = ai or CompleteNonLLMAI(seed_demo=seed_demo)
        self.queue: list[AgentJob] = []
        self.history: list[AgentJob] = []
        self.reflections: list[str] = []
        self.coding = coding  # optional CodingSystem
        self.running = False

    # --- queue API ------------------------------------------------------------

    def submit(
        self,
        instruction: str,
        *,
        priority: float = 1.0,
        source: str = "finderbay",
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> AgentJob:
        job = AgentJob(
            instruction=instruction,
            priority=priority,
            source=source,
            tags=list(tags or []),
            metadata=dict(metadata or {}),
        )
        self.queue.append(job)
        self.queue.sort(key=lambda j: -j.priority)
        return job

    def submit_many(self, instructions: list[str], *, source: str = "finderbay") -> list[AgentJob]:
        return [self.submit(i, source=source) for i in instructions]

    def submit_from_finderbay(self, text: str, **kwargs) -> AgentJob:
        """Accept a natural-language task from the Finderbay AI surface."""
        return self.submit(text, source="finderbay", **kwargs)

    # --- autonomous execution -------------------------------------------------

    def run(
        self,
        *,
        max_jobs: int = 20,
        stop_on_failure: bool = False,
        reset_session_each_job: bool = True,
    ) -> AgentReport:
        """Execute queued jobs until empty or max_jobs reached."""
        t0 = time.perf_counter()
        run_id = new_id("run")
        self.running = True
        report = AgentReport(run_id=run_id)
        processed = 0

        try:
            while self.queue and processed < max_jobs:
                job = self.queue.pop(0)
                processed += 1
                self._execute_job(job, reset_session=reset_session_each_job)
                self.history.append(job)
                report.jobs.append(job.to_dict())
                if job.status == "succeeded":
                    report.succeeded += 1
                elif job.status == "failed":
                    report.failed += 1
                    if stop_on_failure:
                        report.stopped_reason = f"stop_on_failure:{job.job_id}"
                        break
                else:
                    report.skipped += 1
            else:
                if not self.queue:
                    report.stopped_reason = "queue_empty"
                else:
                    report.stopped_reason = f"max_jobs={max_jobs}"
        finally:
            self.running = False

        report.reflections = list(self.reflections[-20:])
        report.duration_ms = (time.perf_counter() - t0) * 1000
        return report

    def run_until_idle(self, *, max_jobs: int = 50) -> AgentReport:
        return self.run(max_jobs=max_jobs, stop_on_failure=False)

    def _execute_job(self, job: AgentJob, *, reset_session: bool) -> None:
        job.status = "running"
        job.attempts += 1
        if reset_session:
            self.ai.reset_session()

        # Coding jobs route to CodingSystem when available
        if self._is_coding_job(job) and self.coding is not None:
            try:
                out = self.coding.handle(job.instruction, metadata=job.metadata)
                job.result = out
                ok = bool(out.get("ok", out.get("success", False)))
                job.status = "succeeded" if ok else "failed"
                if not ok:
                    job.error = out.get("error") or out.get("message") or "coding_failed"
                    self._reflect(job, "coding_tool_reported_failure")
            except Exception as exc:  # noqa: BLE001 — agent must not crash the queue
                job.status = "failed"
                job.error = str(exc)
                self._reflect(job, f"coding_exception:{exc}")
            job.finished_at = utc_now()
            return

        # Vision-tagged jobs still go through CompleteNonLLMAI (YOLO adapters)
        if self._is_vision_job(job):
            job.tags = list(dict.fromkeys([*(job.tags or []), "vision"]))

        # Finderbay / general task via CompleteNonLLMAI
        last_err = None
        while job.attempts <= job.max_attempts:
            try:
                result = self.ai.handle(job.instruction)
                job.result = {
                    "text": result.get("text"),
                    "frame": result.get("frame"),
                    "structured": result.get("structured"),
                    "faithful": result.get("faithful"),
                    "cycle": result.get("cycle"),
                    "goal": result.get("goal"),
                }
                op = (result.get("frame") or {}).get("operation")
                goal_status = (result.get("goal") or {}).get("status")
                if op == "unsupported":
                    job.status = "failed"
                    job.error = "unsupported_operation"
                    self._reflect(job, "Finderbay AI marked operation unsupported")
                    break
                if result.get("frame", {}).get("needs_clarification"):
                    # Autonomous agent cannot ask user mid-queue — fail with clarification needs
                    job.status = "failed"
                    job.error = "needs_clarification:" + ";".join(
                        result["frame"]["needs_clarification"]
                    )
                    self._reflect(job, "missing slots — cannot auto-clarify without user")
                    break
                if goal_status in ("satisfied", None) and result.get("faithful", True):
                    # None goal can happen for status-like ops
                    if goal_status == "failed":
                        last_err = "goal_failed"
                        job.attempts += 1
                        if job.attempts > job.max_attempts:
                            job.status = "failed"
                            job.error = last_err
                            self._reflect(job, "goal failed after retries")
                            break
                        continue
                    job.status = "succeeded"
                    self._reflect(job, f"completed op={op}")
                    break
                if goal_status == "failed":
                    last_err = "goal_failed"
                    job.attempts += 1
                    if job.attempts > job.max_attempts:
                        job.status = "failed"
                        job.error = last_err
                        break
                    continue
                job.status = "succeeded"
                self._reflect(job, f"completed op={op} status={goal_status}")
                break
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                job.attempts += 1
                if job.attempts > job.max_attempts:
                    job.status = "failed"
                    job.error = last_err
                    self._reflect(job, f"exception after retries: {exc}")
                    break
        job.finished_at = utc_now()

    def _is_coding_job(self, job: AgentJob) -> bool:
        if "coding" in (job.tags or []) or job.metadata.get("domain") == "coding":
            return True
        lower = job.instruction.lower()
        hooks = (
            "write code",
            "generate function",
            "generate class",
            "refactor",
            "rename symbol",
            "run tests",
            "analyze code",
            "ast ",
            "create module",
            "fix code",
            "code search",
            "find symbol",
            "add test",
        )
        return any(h in lower for h in hooks)

    def _is_vision_job(self, job: AgentJob) -> bool:
        if "vision" in (job.tags or []) or job.metadata.get("domain") == "vision":
            return True
        lower = job.instruction.lower()
        hooks = (
            "detect objects",
            "detect vehicles",
            "count people",
            "detect fire",
            "vision status",
            "scan document",
            "ppe compliance",
            "estimate queue",
            "yolo ",
            "computer vision",
        )
        return any(h in lower for h in hooks)

    def _reflect(self, job: AgentJob, note: str) -> None:
        msg = f"[{job.job_id}] {job.status}: {note}"
        self.reflections.append(msg)
        # Feed improvement loop on failures
        if job.status == "failed" and hasattr(self.ai, "improvement"):
            self.ai.improvement.record_failure(
                operation=(job.result or {}).get("frame", {}).get("operation") or "agent_job",
                input_text=job.instruction,
                reason=job.error or note,
                stage="agent",
            )

    def status(self) -> dict[str, Any]:
        return {
            "name": "SelfBrainAgent",
            "queued": len(self.queue),
            "history": len(self.history),
            "running": self.running,
            "reflections": len(self.reflections),
            "coding_attached": self.coding is not None,
            "finderbay_ai": self.ai.status(),
            "llm_at_runtime": False,
            "human_level_agi": False,
            "limits": [
                "Executes declared Finderbay + coding-tool tasks only",
                "Cannot invent unsupported capabilities",
                "Clarification-needed jobs fail closed in autonomous mode",
            ],
        }
