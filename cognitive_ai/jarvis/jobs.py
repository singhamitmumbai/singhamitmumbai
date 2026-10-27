"""Durable job store with JARVIS execution states (blueprint §11)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from ..core.ids import new_id
from ..core.records import utc_now
from .contracts import JobState


@dataclass
class JobRecord:
    request_text: str
    intent: str = ""
    state: str = JobState.READY.value
    steps: list[dict[str, Any]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    job_id: str = field(default_factory=lambda: new_id("job"))
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    client_attempt_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobStore:
    def __init__(self) -> None:
        self.jobs: dict[str, JobRecord] = {}
        self._by_attempt: dict[str, str] = {}

    def create(self, request_text: str, *, intent: str = "", attempt_id: Optional[str] = None) -> JobRecord:
        if attempt_id and attempt_id in self._by_attempt:
            return self.jobs[self._by_attempt[attempt_id]]
        job = JobRecord(request_text=request_text, intent=intent, client_attempt_id=attempt_id)
        self.jobs[job.job_id] = job
        if attempt_id:
            self._by_attempt[attempt_id] = job.job_id
        return job

    def transition(self, job_id: str, state: JobState | str, *, error: Optional[str] = None) -> JobRecord:
        job = self.jobs[job_id]
        job.state = state.value if isinstance(state, JobState) else str(state)
        job.updated_at = utc_now()
        if error:
            job.error = error
        return job

    def add_step(self, job_id: str, step: dict[str, Any]) -> None:
        self.jobs[job_id].steps.append(step)
        self.jobs[job_id].updated_at = utc_now()

    def add_result(self, job_id: str, result: dict[str, Any]) -> None:
        self.jobs[job_id].results.append(result)
        self.jobs[job_id].updated_at = utc_now()

    def request_cancel(self, job_id: str) -> JobRecord:
        job = self.jobs[job_id]
        if job.state in (JobState.COMPLETED.value, JobState.CANCELLED.value, JobState.FAILED.value):
            return job
        return self.transition(job_id, JobState.CANCEL_REQUESTED)

    def get(self, job_id: str) -> Optional[JobRecord]:
        return self.jobs.get(job_id)

    def summary(self) -> dict[str, Any]:
        by_state: dict[str, int] = {}
        for j in self.jobs.values():
            by_state[j.state] = by_state.get(j.state, 0) + 1
        return {"total": len(self.jobs), "by_state": by_state}
