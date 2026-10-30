"""Explicit operating cycle with inspectable stage traces (blueprint §2, §20).

Stages share a typed WorkingState so perception, NLU, memory, reasoning,
planning, execution and verification speak a common language of records.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from ..core.records import utc_now
from ..core.ids import new_id


STAGES = (
    "perceive",
    "understand",
    "attend",
    "goal",
    "reason",
    "plan",
    "execute",
    "verify",
    "respond",
    "learn_hook",
)


@dataclass
class StageResult:
    name: str
    ok: bool = True
    summary: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    uncertainties: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CycleTrace:
    """Full inspectable trace of one request through the operating process."""

    cycle_id: str = field(default_factory=lambda: new_id("cycle"))
    started_at: str = field(default_factory=utc_now)
    finished_at: Optional[str] = None
    stages: list[StageResult] = field(default_factory=list)
    distinctions: dict[str, str] = field(
        default_factory=lambda: {
            "unknown_vs_false": "Unknown is different from false",
            "predicted_vs_observed": "Predicted is different from observed",
            "attempted_vs_confirmed": "Attempted is different from confirmed",
        }
    )
    outcome: str = "running"  # running|satisfied|failed|abstained|missing_info|unsupported
    llm_at_runtime: bool = False

    def add(self, stage: StageResult) -> None:
        self.stages.append(stage)

    def finish(self, outcome: str) -> None:
        self.outcome = outcome
        self.finished_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "outcome": self.outcome,
            "llm_at_runtime": self.llm_at_runtime,
            "distinctions": dict(self.distinctions),
            "stages": [s.to_dict() for s in self.stages],
            "stage_names": [s.name for s in self.stages],
        }


class OperatingCycle:
    """Helper to build a CycleTrace consistently across handle()."""

    def __init__(self) -> None:
        self.trace = CycleTrace()

    def stage(
        self,
        name: str,
        *,
        ok: bool = True,
        summary: str = "",
        data: Optional[dict] = None,
        duration_ms: float = 0.0,
        uncertainties: Optional[list[str]] = None,
    ) -> StageResult:
        sr = StageResult(
            name=name,
            ok=ok,
            summary=summary,
            data=data or {},
            duration_ms=duration_ms,
            uncertainties=list(uncertainties or []),
        )
        self.trace.add(sr)
        return sr

    def finish(self, outcome: str) -> CycleTrace:
        self.trace.finish(outcome)
        return self.trace
