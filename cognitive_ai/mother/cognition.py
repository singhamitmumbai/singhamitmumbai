"""Continuous cognition scheduler with hard budgets for Mother AI.

Runs bounded autonomous cycles (not unbounded 24/7 AGI). Prefer interactive
handle() for user turns; use ContinuousCognition for optional background ticks.
human_level_agi=false.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..core.records import utc_now


@dataclass
class CognitionBudget:
    """Hard caps — cognition stops when any budget is exhausted."""

    max_cycles: int = 20
    max_seconds: float = 30.0
    max_tool_calls: int = 10
    max_attention_spend: float = 2.0
    tick_sleep_s: float = 0.05
    human_level_agi: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CognitionTick:
    tick_id: int
    prompt: str
    ok: bool
    text_preview: str = ""
    action: str = ""
    prediction_error: float = 0.0
    duration_ms: float = 0.0
    ts: str = field(default_factory=utc_now)
    stop_reason: Optional[str] = None
    human_level_agi: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContinuousCognition:
    """Budgeted autonomous cycle runner over MotherBrain.handle."""

    version = "mother_cognition_v1"

    DEFAULT_PROMPTS = (
        "what open goals remain?",
        "summarize recent prediction errors",
        "inspect learning status lightly",
        "refresh beliefs about last intent",
    )

    def __init__(
        self,
        mother: Any,
        *,
        budget: Optional[CognitionBudget] = None,
        store_path: Path | str | None = None,
    ) -> None:
        self.mother = mother
        self.budget = budget or CognitionBudget()
        self.store_path = Path(store_path) if store_path else None
        self.ticks: list[CognitionTick] = []
        self.cycles_run = 0
        self.tool_calls = 0
        self.attention_spent = 0.0
        self.started_at: Optional[str] = None
        self.stopped_at: Optional[str] = None
        self.stop_reason: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def remaining(self) -> dict[str, Any]:
        elapsed = 0.0
        if self.started_at and not self.stopped_at:
            # approximate via tick durations if wall clock unavailable in tick metadata
            elapsed = sum(t.duration_ms for t in self.ticks) / 1000.0
        return {
            "cycles_left": max(0, self.budget.max_cycles - self.cycles_run),
            "tool_calls_left": max(0, self.budget.max_tool_calls - self.tool_calls),
            "attention_left": round(max(0.0, self.budget.max_attention_spend - self.attention_spent), 3),
            "elapsed_s_approx": round(elapsed, 3),
            "max_seconds": self.budget.max_seconds,
            "stop_reason": self.stop_reason,
            "human_level_agi": False,
        }

    def _budget_exhausted(self, wall_t0: float) -> Optional[str]:
        if self._stop.is_set():
            return "stopped"
        if self.cycles_run >= self.budget.max_cycles:
            return "max_cycles"
        if (time.perf_counter() - wall_t0) >= self.budget.max_seconds:
            return "max_seconds"
        if self.tool_calls >= self.budget.max_tool_calls:
            return "max_tool_calls"
        if self.attention_spent >= self.budget.max_attention_spend:
            return "max_attention_spend"
        return None

    def run_once(self, prompt: Optional[str] = None) -> CognitionTick:
        """Single budgeted cognitive tick."""
        with self._lock:
            idx = self.cycles_run + 1
            text = prompt or self.DEFAULT_PROMPTS[(idx - 1) % len(self.DEFAULT_PROMPTS)]
            t0 = time.perf_counter()
            try:
                out = self.mother.handle(text)
                ok = bool(out.get("ok", True))
                action = str((out.get("mother_cycle") or {}).get("action") or "")
                err = float((out.get("mother_cycle") or {}).get("prediction_error") or 0.0)
                preview = str(out.get("text") or "")[:160]
                # Count non-dialogue as a tool call
                if action and action not in ("dialogue", "respond"):
                    self.tool_calls += 1
                # Attention spend heuristic from world model if present
                spend = 0.05 if action == "dialogue" else 0.2
                wm = getattr(self.mother, "world_model", None)
                if wm is not None and hasattr(wm, "summary"):
                    # delta from previous attention if available
                    spend = max(spend, 0.05)
                self.attention_spent += spend
            except Exception as exc:  # noqa: BLE001
                ok = False
                action = "error"
                err = 1.0
                preview = str(exc)[:160]
            self.cycles_run += 1
            tick = CognitionTick(
                tick_id=idx,
                prompt=text,
                ok=ok,
                text_preview=preview,
                action=action,
                prediction_error=err,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
            self.ticks.append(tick)
            if len(self.ticks) > 200:
                self.ticks = self.ticks[-200:]
            self._persist()
            return tick

    def run(self, *, prompts: Optional[list[str]] = None) -> dict[str, Any]:
        """Run until a hard budget is hit. Synchronous."""
        self.started_at = utc_now()
        self.stop_reason = None
        wall_t0 = time.perf_counter()
        prompt_list = list(prompts) if prompts else None
        i = 0
        while True:
            reason = self._budget_exhausted(wall_t0)
            if reason:
                self.stop_reason = reason
                break
            prompt = None
            if prompt_list is not None:
                if i >= len(prompt_list):
                    self.stop_reason = "prompts_exhausted"
                    break
                prompt = prompt_list[i]
                i += 1
            tick = self.run_once(prompt)
            if self.budget.tick_sleep_s > 0:
                time.sleep(self.budget.tick_sleep_s)
            # Early stop on repeated hard failures
            if not tick.ok and self.cycles_run >= 3:
                recent = self.ticks[-3:]
                if all(not t.ok for t in recent):
                    self.stop_reason = "repeated_failure"
                    tick.stop_reason = self.stop_reason
                    break
        self.stopped_at = utc_now()
        return self.summary()

    def start_background(self, *, prompts: Optional[list[str]] = None) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def _loop() -> None:
            self.run(prompts=prompts)

        self._thread = threading.Thread(target=_loop, name="MotherContinuousCognition", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def summary(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "cycles_run": self.cycles_run,
            "tool_calls": self.tool_calls,
            "attention_spent": round(self.attention_spent, 3),
            "budget": self.budget.to_dict(),
            "remaining": self.remaining(),
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "stop_reason": self.stop_reason,
            "recent_ticks": [t.to_dict() for t in self.ticks[-10:]],
            "human_level_agi": False,
            "note": "Budgeted autonomous ticks — not unbounded self-improvement or AGI.",
        }

    def _persist(self) -> None:
        if not self.store_path:
            return
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            self.store_path.write_text(json.dumps(self.summary(), indent=2, default=str), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
