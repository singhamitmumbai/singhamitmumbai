"""Background-style scheduler that ticks the SelfBrain queue (stdlib threads)."""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

from .self_brain import SelfBrainAgent, AgentReport


class AgentScheduler:
    """Periodically drain the agent queue — non-LLM automation daemon."""

    def __init__(
        self,
        agent: SelfBrainAgent,
        *,
        interval_sec: float = 2.0,
        jobs_per_tick: int = 3,
    ) -> None:
        self.agent = agent
        self.interval_sec = interval_sec
        self.jobs_per_tick = jobs_per_tick
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.last_report: Optional[AgentReport] = None
        self.ticks = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="SelfBrainScheduler", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def tick_once(self) -> Optional[AgentReport]:
        if not self.agent.queue:
            return None
        report = self.agent.run(max_jobs=self.jobs_per_tick)
        self.last_report = report
        self.ticks += 1
        return report

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.tick_once()
            self._stop.wait(self.interval_sec)

    def status(self) -> dict[str, Any]:
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "ticks": self.ticks,
            "interval_sec": self.interval_sec,
            "jobs_per_tick": self.jobs_per_tick,
            "queue_depth": len(self.agent.queue),
            "last_report": self.last_report.to_dict() if self.last_report else None,
        }
