"""Core Cognitive OS services: store, scheduler, console."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LogEntry:
    level: str
    message: str
    timestamp: str = field(default_factory=_now)
    source: str = "os"

    def format(self) -> str:
        return f"{self.timestamp} [{self.level}] {self.source}: {self.message}"

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "source": self.source,
            "message": self.message,
        }


class ConsoleService:
    """In-memory + optional file-backed message/log console."""

    def __init__(self, log_path: Optional[Path] = None, capacity: int = 2000) -> None:
        self.log_path = log_path
        self.capacity = capacity
        self._entries: deque[LogEntry] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def log(self, message: str, level: str = "INFO", source: str = "os") -> LogEntry:
        entry = LogEntry(level=level.upper(), message=message, source=source)
        with self._lock:
            self._entries.append(entry)
            if self.log_path is not None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(entry.format() + "\n")
        return entry

    def info(self, message: str, source: str = "os") -> LogEntry:
        return self.log(message, "INFO", source)

    def warn(self, message: str, source: str = "os") -> LogEntry:
        return self.log(message, "WARN", source)

    def error(self, message: str, source: str = "os") -> LogEntry:
        return self.log(message, "ERROR", source)

    def recent(self, n: int = 30) -> list[LogEntry]:
        with self._lock:
            items = list(self._entries)
        return items[-n:]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def dump(self, n: int = 50) -> list[dict]:
        return [e.to_dict() for e in self.recent(n)]


class StoreService:
    """Filesystem-backed knowledge + OS state persistence."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.knowledge_path = self.root / "knowledge.json"
        self.state_path = self.root / "os_state.json"
        self.log_path = self.root / "console.log"
        self.mind_dump_path = self.root / "mind_dump.json"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "tmp").mkdir(exist_ok=True)

    def save_knowledge(self, brain: Any) -> Path:
        self.ensure()
        return brain.save(self.knowledge_path)

    def load_knowledge(self, brain: Any) -> bool:
        if not self.knowledge_path.exists():
            return False
        brain.load(self.knowledge_path)
        return True

    def save_state(self, state: dict) -> Path:
        self.ensure()
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return self.state_path

    def load_state(self) -> dict:
        if not self.state_path.exists():
            return {}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def dump_mind(self, brain: Any) -> Path:
        self.ensure()
        self.mind_dump_path.write_text(brain.dump_mind_json(), encoding="utf-8")
        return self.mind_dump_path

    def summary(self) -> dict:
        return {
            "root": str(self.root),
            "knowledge": str(self.knowledge_path),
            "knowledge_exists": self.knowledge_path.exists(),
            "state": str(self.state_path),
            "log": str(self.log_path),
        }


class SchedulerService:
    """Background cognition: think-loop as a scheduled process."""

    def __init__(
        self,
        think_fn: Callable[[int], Any],
        console: ConsoleService,
        interval_sec: float = 2.0,
        steps_per_tick: int = 1,
    ) -> None:
        self.think_fn = think_fn
        self.console = console
        self.interval_sec = interval_sec
        self.steps_per_tick = steps_per_tick
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._pause.set()  # not paused = set means running allowed... use clearer flags
        self.running = False
        self.paused = False
        self.ticks = 0
        self.last_tick_at: Optional[str] = None
        self.last_result: Optional[str] = None
        self._should_run = threading.Event()  # when set, background cognition enabled

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._should_run.set()
        self.paused = False
        self.running = True
        self._thread = threading.Thread(target=self._loop, name="cognition-sched", daemon=True)
        self._thread.start()
        self.console.info("scheduler started", source="sched")

    def stop(self) -> None:
        self._stop.set()
        self._should_run.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.running = False
        self.console.info("scheduler stopped", source="sched")

    def pause(self) -> None:
        self.paused = True
        self.console.info("scheduler paused", source="sched")

    def resume(self) -> None:
        self.paused = False
        self.console.info("scheduler resumed", source="sched")

    def tick_once(self, steps: Optional[int] = None) -> Any:
        n = steps if steps is not None else self.steps_per_tick
        result = self.think_fn(n)
        self.ticks += 1
        self.last_tick_at = _now()
        reason = getattr(result, "stopped_reason", str(result))
        self.last_result = reason
        self.console.info(f"tick#{self.ticks} steps={n} → {reason}", source="sched")
        return result

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._should_run.is_set() and not self.paused:
                try:
                    self.tick_once()
                except Exception as exc:  # noqa: BLE001 — keep OS alive
                    self.console.error(f"tick failed: {exc}", source="sched")
            self._stop.wait(self.interval_sec)

    def status(self) -> dict:
        return {
            "running": self.running,
            "paused": self.paused,
            "interval_sec": self.interval_sec,
            "steps_per_tick": self.steps_per_tick,
            "ticks": self.ticks,
            "last_tick_at": self.last_tick_at,
            "last_result": self.last_result,
        }
