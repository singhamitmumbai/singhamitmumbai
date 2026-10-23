"""CognitiveOS — boot sequence, services, and lifecycle."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..brain import Brain
from ..version import __version__
from .banner import boot_splash, welcome
from .services import ConsoleService, SchedulerService, StoreService


DEFAULT_ROOT = Path.home() / ".cognitive_os"


class CognitiveOS:
    """Self-contained operating environment powered by the symbolic brain."""

    def __init__(
        self,
        root: Optional[Path | str] = None,
        domain: str = "animals",
        name: str = "CognitiveOS-Brain",
        auto_sched: bool = False,
        sched_interval: float = 3.0,
    ) -> None:
        self.root = Path(root) if root else DEFAULT_ROOT
        self.default_domain = domain
        self.auto_sched = auto_sched
        self.sched_interval = sched_interval
        self.version = __version__
        self.booted = False
        self.boot_time: Optional[str] = None

        self.store = StoreService(self.root)
        self.console = ConsoleService(log_path=self.store.log_path)
        self.brain = Brain(name=name, storage=None)
        self.scheduler = SchedulerService(
            think_fn=self._scheduled_think,
            console=self.console,
            interval_sec=sched_interval,
            steps_per_tick=1,
        )
        self._last_session: list[str] = []

    # --- lifecycle ------------------------------------------------------------

    def boot(self, domain: Optional[str] = None, load_persisted: bool = True) -> str:
        """Start brain + services. Idempotent if already booted (re-logs status)."""
        lines: list[str] = []
        self.store.ensure()
        services = ["store", "console"]

        loaded = False
        if load_persisted and self.store.knowledge_path.exists():
            loaded = self.store.load_knowledge(self.brain)
            if loaded:
                self.console.info(
                    f"restored knowledge ({len(self.brain.kb)} facts)",
                    source="store",
                )

        use_domain = domain or self.default_domain
        if not loaded or self.brain.kb.domain in ("empty", "unknown", None):
            self.brain.load_domain(use_domain)
            self.console.info(f"domain loaded: {use_domain}", source="boot")
        else:
            self.console.info(
                f"using persisted domain: {self.brain.kb.domain}",
                source="boot",
            )

        self.brain.mark_booted()
        services.append("brain")

        if self.auto_sched:
            self.scheduler.start()
            services.append("scheduler")
        else:
            self.console.info(
                "scheduler idle (use 'sched start' or think manually)",
                source="boot",
            )

        self.booted = True
        self.boot_time = datetime.now(timezone.utc).isoformat()
        self._persist_state()
        self.console.info("Cognitive OS booted", source="boot")

        lines.append(boot_splash(services))
        lines.append(
            f"[boot] domain={self.brain.kb.domain} facts={len(self.brain.kb)} "
            f"rules={len(self.brain.kb.rules())} root={self.root}"
        )
        return "\n".join(lines)

    def shutdown(self, persist: bool = True) -> str:
        msgs = []
        if self.scheduler.running:
            self.scheduler.stop()
            msgs.append("[shutdown] scheduler stopped")
        if persist:
            path = self.store.save_knowledge(self.brain)
            self.store.dump_mind(self.brain)
            self._persist_state()
            msgs.append(f"[shutdown] knowledge saved → {path}")
        self.booted = False
        self.console.info("Cognitive OS shutdown", source="boot")
        msgs.append("[shutdown] goodbye.")
        return "\n".join(msgs)

    def _persist_state(self) -> None:
        self.store.save_state(
            {
                "version": self.version,
                "booted": self.booted,
                "boot_time": self.boot_time,
                "domain": self.brain.kb.domain,
                "scheduler": self.scheduler.status(),
                "brain": self.brain.status(),
            }
        )

    def _scheduled_think(self, steps: int) -> Any:
        # Only burn cycles when there is work
        if not self.brain.planner.open_goals():
            return type("R", (), {"stopped_reason": "idle (no open goals)"})()
        return self.brain.think(steps=steps)

    # --- queries --------------------------------------------------------------

    def status_text(self) -> str:
        b = self.brain.status()
        s = self.scheduler.status()
        store = self.store.summary()
        lines = [
            "=== Cognitive OS status ===",
            f"  booted:     {self.booted}  since {self.boot_time or '—'}",
            f"  version:    {self.version}",
            f"  root:       {store['root']}",
            f"  knowledge:   {store['knowledge']} (exists={store['knowledge_exists']})",
            f"  brain:      {b['name']}  domain={b['domain']}",
            f"  memory:     facts={b['facts']} rules={b['rules']} beliefs={b['beliefs']}",
            f"  goals:      open={b['open_goals']}  actions={b['actions']}",
            f"  cognition:  cycles={b['think_cycles']} thoughts={b['thoughts']}",
            f"  scheduler:  running={s['running']} paused={s['paused']} "
            f"ticks={s['ticks']} interval={s['interval_sec']}s",
            f"  last tick:  {s['last_result'] or '—'}",
        ]
        return "\n".join(lines)

    def ps_text(self) -> str:
        goals = list(self.brain.planner.goals.values())
        if not goals:
            return "(no goals)"
        lines = ["PID/ID       STATUS      PRI   GOAL"]
        for g in sorted(goals, key=lambda x: -x.priority):
            lines.append(
                f"{g.goal_id:<12} {g.status:<10} {g.priority:<5.1f} {g.description} → {g.target}"
            )
        return "\n".join(lines)

    def mem_text(self, limit: int = 25) -> str:
        facts = self.brain.kb.facts()
        beliefs = self.brain.working.active_beliefs()
        lines = [
            f"=== Memory ({len(facts)} facts, {len(beliefs)} beliefs) ===",
            f"Domain: {self.brain.kb.domain}",
            "— Facts —",
        ]
        for t in facts[:limit]:
            lines.append(f"  {t}")
        if len(facts) > limit:
            lines.append(f"  ... +{len(facts) - limit} more")
        lines.append("— Beliefs —")
        if not beliefs:
            lines.append("  (none)")
        for b in beliefs[:limit]:
            lines.append(f"  {b.content} conf={b.confidence:.2f} src={b.source}")
        lines.append("— Open questions —")
        qs = self.brain.working.open_questions()
        if not qs:
            lines.append("  (none)")
        for q in qs:
            lines.append(f"  [{q.priority:.1f}] {q.text}")
        return "\n".join(lines)

    def dmesg_text(self, n: int = 20) -> str:
        entries = self.console.recent(n)
        if not entries:
            return "(console empty)"
        return "\n".join(e.format() for e in entries)

    def welcome(self) -> str:
        return welcome(self.version, str(self.root))

    def ensure_booted(self) -> Optional[str]:
        if self.booted:
            return None
        return self.boot()
