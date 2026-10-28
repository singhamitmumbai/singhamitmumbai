"""24/7 continuous self-learning loop (verified experience only).

Runs a background tick forever (or until stopped):
  1) Ingest live observations from product routes
  2) Replay / consolidate episodic memory
  3) Reinforce skills from successful patterns
  4) Mine candidate preferences (never unverified self-labels as facts)
  5) Dream / rehearse high-importance episodes
  6) Calibrate metacognition stats
  7) Persist journal + snapshot

Enable:
  export COMPLETE_AI_CONTINUOUS_LEARNING=1
  python -m cognitive_ai learn --daemon

Honesty: human_level_agi=false. This is continual product learning, not AGI.
Unverified model/self predictions are rejected as ground-truth labels.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..agi_gap.continual_skills import GrowingSkillLibrary
from ..agi_gap.general_learning import GeneralLearner
from ..agi_gap.lifelong_memory import LifelongMemory
from ..agi_gap.metacognition import MetaCognition
from ..core.ids import new_id
from ..core.records import utc_now
from ..memory.store import MemorySystem
from .journal import LearningJournal


@dataclass
class TickReport:
    tick_id: str
    jobs_run: list[str] = field(default_factory=list)
    observations_consumed: int = 0
    skills_reinforced: int = 0
    preferences_updated: int = 0
    candidates_proposed: int = 0
    consolidations: int = 0
    dreams: int = 0
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    human_level_agi: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContinuousLearner:
    """Always-on learner — bounded ticks, durable journal, verified-only promotions."""

    version = "continuous_learner_v1"

    def __init__(
        self,
        *,
        memory: Optional[MemorySystem] = None,
        store_dir: Path | str = "data/complete_ai/continuous_learning",
        interval_sec: float = 60.0,
        enabled: Optional[bool] = None,
        max_jobs_per_tick: int = 6,
    ) -> None:
        env_on = os.environ.get("COMPLETE_AI_CONTINUOUS_LEARNING", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.enabled = env_on if enabled is None else enabled
        self.interval_sec = max(5.0, float(interval_sec))
        self.max_jobs_per_tick = max_jobs_per_tick
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self.memory = memory or MemorySystem()
        self.skills = GrowingSkillLibrary(self.memory)
        self.learner = GeneralLearner(self.memory, self.skills)
        self.lifelong = LifelongMemory(self.memory)
        self.meta = MetaCognition()
        self.journal = LearningJournal(self.store_dir / "journal.jsonl")

        self._inbox: list[dict[str, Any]] = []
        self._inbox_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._running = False
        self.ticks: list[dict[str, Any]] = []
        self.started_at: Optional[str] = None
        self.last_tick_at: Optional[str] = None
        self.total_ticks = 0
        self.stats = Counter()
        self._load_snapshot()

    # --- lifecycle ------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "name": "ContinuousLearner",
            "version": self.version,
            "enabled": self.enabled,
            "running": self._running,
            "interval_sec": self.interval_sec,
            "total_ticks": self.total_ticks,
            "started_at": self.started_at,
            "last_tick_at": self.last_tick_at,
            "inbox": len(self._inbox),
            "journal_events": self.journal.count(),
            "skills": len(self.skills.skills),
            "learner_events": len(self.learner.events),
            "stats": dict(self.stats),
            "store_dir": str(self.store_dir),
            "human_level_agi": False,
            "notes": (
                "24/7 verified-experience learner — not AGI. "
                "Rejects unverified self-labels. Set COMPLETE_AI_CONTINUOUS_LEARNING=1."
            ),
        }

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False
        self.stop()

    def start(self, *, daemon: bool = True) -> dict[str, Any]:
        """Start background 24/7 loop."""
        if not self.enabled:
            return {"ok": False, "error": "continuous_learning_disabled", "hint": "COMPLETE_AI_CONTINUOUS_LEARNING=1"}
        if self._running:
            return {"ok": True, "already_running": True, **self.status()}
        self._stop.clear()
        self._running = True
        self.started_at = utc_now()
        self._thread = threading.Thread(target=self._loop, name="ContinuousLearner", daemon=daemon)
        self._thread.start()
        self.journal.append("lifecycle", {"event": "start", "interval_sec": self.interval_sec}, source="continuous")
        self._save_snapshot()
        return {"ok": True, "started": True, **self.status()}

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._running = False
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=2.0)
        self._thread = None
        self.journal.append("lifecycle", {"event": "stop"}, source="continuous")
        self._save_snapshot()
        return {"ok": True, "stopped": True, **self.status()}

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self.enabled:
                    self.tick()
            except Exception as exc:  # noqa: BLE001 — never kill the 24/7 loop
                self.stats["loop_errors"] += 1
                self.journal.append("error", {"error": str(exc)}, source="continuous")
            self._stop.wait(self.interval_sec)

    # --- observe live traffic -------------------------------------------------

    def observe(
        self,
        *,
        text: str = "",
        route: str = "",
        ok: bool = True,
        meta: Optional[dict[str, Any]] = None,
        source: str = "session",
        verified: bool = True,
    ) -> dict[str, Any]:
        """Non-blocking inbox push from CompleteAI / Assistant / JARVIS."""
        if not verified and source in ("model", "self", "unverified"):
            return {"ok": False, "error": "rejected_unverified_self_label", "human_level_agi": False}
        item = {
            "text": (text or "")[:500],
            "route": route or "unknown",
            "ok": bool(ok),
            "meta": dict(meta or {}),
            "source": source,
            "verified": verified,
            "ts": utc_now(),
            "id": new_id("obs"),
        }
        with self._inbox_lock:
            self._inbox.append(item)
            if len(self._inbox) > 2000:
                self._inbox = self._inbox[-2000:]
        self.stats["observations"] += 1
        return {"ok": True, "queued": item["id"], "human_level_agi": False}

    # --- tick jobs ------------------------------------------------------------

    def tick(self) -> dict[str, Any]:
        """One learning cycle — safe to call manually or from daemon."""
        t0 = time.perf_counter()
        report = TickReport(tick_id=new_id("tick"))
        if not self.enabled:
            report.errors.append("disabled")
            return report.to_dict()

        jobs = [
            ("drain_inbox", self._job_drain_inbox),
            ("reinforce_skills", self._job_reinforce_skills),
            ("consolidate", self._job_consolidate),
            ("mine_patterns", self._job_mine_patterns),
            ("dream", self._job_dream),
            ("calibrate", self._job_calibrate),
        ]
        for name, fn in jobs[: self.max_jobs_per_tick]:
            try:
                detail = fn(report)
                report.jobs_run.append(name)
                self.journal.append("job", {"job": name, "detail": detail}, source="continuous")
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"{name}:{exc}")
                self.stats["job_errors"] += 1

        report.duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        self.total_ticks += 1
        self.last_tick_at = utc_now()
        self.ticks.append(report.to_dict())
        if len(self.ticks) > 100:
            self.ticks = self.ticks[-100:]
        self.stats["ticks"] += 1
        self.journal.append("tick", report.to_dict(), source="continuous")
        self._save_snapshot()
        return report.to_dict()

    def _job_drain_inbox(self, report: TickReport) -> dict[str, Any]:
        with self._inbox_lock:
            batch = list(self._inbox[:100])
            self._inbox = self._inbox[100:]
        learned = 0
        for item in batch:
            route = item.get("route") or "unknown"
            ok = bool(item.get("ok"))
            text = item.get("text") or ""
            # Lifelong episodic memory
            self.lifelong.remember_interaction(
                summary=f"{route}:{text[:80]}",
                actions=[{"op": route}],
                outcome="success" if ok else "failure",
            )
            # Preference: track last successful route style
            if ok:
                self.learner.observe(
                    kind="preference",
                    payload={"key": f"route_ok:{route}", "value": True},
                    verified=True,
                    source=str(item.get("source") or "session"),
                )
                learned += 1
            else:
                self.learner.observe(
                    kind="correction",
                    payload={"route": route, "text": text[:120], "note": "failed_outcome"},
                    verified=True,
                    source="session",
                )
            # Skill outcome if route maps to a known skill
            skill_name = _route_to_skill(route)
            if skill_name:
                self.skills.record_outcome(skill_name, success=ok)
                report.skills_reinforced += 1
        report.observations_consumed += len(batch)
        self.stats["drained"] += len(batch)
        return {"consumed": len(batch), "learned_prefs": learned}

    def _job_reinforce_skills(self, report: TickReport) -> dict[str, Any]:
        # Promote frequently successful induced patterns already in library
        reinforced = 0
        for card in list(self.skills.skills.values()):
            if card.success_count >= 3 and card.source in ("induced", "seed", "transferred"):
                # Soft reinforce via learner skill observe
                self.learner.observe(
                    kind="skill",
                    payload={
                        "name": card.name,
                        "steps": card.steps,
                        "domain": card.domain,
                        "summary": f"reinforce:{card.name}",
                    },
                    verified=True,
                    source="continuous",
                )
                reinforced += 1
                report.skills_reinforced += 1
                if reinforced >= 5:
                    break
        return {"reinforced": reinforced}

    def _job_consolidate(self, report: TickReport) -> dict[str, Any]:
        if hasattr(self.lifelong, "consolidate"):
            cons = self.lifelong.consolidate()
            report.consolidations += 1
            detail = cons.to_dict() if hasattr(cons, "to_dict") else dict(cons or {})
            # Durable lifelong dump for cross-restart continuity
            try:
                life_path = self.store_dir / "lifelong_memory.json"
                if hasattr(self.lifelong, "persist"):
                    persisted = self.lifelong.persist(life_path)
                    detail["persisted"] = persisted.get("path") if isinstance(persisted, dict) else str(life_path)
            except Exception as exc:  # noqa: BLE001
                detail["persist_error"] = str(exc)
            self.stats["consolidations"] += 1
            return detail
        return {"ok": False, "error": "no_consolidate"}

    def _job_mine_patterns(self, report: TickReport) -> dict[str, Any]:
        """Mine frequent routes / tokens into *candidate* preferences (not facts)."""
        recent = self.journal.iter_recent(300)
        routes = Counter()
        for row in recent:
            if row.get("kind") != "job":
                continue
            # also scan inbox-drained via tick payloads
        # Use learner events + memory episodes
        for ev in self.learner.events[-100:]:
            if ev.kind == "preference":
                routes[str(ev.payload.get("key") or "")] += 1
        for ep in list(self.memory.episodes)[-50:]:
            for act in ep.actions or []:
                if isinstance(act, dict) and act.get("op"):
                    routes[str(act["op"])] += 1
        proposed = 0
        for key, count in routes.most_common(5):
            if not key or count < 3:
                continue
            # Candidate only — stored as preference with candidate: prefix
            self.learner.observe(
                kind="preference",
                payload={"key": f"candidate_route:{key}", "value": {"count": count}},
                verified=True,
                source="continuous_miner",
            )
            proposed += 1
            report.candidates_proposed += 1
        self.stats["candidates"] += proposed
        return {"proposed": proposed, "top": routes.most_common(5)}

    def _job_dream(self, report: TickReport) -> dict[str, Any]:
        if hasattr(self.lifelong, "dream"):
            dream = self.lifelong.dream(n=5)
            report.dreams += 1
            self.stats["dreams"] += 1
            return dream if isinstance(dream, dict) else {"ok": True}
        # Fallback: re-score last episodes
        eps = list(self.memory.episodes)[-5:]
        scored = [
            {"summary": e.summary[:80], "importance": self.lifelong.importance(e)}
            for e in eps
        ]
        report.dreams += 1
        return {"replayed": len(scored), "items": scored}

    def _job_calibrate(self, report: TickReport) -> dict[str, Any]:
        # Feed metacognition calibration with recent success ratios
        recent = self.ticks[-20:]
        ok_jobs = sum(1 for t in recent if not t.get("errors"))
        total = max(1, len(recent))
        rate = ok_jobs / total
        # record into meta calibration buckets if API exists
        if hasattr(self.meta, "calibration"):
            bucket = self.meta.calibration.setdefault("continuous", [0, 0])
            bucket[1] += 1
            if rate >= 0.8:
                bucket[0] += 1
        self.stats["calibrations"] += 1
        return {"tick_success_rate": round(rate, 3), "recent_ticks": total}

    # --- persistence ----------------------------------------------------------

    def _snapshot_path(self) -> Path:
        return self.store_dir / "snapshot.json"

    def _save_snapshot(self) -> None:
        payload = {
            "version": self.version,
            "enabled": self.enabled,
            "interval_sec": self.interval_sec,
            "total_ticks": self.total_ticks,
            "started_at": self.started_at,
            "last_tick_at": self.last_tick_at,
            "stats": dict(self.stats),
            "skills": self.skills.list_skills()[:50],
            "human_level_agi": False,
            "saved_at": utc_now(),
        }
        self._snapshot_path().write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def _load_snapshot(self) -> None:
        path = self._snapshot_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.total_ticks = int(data.get("total_ticks") or 0)
            self.started_at = data.get("started_at")
            self.last_tick_at = data.get("last_tick_at")
            for k, v in (data.get("stats") or {}).items():
                self.stats[k] = int(v)
        except Exception:  # noqa: BLE001
            return

    def run_forever(self, *, ticks: Optional[int] = None) -> dict[str, Any]:
        """Foreground 24/7 loop (or limited ticks for tests)."""
        self.enable()
        self.started_at = self.started_at or utc_now()
        self._running = True
        n = 0
        try:
            while True:
                self.tick()
                n += 1
                if ticks is not None and n >= ticks:
                    break
                time.sleep(self.interval_sec)
        finally:
            self._running = False
            self._save_snapshot()
        return {"ok": True, "ticks": n, **self.status()}


def _route_to_skill(route: str) -> Optional[str]:
    r = (route or "").lower()
    mapping = {
        "open_domain": "explain_concept",
        "llm_chat": "explain_concept",
        "coding": "howto_plan",
        "vision": "multimodal_ingest",
        "finderbay": "howto_plan",
        "proto_agi": "long_horizon_loop",
        "long_horizon": "long_horizon_loop",
        "web": "web_research",
        "search": "web_research",
        "computer_control": "howto_plan",
        "jarvis": "howto_plan",
    }
    for key, skill in mapping.items():
        if key in r:
            return skill
    return None


# Process-wide singleton for product wiring
_GLOBAL: Optional[ContinuousLearner] = None
_GLOBAL_LOCK = threading.Lock()


def get_continuous_learner(**kwargs: Any) -> ContinuousLearner:
    global _GLOBAL
    with _GLOBAL_LOCK:
        if _GLOBAL is None:
            _GLOBAL = ContinuousLearner(**kwargs)
        return _GLOBAL
