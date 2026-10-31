"""Self-modification lab — isolated candidates + rollback (scaffold).

Never auto-applies code changes to the live product. Candidates live under an
isolated directory; promotion requires an explicit human reviewer string.
human_level_agi=false.
"""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..core.ids import new_id
from ..core.records import utc_now


FORBIDDEN_REVIEWERS = frozenset({"model", "self", "unverified", "auto", "mother", ""})


@dataclass
class ModCandidate:
    candidate_id: str
    kind: str  # belief_rule|cue_map|prompt|procedure|config
    title: str
    payload: dict[str, Any]
    status: str = "isolated"  # isolated|evaluated|accepted|rejected|rolled_back|promoted
    evidence: list[str] = field(default_factory=list)
    eval_score: Optional[float] = None
    eval_notes: str = ""
    snapshot_before: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    reviewer: Optional[str] = None
    human_level_agi: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SelfModLab:
    """Sandbox for Mother-side candidate improvements with rollback."""

    version = "mother_self_mod_v1"

    def __init__(self, lab_dir: Path | str = "data/complete_ai/mother/self_mod_lab") -> None:
        self.lab_dir = Path(lab_dir)
        self.lab_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_dir = self.lab_dir / "candidates"
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir = self.lab_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.lab_dir / "index.json"
        self.candidates: list[ModCandidate] = []
        self._load()

    def propose(
        self,
        *,
        kind: str,
        title: str,
        payload: dict[str, Any],
        evidence: Optional[list[str]] = None,
        snapshot_before: Optional[dict[str, Any]] = None,
    ) -> ModCandidate:
        """Create an isolated candidate — does not mutate live Mother state."""
        cid = new_id("mod")
        cand = ModCandidate(
            candidate_id=cid,
            kind=kind,
            title=title[:200],
            payload=dict(payload or {}),
            evidence=list(evidence or []),
            snapshot_before=deepcopy(snapshot_before or {}),
        )
        # Write isolated artifact
        cdir = self.candidates_dir / cid
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "candidate.json").write_text(json.dumps(cand.to_dict(), indent=2, default=str), encoding="utf-8")
        if cand.snapshot_before:
            (cdir / "snapshot_before.json").write_text(
                json.dumps(cand.snapshot_before, indent=2, default=str), encoding="utf-8"
            )
        self.candidates.append(cand)
        self._save()
        return cand

    def evaluate(
        self,
        candidate_id: str,
        *,
        score_fn: Optional[Any] = None,
        notes: str = "",
    ) -> ModCandidate:
        """Score a candidate in isolation (default heuristic)."""
        cand = self._get(candidate_id)
        if cand.status not in ("isolated", "evaluated", "rejected"):
            raise ValueError(f"Cannot evaluate candidate in status={cand.status}")
        score = 0.5
        if score_fn is not None:
            score = float(score_fn(cand))
        else:
            # Heuristic: prefer cue_map/belief_rule with evidence
            score = 0.4
            if cand.evidence:
                score += 0.15
            if cand.kind in ("belief_rule", "cue_map", "config"):
                score += 0.1
            if cand.payload.get("expected_cues") or cand.payload.get("rule"):
                score += 0.1
            if cand.payload.get("dangerous") or cand.payload.get("shell"):
                score -= 0.5
            score = max(0.0, min(1.0, score))
        cand.eval_score = round(score, 3)
        cand.eval_notes = notes or "heuristic_isolated_eval"
        cand.status = "evaluated"
        cand.updated_at = utc_now()
        self._write_cand(cand)
        self._save()
        return cand

    def review(self, candidate_id: str, *, accept: bool, reviewer: str) -> ModCandidate:
        """Human review gate — model/self/auto reviewers forbidden."""
        rev = (reviewer or "").strip()
        if rev.lower() in FORBIDDEN_REVIEWERS:
            raise ValueError(
                "Unverified/model/self reviewers cannot accept or reject self-mod candidates. "
                "Provide an explicit human reviewer id."
            )
        cand = self._get(candidate_id)
        if cand.status not in ("evaluated", "isolated", "rejected", "accepted"):
            raise ValueError(f"Cannot review candidate in status={cand.status}")
        cand.reviewer = rev
        cand.status = "accepted" if accept else "rejected"
        cand.updated_at = utc_now()
        self._write_cand(cand)
        self._save()
        return cand

    def apply_isolated(
        self,
        candidate_id: str,
        live_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply candidate payload onto a *copy* of live_state (sandbox only).

        Returns the mutated copy — never writes back to MotherBrain automatically.
        """
        cand = self._get(candidate_id)
        if cand.status not in ("accepted", "evaluated", "isolated"):
            raise ValueError("Only isolated/evaluated/accepted candidates can be sandbox-applied")
        sandbox = deepcopy(live_state)
        kind = cand.kind
        payload = cand.payload
        if kind == "cue_map":
            cues = sandbox.setdefault("expected_cues", {})
            for action, vals in (payload.get("expected_cues") or {}).items():
                cues[action] = list(vals)
        elif kind == "belief_rule":
            beliefs = sandbox.setdefault("beliefs", [])
            beliefs.append(dict(payload.get("belief") or payload))
        elif kind == "config":
            cfg = sandbox.setdefault("config", {})
            cfg.update(dict(payload.get("config") or payload))
        elif kind == "prompt":
            sandbox["prompt_overlay"] = str(payload.get("prompt") or "")
        else:
            sandbox.setdefault("procedures", []).append(dict(payload))
        # Persist sandbox result under candidate dir
        cdir = self.candidates_dir / candidate_id
        (cdir / "sandbox_after.json").write_text(json.dumps(sandbox, indent=2, default=str), encoding="utf-8")
        return {"ok": True, "candidate_id": candidate_id, "sandbox": sandbox, "live_mutated": False, "human_level_agi": False}

    def promote(self, candidate_id: str, *, reviewer: str) -> dict[str, Any]:
        """Mark promoted after human acceptance — still does not patch source code."""
        cand = self.review(candidate_id, accept=True, reviewer=reviewer)
        # Snapshot for rollback
        snap_id = new_id("snap")
        snap_path = self.snapshots_dir / f"{snap_id}.json"
        snap_path.write_text(
            json.dumps(
                {
                    "snap_id": snap_id,
                    "candidate_id": candidate_id,
                    "snapshot_before": cand.snapshot_before,
                    "payload": cand.payload,
                    "ts": utc_now(),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        cand.status = "promoted"
        cand.payload["snap_id"] = snap_id
        cand.updated_at = utc_now()
        self._write_cand(cand)
        self._save()
        return {
            "ok": True,
            "candidate_id": candidate_id,
            "snap_id": snap_id,
            "note": "Promoted metadata only — live product code unchanged. Apply payload via explicit caller.",
            "human_level_agi": False,
        }

    def rollback(self, candidate_id: str) -> dict[str, Any]:
        """Roll back a promoted/accepted candidate using stored snapshot_before."""
        cand = self._get(candidate_id)
        restored = deepcopy(cand.snapshot_before)
        cand.status = "rolled_back"
        cand.updated_at = utc_now()
        self._write_cand(cand)
        cdir = self.candidates_dir / candidate_id
        (cdir / "rollback.json").write_text(
            json.dumps({"restored": restored, "ts": utc_now()}, indent=2, default=str),
            encoding="utf-8",
        )
        self._save()
        return {
            "ok": True,
            "candidate_id": candidate_id,
            "restored": restored,
            "status": cand.status,
            "human_level_agi": False,
        }

    def discard(self, candidate_id: str) -> None:
        cand = self._get(candidate_id)
        cand.status = "rejected"
        cand.updated_at = utc_now()
        cdir = self.candidates_dir / candidate_id
        if cdir.exists():
            shutil.rmtree(cdir, ignore_errors=True)
        self.candidates = [c for c in self.candidates if c.candidate_id != candidate_id]
        self._save()

    def summary(self) -> dict[str, Any]:
        by = {}
        for c in self.candidates:
            by[c.status] = by.get(c.status, 0) + 1
        return {
            "version": self.version,
            "lab_dir": str(self.lab_dir),
            "total": len(self.candidates),
            "by_status": by,
            "recent": [c.to_dict() for c in self.candidates[-8:]],
            "human_level_agi": False,
            "note": "Isolated lab only — no autonomous live code self-modification.",
        }

    def propose_from_prediction_error(
        self,
        *,
        action: str,
        prediction_error: float,
        observation_preview: str,
        snapshot_before: Optional[dict[str, Any]] = None,
    ) -> ModCandidate:
        """Scaffold a cue_map candidate when prediction error is high."""
        return self.propose(
            kind="cue_map",
            title=f"Improve expected cues for {action}",
            payload={
                "expected_cues": {action: [w for w in (observation_preview or "").lower().split() if len(w) > 3][:8]},
                "from_prediction_error": prediction_error,
            },
            evidence=[f"prediction_error={prediction_error}", f"action={action}"],
            snapshot_before=snapshot_before,
        )

    # --- persistence ----------------------------------------------------------

    def _get(self, candidate_id: str) -> ModCandidate:
        for c in self.candidates:
            if c.candidate_id == candidate_id:
                return c
        raise KeyError(candidate_id)

    def _write_cand(self, cand: ModCandidate) -> None:
        cdir = self.candidates_dir / cand.candidate_id
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "candidate.json").write_text(json.dumps(cand.to_dict(), indent=2, default=str), encoding="utf-8")

    def _save(self) -> None:
        data = {
            "version": self.version,
            "candidates": [c.to_dict() for c in self.candidates],
            "updated_at": utc_now(),
            "human_level_agi": False,
        }
        self.index_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            self.candidates = []
            for row in data.get("candidates") or []:
                known = {f.name for f in ModCandidate.__dataclass_fields__.values()}  # type: ignore[attr-defined]
                self.candidates.append(ModCandidate(**{k: v for k, v in row.items() if k in known}))
        except Exception:  # noqa: BLE001
            self.candidates = []
