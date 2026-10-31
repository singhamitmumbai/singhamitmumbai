"""Mother world model — structured state_t + action → predicted state_t+1.

Builds on domain WorldModel transitions where applicable, plus a lightweight
entity/relation scratch state for dialogue/task cycles.
human_level_agi=false.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from ..core.records import utc_now
from ..world.model import WorldModel


@dataclass
class WorldState:
    entities: dict[str, dict[str, Any]] = field(default_factory=dict)
    relations: list[dict[str, Any]] = field(default_factory=list)
    resources: dict[str, float] = field(default_factory=dict)
    flags: dict[str, Any] = field(default_factory=dict)
    last_action: str = ""
    updated_at: str = field(default_factory=utc_now)
    human_level_agi: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorldState":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


class MotherWorldModel:
    """Predict / observe / score prediction error with structured state."""

    version = "mother_world_v1"

    def __init__(self) -> None:
        self.state = WorldState(
            resources={"attention": 1.0, "tool_budget": 5.0},
            flags={"session_active": True},
        )
        self.domain = WorldModel()
        self.history: list[dict[str, Any]] = []

    def snapshot(self) -> dict[str, Any]:
        return {"version": self.version, "state": self.state.to_dict(), "human_level_agi": False}

    def ingest_utterance(self, text: str, *, intent: str = "") -> WorldState:
        """Update entities lightly from text (perceive)."""
        s = self.state
        for m in re.finditer(r"\b(?:project\s+[a-z0-9_-]+|[A-Z][a-zA-Z0-9_-]{2,})\b", text or ""):
            name = m.group(0)
            key = name.lower().replace(" ", "_")
            s.entities.setdefault(key, {"name": name, "mentions": 0})
            s.entities[key]["mentions"] = int(s.entities[key].get("mentions") or 0) + 1
        if intent:
            s.flags["last_intent"] = intent
        s.updated_at = utc_now()
        return s

    def predict(self, action: str, *, args: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """state_t + action → predicted_state_t+1 (+ optional domain transition)."""
        args = dict(args or {})
        predicted = deepcopy(self.state)
        predicted.last_action = action
        predicted.flags["predicted_action"] = action
        # Resource burn model
        cost = float(args.get("cost") or {"dialogue": 0.05, "complete_ai": 0.2, "coding": 0.35, "inspect_repo": 0.25}.get(action, 0.1))
        predicted.resources["attention"] = max(0.0, float(predicted.resources.get("attention", 1.0)) - cost)
        predicted.resources["tool_budget"] = max(
            0.0, float(predicted.resources.get("tool_budget", 5.0)) - (1.0 if action not in ("dialogue",) else 0.0)
        )
        # Domain transition if args carry stock/invoice state
        domain_pred = None
        if action in self.domain.transitions and args.get("state"):
            domain_pred = self.domain.predict(dict(args["state"]), action, args)
            predicted.flags["domain_prediction"] = domain_pred
        # Expected observation cues
        expected_cues = {
            "dialogue": ["hello", "ready", "mother", "help"],
            "inspect_repo": ["diagnostic", "repository", "mother", "complete"],
            "coding": ["def ", "code", "function", "generated", "verify"],
            "complete_ai": ["invoice", "project", "cannot", "gravity", "ok", "thinking"],
        }.get(action, ["ok"])
        predicted.flags["expected_cues"] = expected_cues
        predicted.updated_at = utc_now()
        return {
            "ok": True,
            "action": action,
            "predicted": predicted.to_dict(),
            "expected_cues": expected_cues,
            "domain": domain_pred,
            "human_level_agi": False,
        }

    def observe(self, action: str, observation: str, *, predicted: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Commit observation; compute structured prediction error."""
        cues = list((predicted or {}).get("expected_cues") or [])
        low = (observation or "").lower()
        hits = sum(1 for c in cues if c.lower() in low)
        cue_score = hits / max(1, len(cues))
        # Also track resource consistency (attention shouldn't increase unexpectedly)
        prev_attn = float(self.state.resources.get("attention", 1.0))
        # Apply soft updates from observation length as evidence of work done
        self.state.last_action = action
        self.state.flags["last_observation_len"] = len(observation or "")
        self.state.flags["last_cue_score"] = round(cue_score, 3)
        self.state.resources["attention"] = max(0.0, prev_attn - 0.02)
        self.state.updated_at = utc_now()
        error = round(1.0 - cue_score, 3)
        row = {
            "action": action,
            "cue_score": round(cue_score, 3),
            "prediction_error": error,
            "observation_preview": (observation or "")[:160],
            "ts": utc_now(),
            "human_level_agi": False,
        }
        self.history.append(row)
        if len(self.history) > 100:
            self.history = self.history[-100:]
        return row

    def summary(self) -> dict[str, Any]:
        recent_err = [h.get("prediction_error", 0) for h in self.history[-20:]]
        avg = sum(recent_err) / len(recent_err) if recent_err else 0.0
        return {
            "version": self.version,
            "entities": len(self.state.entities),
            "relations": len(self.state.relations),
            "resources": dict(self.state.resources),
            "history": len(self.history),
            "avg_prediction_error": round(avg, 3),
            "human_level_agi": False,
        }
