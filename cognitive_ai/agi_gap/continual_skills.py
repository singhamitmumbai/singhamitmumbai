"""Continual learning + automatically growing skill library (verified only)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from ..core.ids import new_id
from ..core.records import LearningExample, utc_now
from ..learning_sys.induction import ProgramInducer
from ..learning_sys.learner import LearningSystem
from ..memory.store import Episode, MemorySystem, Procedure


@dataclass
class SkillCard:
    name: str
    steps: list[str]
    domain: str = "general"
    success_count: int = 0
    failure_count: int = 0
    source: str = "seed"  # seed|induced|transferred|user
    version: str = "1"
    skill_id: str = field(default_factory=lambda: new_id("skill"))
    created_at: str = field(default_factory=utc_now)
    tags: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        n = self.success_count + self.failure_count
        if n == 0:
            return 0.5
        return self.success_count / n

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["score"] = self.score
        return d


class GrowingSkillLibrary:
    """Skills grow from verified episodes/examples — never from unverified self-labels."""

    SEED_SKILLS: list[tuple[str, list[str], str]] = [
        ("explain_concept", ["normalize_question", "retrieve_kb", "compose_answer", "cite"], "open_world"),
        ("web_research", ["rewrite_query", "local_lookup", "live_search", "synthesize_cite"], "search"),
        ("compare_entities", ["fetch_a", "fetch_b", "diff", "compose_compare"], "open_world"),
        ("howto_plan", ["clarify_goal", "list_steps", "verify_safety", "respond"], "planning"),
        ("math_eval", ["parse_expr", "safe_eval", "respond"], "math"),
        ("transfer_procedure", ["load_proc", "map_entities", "scale_qty", "store_new"], "transfer"),
        ("multimodal_ingest", ["detect_modality", "recognize", "attach_context", "route"], "perception"),
        ("long_horizon_loop", ["decompose", "act", "verify", "reflect", "continue_or_stop"], "autonomy"),
    ]

    def __init__(self, memory: Optional[MemorySystem] = None) -> None:
        self.memory = memory or MemorySystem()
        self.learner = LearningSystem(self.memory)
        self.inducer = ProgramInducer()
        self.skills: dict[str, SkillCard] = {}
        self.growth_log: list[dict[str, Any]] = []
        for name, steps, domain in self.SEED_SKILLS:
            self._register(SkillCard(name=name, steps=steps, domain=domain, source="seed"))

    def _register(self, card: SkillCard) -> SkillCard:
        self.skills[card.name] = card
        self.memory.add_procedure(
            Procedure(
                name=card.name,
                steps=card.steps,
                applicability=[f"domain:{card.domain}", card.source],
                version=card.version,
                success_count=card.success_count,
                failure_count=card.failure_count,
            )
        )
        return card

    def list_skills(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in sorted(self.skills.values(), key=lambda x: -x.score)]

    def record_outcome(self, name: str, *, success: bool) -> Optional[SkillCard]:
        card = self.skills.get(name)
        if not card:
            return None
        if success:
            card.success_count += 1
        else:
            card.failure_count += 1
        proc = self.memory.procedures.get(name)
        if proc:
            proc.success_count = card.success_count
            proc.failure_count = card.failure_count
        return card

    def learn_from_episode(self, episode: Episode, *, domain: str = "general") -> Optional[SkillCard]:
        """Induce or reinforce a skill from a verified episode outcome."""
        if not episode.actions:
            return None
        steps = []
        for a in episode.actions:
            if isinstance(a, dict):
                steps.append(str(a.get("op") or a.get("step") or a.get("name") or "act"))
            else:
                steps.append(str(a))
        if not steps:
            return None
        success = str(episode.outcome).lower() in ("ok", "success", "succeeded", "verified", "true")
        # Name from step signature
        sig = hashlib.sha1("|".join(steps).encode()).hexdigest()[:8]
        name = f"auto_{domain}_{sig}"
        if name in self.skills:
            return self.record_outcome(name, success=success)
        if not success:
            # Do not grow skills from failures
            self.growth_log.append({"event": "skip_failed_episode", "steps": steps})
            return None
        card = SkillCard(
            name=name,
            steps=steps,
            domain=domain,
            source="induced",
            success_count=1,
            tags=["continual", "auto"],
        )
        self._register(card)
        self.growth_log.append({"event": "grew_skill", "skill": name, "steps": steps})
        return card

    def learn_from_examples(
        self,
        name: str,
        positives: list[LearningExample],
        counterexamples: Optional[list[LearningExample]] = None,
        *,
        domain: str = "general",
        release: bool = True,
    ) -> dict[str, Any]:
        prog = self.inducer.induce(
            name,
            positives,
            counterexamples,
            applicability=[f"domain:{domain}", "continual"],
            memory=self.memory,
            release=release,
        )
        if prog.status in ("candidate", "released") and prog.steps:
            card = SkillCard(
                name=name,
                steps=prog.steps,
                domain=domain,
                source="induced",
                version="induced_v1",
                tags=["continual"],
            )
            if release:
                card.success_count = prog.support
            self._register(card)
            self.growth_log.append({"event": "induced", "program": prog.to_dict()})
        return prog.to_dict()

    def continual_tick(self, *, train_fn=None) -> dict[str, Any]:
        cycle = self.learner.continual_cycle(train_fn=train_fn)
        cycle["skill_count"] = len(self.skills)
        cycle["growth_events"] = len(self.growth_log)
        cycle["human_level_agi"] = False
        return cycle

    def summary(self) -> dict[str, Any]:
        return {
            "skills": len(self.skills),
            "growth_events": len(self.growth_log),
            "top": self.list_skills()[:5],
            "human_level_agi": False,
        }
