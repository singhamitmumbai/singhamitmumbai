"""Brain: wires all cognitive modules into one agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .beliefs import WorkingMemory
from .inference import InferenceEngine
from .knowledge import KnowledgeBase, Rule, Triple
from .learning import Learner
from .planner import Action, Goal, Planner
from .self_model import SelfModel
from .think_loop import ThinkLoop, ThinkReport


class Brain:
    """Top-level symbolic cognitive agent."""

    def __init__(
        self,
        name: str = "SymbolicBrain",
        storage: Optional[Path | str] = None,
    ) -> None:
        self.name = name
        self.kb = KnowledgeBase(storage)
        self.working = WorkingMemory()
        self.engine = InferenceEngine(self.kb)
        self.planner = Planner(self.kb, self.engine)
        self.learner = Learner(self.kb)
        self.self_model = SelfModel(name=name)
        self.self_model.attach(self)
        self.think_loop = ThinkLoop(self)
        self._utterances: list[str] = []
        self.booted_at: Optional[str] = None

    # --- knowledge / beliefs --------------------------------------------------

    def assert_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        believe: bool = True,
        confidence: float = 1.0,
        learn: bool = False,
    ) -> None:
        self.kb.assert_fact(subject, predicate, obj)
        if believe:
            self.working.perceive(
                subject,
                predicate,
                obj,
                confidence=confidence,
                source="user",
                justification="user assertion",
            )
        if learn:
            self.learner.learn_fact(subject, predicate, obj, source="user")
        self.self_model.record(
            "perceive",
            f"Asserted ({subject} {predicate} {obj})",
        )

    def add_rule(self, rule: Rule) -> None:
        self.kb.add_rule(rule)
        self.self_model.record("learn", f"Rule installed: {rule.name}")

    def register_action(self, action: Action) -> None:
        self.planner.register_action(action)

    def set_goal(
        self,
        description: str,
        subject: str,
        predicate: str,
        obj: str,
        priority: float = 1.0,
    ) -> Goal:
        goal = self.planner.set_goal(description, subject, predicate, obj, priority)
        self.working.update_situation(
            description=f"pursuing: {description}",
            focus=subject,
        )
        self.self_model.record("plan", f"Goal set: {description} → {goal.target}")
        return goal

    # --- cognition ------------------------------------------------------------

    def reason(self, write_back: bool = True) -> dict:
        extras = self.working.belief_triples()
        trace = self.engine.forward_chain(extra_facts=extras, write_back=write_back)
        for d in trace.derived:
            self.working.perceive(
                *d.as_tuple(),
                confidence=0.9,
                source="inference",
                justification=trace.explanation,
            )
        self.self_model.record("reason", trace.explanation, trace.to_dict())
        return trace.to_dict()

    def prove(self, subject: str, predicate: str, obj: str) -> dict:
        goal = Triple(subject, predicate, obj)
        extras = self.working.belief_triples()
        trace = self.engine.backward_chain(goal, extra_facts=extras)
        self.self_model.record("reason", trace.explanation, trace.to_dict())
        return trace.to_dict()

    def think(
        self,
        steps: int = 5,
        perceptions: Optional[list[tuple[str, str, str]]] = None,
    ) -> ThinkReport:
        report = self.think_loop.run(max_steps=steps, perceptions=perceptions)
        self._utterances.extend(report.final_speech)
        return report

    def ask(self, question: str) -> str:
        """Answer introspective, factual, or lightly natural questions."""
        from .dialogue import respond

        answer = respond(self, question)
        self.self_model.record("speak", answer, {"question": question})
        self._utterances.append(answer)
        return answer

    def ask_about_world(self, question: str) -> Optional[str]:
        """Very light factual lookup: 'is X Y Z?' or query pattern."""
        q = question.strip()
        import re

        m = re.search(r"\((\S+)\s+(\S+)\s+(\S+)\)", q)
        if m:
            s, p, o = m.group(1), m.group(2), m.group(3)
            if self.kb.has_fact(s, p, o):
                return f"Yes — ({s} {p} {o}) is known."
            # Try prove
            trace = self.engine.backward_chain(Triple(s, p, o))
            if trace.success:
                return f"Yes — derived {trace.derived[0]}: {trace.explanation}"
            hits = self.kb.query(s if s != "?" else "*", p if p != "?" else "*", o if o != "?" else "*")
            if hits:
                return "Matches:\n" + "\n".join(f"  {t}" for t in hits[:20])
            return f"No — I cannot establish ({s} {p} {o})."
        return None

    def why(self, subject: str, predicate: str, obj: str) -> str:
        return self.engine.why(subject, predicate, obj)

    def retract_fact(self, subject: str, predicate: str, obj: str) -> bool:
        ok = self.kb.retract_fact(subject, predicate, obj)
        if ok:
            self.self_model.record("perceive", f"Retracted ({subject} {predicate} {obj})")
        return ok

    def abandon_goal(self, goal_id: str) -> bool:
        return self.planner.abandon_goal(goal_id)

    def status(self) -> dict[str, Any]:
        """Compact operational snapshot for OS / monitoring."""
        opens = self.planner.open_goals()
        return {
            "name": self.name,
            "domain": self.kb.domain,
            "facts": len(self.kb),
            "rules": len(self.kb.rules()),
            "beliefs": len(self.working.active_beliefs()),
            "open_questions": len(self.working.open_questions()),
            "open_goals": len(opens),
            "goals": [g.to_dict() for g in opens],
            "actions": len(self.planner.actions),
            "think_cycles": self.think_loop.cycle_count,
            "thoughts": len(self.self_model.thoughts),
            "lessons": len(self.learner.lessons),
            "storage": self.kb.summary().get("path"),
            "booted_at": self.booted_at,
        }

    # --- persistence / dump ---------------------------------------------------

    def save(self, path: Optional[Path | str] = None) -> Path:
        return self.kb.save(path)

    def load(self, path: Optional[Path | str] = None) -> None:
        self.kb.load(path)

    def dump_mind(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status(),
            "knowledge": self.kb.summary(),
            "facts": [str(t) for t in self.kb.facts()],
            "rules": [r.to_dict() for r in self.kb.rules()],
            "working_memory": self.working.dump(),
            "goals": [g.to_dict() for g in self.planner.goals.values()],
            "actions": [a.to_dict() for a in self.planner.actions.values()],
            "self_model": self.self_model.dump(),
            "lessons": self.learner.dump(),
            "recent_inference": (
                self.engine.history[-1].to_dict() if self.engine.history else None
            ),
            "recent_plan": (
                self.planner.plan_history[-1].to_dict()
                if self.planner.plan_history
                else None
            ),
        }

    def dump_mind_json(self, indent: int = 2) -> str:
        return json.dumps(self.dump_mind(), indent=indent)

    def load_domain(self, domain: str) -> None:
        from . import domains

        domains.load_domain(self, domain)

    def mark_booted(self) -> None:
        from datetime import datetime, timezone

        self.booted_at = datetime.now(timezone.utc).isoformat()
        self.self_model.record("reflect", f"Brain '{self.name}' booted")
