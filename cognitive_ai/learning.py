"""Learning: acquire facts/rules from interaction and failed plans."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from .knowledge import KnowledgeBase, Rule, Triple
from .planner import Plan


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Lesson:
    kind: str  # fact | rule | failure
    content: str
    detail: dict = field(default_factory=dict)
    lesson_id: str = field(default_factory=lambda: uuid4().hex[:10])
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return {
            "lesson_id": self.lesson_id,
            "kind": self.kind,
            "content": self.content,
            "detail": self.detail,
            "timestamp": self.timestamp,
        }


class Learner:
    """Heuristic symbolic learner — memory writes and simple rule induction."""

    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb
        self.lessons: list[Lesson] = []
        self._pair_counts: dict[tuple[tuple, tuple], int] = defaultdict(int)
        self._induced = 0

    def learn_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        source: str = "interaction",
    ) -> Lesson:
        added = self.kb.assert_fact(subject, predicate, obj)
        lesson = Lesson(
            kind="fact",
            content=f"{'Learned' if added else 'Reinforced'} fact ({subject} {predicate} {obj})",
            detail={"subject": subject, "predicate": predicate, "object": obj, "source": source, "new": added},
        )
        self.lessons.append(lesson)
        return lesson

    def learn_rule(self, rule: Rule) -> Lesson:
        rule.learned = True
        self.kb.add_rule(rule)
        lesson = Lesson(
            kind="rule",
            content=f"Learned rule '{rule.name}'",
            detail=rule.to_dict(),
        )
        self.lessons.append(lesson)
        return lesson

    def observe_cooccurrence(self, a: Triple, b: Triple) -> None:
        """Track that whenever a is true, b was also observed (for induction)."""
        self._pair_counts[(a.as_tuple(), b.as_tuple())] += 1

    def induce_rules(self, min_support: int = 2) -> list[Rule]:
        """Induce simple one-antecedent → one-consequent rules from co-occurrence."""
        induced: list[Rule] = []
        for (a_t, b_t), count in self._pair_counts.items():
            if count < min_support:
                continue
            if a_t == b_t:
                continue
            # Generalize constants that differ into variables when subjects match
            ant = Triple(*a_t)
            cons = Triple(*b_t)
            name = f"induced_{self._induced}"
            # Prefer subject-sharing generalization
            if ant.subject == cons.subject and ant.subject:
                ant = Triple("?x", ant.predicate, ant.object)
                cons = Triple("?x", cons.predicate, cons.object)
                name = f"induced_{ant.predicate}_implies_{cons.predicate}_{self._induced}"
            # Skip if identical rule exists
            if any(
                r.antecedents == [ant] and r.consequents == [cons]
                for r in self.kb.rules()
            ):
                continue
            rule = Rule(
                name=name,
                antecedents=[ant],
                consequents=[cons],
                confidence=min(1.0, 0.5 + 0.1 * count),
                learned=True,
            )
            self.kb.add_rule(rule)
            self._induced += 1
            lesson = Lesson(
                kind="rule",
                content=f"Induced rule {rule.name} (support={count})",
                detail=rule.to_dict(),
            )
            self.lessons.append(lesson)
            induced.append(rule)
        return induced

    def learn_from_failed_plan(self, plan: Plan) -> Lesson:
        """Record a failure lesson and optionally a blocking-fact question."""
        blockers = [str(sg.target) for sg in plan.subgoals]
        content = (
            f"Plan failed for goal '{plan.goal.description}'. "
            f"Missing: {', '.join(blockers) if blockers else 'unknown'}."
        )
        lesson = Lesson(
            kind="failure",
            content=content,
            detail={
                "goal": plan.goal.to_dict(),
                "subgoals": [g.to_dict() for g in plan.subgoals],
                "partial_steps": [s.to_dict() for s in plan.steps],
            },
        )
        self.lessons.append(lesson)
        # Write a meta-fact about the failure for future introspection
        self.kb.assert_fact(
            plan.goal.goal_id,
            "plan_failed_for",
            plan.goal.description.replace(" ", "_"),
        )
        if blockers:
            self.kb.assert_fact(
                plan.goal.goal_id,
                "blocked_by",
                blockers[0].replace(" ", "_"),
            )
        return lesson

    def learn_from_success(self, facts_used: list[Triple], conclusion: Triple) -> Optional[Rule]:
        """Strengthen associations between premises and a successful conclusion."""
        for f in facts_used:
            self.observe_cooccurrence(f, conclusion)
        induced = self.induce_rules(min_support=2)
        return induced[0] if induced else None

    def recent_lessons(self, n: int = 10) -> list[Lesson]:
        return self.lessons[-n:]

    def dump(self) -> dict:
        return {
            "lesson_count": len(self.lessons),
            "pair_counts": len(self._pair_counts),
            "recent": [x.to_dict() for x in self.recent_lessons(15)],
        }
