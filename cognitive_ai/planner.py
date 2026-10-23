"""Planner: goals → subgoals → actions via means-ends analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
from uuid import uuid4

from .knowledge import KnowledgeBase, Triple
from .inference import InferenceEngine, unify, substitute


@dataclass
class Goal:
    """A desired world state expressed as a triple pattern (may use ?vars)."""

    description: str
    target: Triple
    priority: float = 1.0
    goal_id: str = field(default_factory=lambda: uuid4().hex[:10])
    status: str = "open"  # open | active | satisfied | failed | abandoned
    parent_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "target": str(self.target),
            "priority": self.priority,
            "status": self.status,
            "parent_id": self.parent_id,
        }


@dataclass
class Action:
    """An executable (symbolic) action the agent can take."""

    name: str
    description: str
    preconditions: list[Triple] = field(default_factory=list)
    effects: list[Triple] = field(default_factory=list)
    # Optional: effects that remove facts
    deletes: list[Triple] = field(default_factory=list)
    cost: float = 1.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "preconditions": [str(p) for p in self.preconditions],
            "effects": [str(e) for e in self.effects],
            "deletes": [str(d) for d in self.deletes],
            "cost": self.cost,
        }


@dataclass
class PlanStep:
    action: Action
    bindings: dict[str, str]
    achieves: Triple

    def grounded_action_name(self) -> str:
        if not self.bindings:
            return self.action.name
        parts = ", ".join(f"{k}={v}" for k, v in sorted(self.bindings.items()))
        return f"{self.action.name}[{parts}]"

    def to_dict(self) -> dict:
        return {
            "action": self.grounded_action_name(),
            "description": self.action.description,
            "bindings": dict(self.bindings),
            "achieves": str(self.achieves),
        }


@dataclass
class Plan:
    goal: Goal
    steps: list[PlanStep] = field(default_factory=list)
    subgoals: list[Goal] = field(default_factory=list)
    success: bool = False
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "goal": self.goal.to_dict(),
            "success": self.success,
            "explanation": self.explanation,
            "steps": [s.to_dict() for s in self.steps],
            "subgoals": [g.to_dict() for g in self.subgoals],
        }


class Planner:
    """Hierarchical / means-ends planner over symbolic actions and rules."""

    def __init__(
        self,
        kb: KnowledgeBase,
        engine: Optional[InferenceEngine] = None,
        actions: Optional[list[Action]] = None,
    ) -> None:
        self.kb = kb
        self.engine = engine or InferenceEngine(kb)
        self.actions: dict[str, Action] = {}
        self.goals: dict[str, Goal] = {}
        self.plan_history: list[Plan] = []
        if actions:
            for a in actions:
                self.register_action(a)

    def register_action(self, action: Action) -> None:
        self.actions[action.name] = action

    def set_goal(
        self,
        description: str,
        subject: str,
        predicate: str,
        obj: str,
        priority: float = 1.0,
    ) -> Goal:
        goal = Goal(
            description=description,
            target=Triple(subject, predicate, obj),
            priority=priority,
        )
        self.goals[goal.goal_id] = goal
        return goal

    def abandon_goal(self, goal_id: str) -> bool:
        g = self.goals.get(goal_id)
        if not g:
            return False
        g.status = "abandoned"
        return True

    def open_goals(self) -> list[Goal]:
        return sorted(
            (g for g in self.goals.values() if g.status in ("open", "active")),
            key=lambda g: -g.priority,
        )

    def is_satisfied(self, goal: Goal, extra: Optional[list[Triple]] = None) -> bool:
        facts = list(self.kb.facts())
        if extra:
            facts.extend(extra)
        for fact in facts:
            if unify(goal.target, fact) is not None:
                return True
        # Try proving via inference
        trace = self.engine.backward_chain(goal.target, extra_facts=extra)
        return trace.success

    def plan(
        self,
        goal: Optional[Goal] = None,
        max_depth: int = 8,
        extra_facts: Optional[list[Triple]] = None,
    ) -> Plan:
        """Means-ends: reduce differences between current state and goal."""
        if goal is None:
            opens = self.open_goals()
            if not opens:
                empty = Goal("none", Triple("*", "*", "*"))
                return Plan(
                    goal=empty,
                    success=True,
                    explanation="No open goals.",
                )
            goal = opens[0]

        goal.status = "active"
        result = Plan(goal=goal)

        if self.is_satisfied(goal, extra_facts):
            goal.status = "satisfied"
            result.success = True
            result.explanation = f"Goal already satisfied: {goal.target}"
            self.plan_history.append(result)
            return result

        steps, subgoals, ok = self._means_ends(
            goal.target, extra_facts or [], max_depth, set()
        )
        result.steps = steps
        result.subgoals = subgoals
        for sg in subgoals:
            self.goals[sg.goal_id] = sg
        result.success = ok
        if ok:
            result.explanation = (
                f"Plan for '{goal.description}' has {len(steps)} action(s) "
                f"and {len(subgoals)} subgoal(s)."
            )
        else:
            result.explanation = (
                f"Failed to find a complete plan for '{goal.description}'. "
                f"Partial: {len(steps)} step(s), {len(subgoals)} open subgoal(s)."
            )
            goal.status = "failed"
        self.plan_history.append(result)
        return result

    def _means_ends(
        self,
        target: Triple,
        extra: list[Triple],
        depth: int,
        visiting: set[tuple],
    ) -> tuple[list[PlanStep], list[Goal], bool]:
        if depth <= 0:
            return [], [], False

        key = target.as_tuple()
        if key in visiting:
            return [], [], False
        visiting = set(visiting) | {key}

        facts = list(self.kb.facts()) + list(extra)
        for fact in facts:
            if unify(target, fact) is not None:
                return [], [], True

        # Inference may establish the goal without actions
        if self.engine.backward_chain(target, extra_facts=extra).success:
            return [], [], True

        # Find actions whose effects unify with the target
        candidates: list[tuple[Action, dict[str, str], Triple]] = []
        for action in self.actions.values():
            for effect in action.effects:
                env = unify(effect, target)
                if env is not None:
                    candidates.append((action, env, substitute(effect, env)))

        # Prefer cheaper actions
        candidates.sort(key=lambda c: c[0].cost)

        best_partial: Optional[tuple[list[PlanStep], list[Goal]]] = None

        for action, env, achieves in candidates:
            # Ensure preconditions: plan for each unmet precondition
            pre_steps: list[PlanStep] = []
            pre_subgoals: list[Goal] = []
            all_ok = True
            simulated_extra = list(extra)

            for prec in action.preconditions:
                need = substitute(prec, env)
                # Already true?
                satisfied = any(
                    unify(need, f) is not None for f in facts + simulated_extra
                )
                if satisfied:
                    continue
                # Try inference (depth-limited via engine)
                if self.engine.backward_chain(need, extra_facts=simulated_extra).success:
                    continue
                # Recurse
                steps, sgs, ok = self._means_ends(
                    need, simulated_extra, depth - 1, visiting
                )
                if ok:
                    pre_steps.extend(steps)
                    for st in steps:
                        for eff in st.action.effects:
                            simulated_extra.append(substitute(eff, st.bindings))
                else:
                    pre_subgoals.extend(sgs)
                    if not sgs:
                        pre_subgoals.append(
                            Goal(
                                description=f"Satisfy precondition {need}",
                                target=need,
                                priority=0.5,
                            )
                        )
                    all_ok = False

            step = PlanStep(action=action, bindings=env, achieves=achieves)
            if all_ok:
                return pre_steps + [step], pre_subgoals, True

            # Keep best partial, but try other actions that may fully succeed
            partial = (pre_steps + [step], pre_subgoals)
            if best_partial is None or len(partial[1]) < len(best_partial[1]):
                best_partial = partial

        if best_partial is not None:
            return best_partial[0], best_partial[1], False

        # No action found — leave as open subgoal
        sg = Goal(
            description=f"Achieve {target}",
            target=target,
            priority=0.4,
        )
        return [], [sg], False

    def execute_step(
        self,
        step: PlanStep,
        assert_fn: Optional[Callable[[str, str, str], None]] = None,
        retract_fn: Optional[Callable[[str, str, str], None]] = None,
    ) -> list[Triple]:
        """Apply action effects to the knowledge base / callback."""
        applied: list[Triple] = []
        for d in step.action.deletes:
            fact = substitute(d, step.bindings)
            if retract_fn:
                retract_fn(*fact.as_tuple())
            else:
                self.kb.retract_fact(*fact.as_tuple())
        for e in step.action.effects:
            fact = substitute(e, step.bindings)
            if any(x.startswith("?") for x in fact.as_tuple()):
                continue
            if assert_fn:
                assert_fn(*fact.as_tuple())
            else:
                self.kb.assert_fact(*fact.as_tuple())
            applied.append(fact)
        return applied
