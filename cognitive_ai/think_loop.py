"""Think loop: perceive → update beliefs → reason → plan → act → reflect."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .knowledge import Triple


@dataclass
class CycleResult:
    step: int
    perceptions: list[str] = field(default_factory=list)
    inferences: list[str] = field(default_factory=list)
    plan_summary: str = ""
    actions: list[str] = field(default_factory=list)
    utterances: list[str] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)
    goals_satisfied: bool = False

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "perceptions": list(self.perceptions),
            "inferences": list(self.inferences),
            "plan_summary": self.plan_summary,
            "actions": list(self.actions),
            "utterances": list(self.utterances),
            "reflections": list(self.reflections),
            "goals_satisfied": self.goals_satisfied,
        }


@dataclass
class ThinkReport:
    cycles: list[CycleResult] = field(default_factory=list)
    stopped_reason: str = ""
    final_speech: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cycles": [c.to_dict() for c in self.cycles],
            "stopped_reason": self.stopped_reason,
            "final_speech": list(self.final_speech),
        }


class ThinkLoop:
    """Autonomous cognitive cycle runner bound to a Brain."""

    def __init__(self, brain: Any) -> None:
        self.brain = brain
        self.cycle_count = 0
        self.last_report: Optional[ThinkReport] = None

    def run(
        self,
        max_steps: int = 5,
        perceptions: Optional[list[tuple[str, str, str]]] = None,
        stop_when_goals_done: bool = True,
    ) -> ThinkReport:
        report = ThinkReport()
        pending = list(perceptions or [])

        for i in range(max_steps):
            self.cycle_count += 1
            cycle = self._one_cycle(self.cycle_count, pending)
            pending = []  # only inject perceptions on first cycle unless re-queued
            report.cycles.append(cycle)
            report.final_speech.extend(cycle.utterances)

            if stop_when_goals_done and cycle.goals_satisfied:
                report.stopped_reason = "all goals satisfied"
                break
        else:
            report.stopped_reason = f"reached max_steps={max_steps}"

        self.last_report = report
        self.brain.self_model.record(
            "reflect",
            f"Think loop finished: {report.stopped_reason} after {len(report.cycles)} cycle(s).",
            report.to_dict(),
        )
        return report

    def _one_cycle(
        self,
        step: int,
        perceptions: list[tuple[str, str, str]],
    ) -> CycleResult:
        brain = self.brain
        result = CycleResult(step=step)

        # 1. Perceive
        for s, p, o in perceptions:
            brain.working.perceive(s, p, o, source="perception")
            brain.kb.assert_fact(s, p, o)
            msg = f"Perceived ({s} {p} {o})"
            result.perceptions.append(msg)
            brain.self_model.record("perceive", msg)

        # Sync high-confidence beliefs into focus
        entities = []
        for b in brain.working.active_beliefs(0.5)[:10]:
            entities.extend([b.content.subject, b.content.object])
        brain.working.update_situation(
            description=f"cycle-{step}",
            entities=sorted(set(entities))[:20],
            focus=entities[0] if entities else None,
        )

        # 2. Reason (forward chain)
        extras = brain.working.belief_triples(0.3)
        fwd = brain.engine.forward_chain(extra_facts=extras, write_back=True)
        for derived in fwd.derived:
            brain.working.perceive(
                *derived.as_tuple(),
                confidence=0.9,
                source="inference",
                justification=fwd.explanation,
            )
            result.inferences.append(str(derived))
        if fwd.derived:
            brain.self_model.record(
                "reason",
                fwd.explanation,
                {"derived": [str(d) for d in fwd.derived]},
            )

        # Resolve open questions if we can
        for q in list(brain.working.open_questions()):
            if q.pattern:
                trace = brain.engine.backward_chain(q.pattern, extra_facts=extras)
                if trace.success and trace.derived:
                    res = str(trace.derived[0])
                    brain.working.resolve_question(q.question_id, res)
                    utter = f"Resolved question '{q.text}' → {res}"
                    result.utterances.append(utter)
                    brain.self_model.record("speak", utter)

        # 3. Plan
        opens = brain.planner.open_goals()
        if not opens:
            result.plan_summary = "no open goals"
            result.goals_satisfied = True
            result.reflections.append("Nothing to pursue; idle.")
            brain.self_model.record("reflect", "Idle — no open goals.")
            return result

        plan = brain.planner.plan(opens[0])
        result.plan_summary = plan.explanation
        brain.self_model.record("plan", plan.explanation, plan.to_dict())

        # 4. Act
        if plan.success and plan.steps:
            for step_plan in plan.steps:
                applied = brain.planner.execute_step(step_plan)
                for fact in applied:
                    brain.working.perceive(
                        *fact.as_tuple(),
                        confidence=1.0,
                        source="action",
                        justification=f"effect of {step_plan.grounded_action_name()}",
                    )
                    act_msg = f"Acted {step_plan.grounded_action_name()} → {fact}"
                    result.actions.append(act_msg)
                    brain.self_model.record("act", act_msg)
            # Re-check goal
            if brain.planner.is_satisfied(plan.goal):
                plan.goal.status = "satisfied"
                utter = f"Goal achieved: {plan.goal.description}"
                result.utterances.append(utter)
                brain.self_model.record("speak", utter)
                # Learn associations from successful plan effects
                if plan.steps:
                    last = plan.steps[-1]
                    for prem in last.action.preconditions:
                        from .inference import substitute

                        brain.learner.observe_cooccurrence(
                            substitute(prem, last.bindings),
                            last.achieves,
                        )
        elif not plan.success:
            lesson = brain.learner.learn_from_failed_plan(plan)
            result.reflections.append(lesson.content)
            brain.self_model.record("learn", lesson.content, lesson.to_dict())
            for sg in plan.subgoals:
                brain.working.add_question(
                    f"How to achieve {sg.target}?",
                    pattern=sg.target,
                    priority=0.6,
                )
            utter = f"I cannot fully achieve '{plan.goal.description}' yet."
            result.utterances.append(utter)
            brain.self_model.record("speak", utter)

        # 5. Reflect
        remaining = brain.planner.open_goals()
        result.goals_satisfied = len(remaining) == 0
        reflection = (
            f"Cycle {step}: beliefs={len(brain.working.active_beliefs())}, "
            f"open_goals={len(remaining)}, "
            f"inferences={len(result.inferences)}, actions={len(result.actions)}"
        )
        result.reflections.append(reflection)
        brain.self_model.record("reflect", reflection)
        brain.working.decay_all()
        return result
