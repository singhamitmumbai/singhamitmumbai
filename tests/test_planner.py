"""Tests for means-ends planning and goal satisfaction."""

from cognitive_ai.brain import Brain
from cognitive_ai.knowledge import Triple
from cognitive_ai.planner import Action


def test_goal_already_satisfied():
    brain = Brain()
    brain.load_domain("animals")
    brain.kb.assert_fact("tweety", "fed", "true")
    goal = brain.set_goal("already done", "tweety", "fed", "true")
    plan = brain.planner.plan(goal)
    assert plan.success
    assert goal.status == "satisfied"
    assert plan.steps == []


def test_feed_plan_animals():
    brain = Brain()
    brain.load_domain("animals")
    # Derive tweety needs food
    brain.engine.forward_chain()
    assert brain.kb.has_fact("tweety", "needs", "food") or brain.engine.backward_chain(
        Triple("tweety", "needs", "food")
    ).success
    goal = brain.set_goal("Feed tweety", "tweety", "fed", "true")
    plan = brain.planner.plan(goal)
    assert plan.success
    assert any(s.action.name == "feed" for s in plan.steps)
    applied = []
    for step in plan.steps:
        applied.extend(brain.planner.execute_step(step))
    assert brain.kb.has_fact("tweety", "fed", "true")
    assert brain.planner.is_satisfied(goal)


def test_kitchen_cook_omelette_plan():
    brain = Brain()
    brain.load_domain("kitchen")
    goal = brain.set_goal("Make omelette", "omelette", "status", "ready")
    plan = brain.planner.plan(goal)
    assert plan.success
    names = [s.action.name for s in plan.steps]
    assert "cook_omelette" in names
    assert "get_from_fridge" in names or "get_pan" in names
    for step in plan.steps:
        brain.planner.execute_step(step)
    assert brain.kb.has_fact("omelette", "status", "ready")


def test_unachievable_goal_records_subgoals():
    brain = Brain()
    brain.load_domain("animals")
    # No action produces rainbow
    goal = brain.set_goal("Impossible", "sky", "color", "rainbow")
    plan = brain.planner.plan(goal)
    assert not plan.success
    assert goal.status == "failed"
    assert plan.subgoals or "Failed" in plan.explanation
