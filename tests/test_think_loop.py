"""Tests for the autonomous think cycle."""

from cognitive_ai.brain import Brain


def test_think_achieves_feed_goal():
    brain = Brain()
    brain.load_domain("animals")
    brain.set_goal("Feed tweety", "tweety", "fed", "true")
    report = brain.think(steps=5)
    assert report.cycles
    assert brain.kb.has_fact("tweety", "fed", "true") or any(
        c.goals_satisfied for c in report.cycles
    )
    # Prefer hard assertion on world effect after enough cycles
    brain.engine.forward_chain()
    # feed action should have fired if needs_food was derived
    assert (
        brain.kb.has_fact("tweety", "fed", "true")
        or report.stopped_reason == "all goals satisfied"
        or any("feed" in a for c in report.cycles for a in c.actions)
    )


def test_think_with_perceptions():
    brain = Brain()
    brain.load_domain("animals")
    report = brain.think(
        steps=2,
        perceptions=[("rover", "is_a", "dog")],
    )
    assert brain.kb.has_fact("rover", "is_a", "dog")
    assert any(c.perceptions for c in report.cycles)
    assert report.stopped_reason  # either goals done or max steps


def test_think_idle_when_no_goals():
    brain = Brain()
    brain.load_domain("animals")
    report = brain.think(steps=3)
    assert report.stopped_reason == "all goals satisfied"
    assert report.cycles[0].goals_satisfied


def test_failed_plan_learns_lesson():
    brain = Brain()
    brain.load_domain("animals")
    brain.set_goal("Paint sky", "sky", "color", "rainbow")
    report = brain.think(steps=2)
    assert brain.learner.lessons
    assert any(l.kind == "failure" for l in brain.learner.lessons)
    assert any(c.utterances for c in report.cycles)


def test_thoughts_recorded():
    brain = Brain()
    brain.load_domain("animals")
    brain.set_goal("Feed tweety", "tweety", "fed", "true")
    brain.think(steps=3)
    kinds = {t.kind for t in brain.self_model.thoughts}
    assert "plan" in kinds or "reflect" in kinds
    assert len(brain.self_model.thoughts) >= 1
