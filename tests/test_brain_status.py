"""Extra brain API coverage for production OS integration."""

from cognitive_ai.brain import Brain


def test_status_snapshot():
    brain = Brain(name="S")
    brain.load_domain("animals")
    brain.mark_booted()
    st = brain.status()
    assert st["name"] == "S"
    assert st["domain"] == "animals"
    assert st["facts"] > 0
    assert st["booted_at"]


def test_retract_fact():
    brain = Brain()
    brain.assert_fact("a", "b", "c")
    assert brain.kb.has_fact("a", "b", "c")
    assert brain.retract_fact("a", "b", "c")
    assert not brain.kb.has_fact("a", "b", "c")


def test_abandon_goal():
    brain = Brain()
    brain.load_domain("animals")
    g = brain.set_goal("x", "tweety", "fed", "true")
    assert brain.abandon_goal(g.goal_id)
    assert g.status == "abandoned"
