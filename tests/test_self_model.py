"""Tests for introspection / self-model answers."""

from cognitive_ai.brain import Brain


def test_who_are_you():
    brain = Brain(name="TestBrain")
    ans = brain.ask("Who are you?")
    assert "TestBrain" in ans
    assert "LLM" in ans or "llm" in ans.lower() or "symbolic" in ans.lower()


def test_what_do_you_know():
    brain = Brain()
    brain.load_domain("animals")
    ans = brain.ask("What do you know?")
    assert "animals" in ans.lower() or "Facts" in ans
    assert "tweety" in ans or "fact" in ans.lower()


def test_trying_to_do_reports_goals():
    brain = Brain()
    brain.load_domain("animals")
    brain.set_goal("Feed tweety", "tweety", "fed", "true")
    ans = brain.ask("What are you trying to do?")
    assert "Feed tweety" in ans or "fed" in ans


def test_modules_and_limits():
    brain = Brain()
    mods = brain.ask("modules")
    assert "inference" in mods
    assert "planner" in mods
    limits = brain.ask("What are your limits?")
    assert "LLM" in limits or "neural" in limits.lower() or "symbolic" in limits.lower()


def test_why_via_ask():
    brain = Brain()
    brain.load_domain("animals")
    brain.reason()
    ans = brain.ask("why did you conclude (tweety can fly)?")
    assert "tweety" in ans and "fly" in ans
