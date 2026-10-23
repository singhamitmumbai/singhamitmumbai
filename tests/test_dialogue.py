"""Tests for symbolic natural-language dialogue."""

from cognitive_ai.brain import Brain
from cognitive_ai.dialogue import respond


def test_greeting():
    brain = Brain()
    brain.load_domain("animals")
    brain.reason()
    ans = respond(brain, "hello")
    assert "symbolic" in ans.lower() or "ChatBrain" in ans or "Hello" in ans


def test_can_tweety_fly():
    brain = Brain()
    brain.load_domain("animals")
    brain.reason()
    ans = respond(brain, "can tweety fly?")
    assert ans.lower().startswith("yes")
    assert "fly" in ans.lower()


def test_tell_me_about_tweety():
    brain = Brain()
    brain.load_domain("animals")
    brain.reason()
    ans = respond(brain, "tell me about tweety")
    assert "tweety" in ans.lower()
    assert "is_a" in ans or "canary" in ans.lower() or "bird" in ans.lower()


def test_feed_tweety_acts():
    brain = Brain()
    brain.load_domain("animals")
    brain.reason()
    ans = respond(brain, "feed tweety")
    assert "Feed tweety" in ans or "fed" in ans.lower()
    assert brain.kb.has_fact("tweety", "fed", "true")


def test_help():
    brain = Brain()
    brain.load_domain("animals")
    ans = respond(brain, "help")
    assert "can tweety fly" in ans.lower() or "assert" in ans.lower()


def test_is_opus_a_bird():
    brain = Brain()
    brain.load_domain("animals")
    brain.reason()
    ans = respond(brain, "is opus a bird?")
    assert ans.lower().startswith("yes")
