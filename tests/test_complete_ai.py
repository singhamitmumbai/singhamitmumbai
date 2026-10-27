"""Complete AI — ChatGPT-class general chat + full stack."""

from __future__ import annotations

from cognitive_ai.chat.llm_bridge import LLMChatBridge
from cognitive_ai.complete_ai import CompleteAI
from cognitive_ai.commonsense.chat import OpenDomainChat


def test_complete_ai_identity_and_status():
    ai = CompleteAI(seed_demo=True, enable_llm=False)
    st = ai.status()
    assert st["name"] == "Complete AI"
    assert st["human_level_agi"] is False
    assert st["tagline"] == "Complete AI"
    assert st["version"].startswith("1.")
    assert "Complete AI" in ai.greeting()


def test_complete_ai_routes_tasks_and_chat():
    ai = CompleteAI(seed_demo=True, enable_llm=False)
    inv = ai.handle("Show unpaid invoices for Project A")
    assert inv["route"] in ("finderbay", "general")
    assert inv["human_level_agi"] is False
    assert inv["llm_at_runtime"] is False

    who = ai.handle("Who are you?")
    assert "Complete AI" in who["text"]

    peng = ai.handle("Can penguins fly?")
    assert peng["llm_at_runtime"] is False
    assert "penguin" in peng["text"].lower() or "fly" in peng["text"].lower()


def test_complete_ai_vision_and_coding_hooks():
    ai = CompleteAI(seed_demo=True, enable_llm=False)
    vis = ai.handle("Count people in image ./crowd_people.jpg")
    assert vis["route"] in ("vision", "finderbay", "general")
    assert "person" in vis["text"].lower() or "Counted" in vis["text"]

    code = ai.handle("generate function add(a, b)")
    assert code["route"] == "coding"
    assert code.get("ok") or "def add" in str(code.get("code") or code.get("text") or "")


def test_llm_auto_enables_with_api_key():
    bridge = LLMChatBridge(enabled=None, api_key="sk-test-not-real")
    assert bridge.enabled is True
    assert bridge.available is True


def test_llm_bridge_disabled_explicitly():
    bridge = LLMChatBridge(enabled=False, api_key="sk-test")
    assert bridge.available is False
    out = bridge.chat("Hello")
    assert out["ok"] is False
    assert out["human_level_agi"] is False


def test_llm_forced_mode_without_key():
    ai = CompleteAI(seed_demo=False, enable_llm=True, llm=LLMChatBridge(enabled=True, api_key=None))
    out = ai.handle("Invent a poem about Mars", mode="llm")
    assert out["route"] == "llm"
    assert out["llm_at_runtime"] is False
    assert out["human_level_agi"] is False


def test_auto_prefers_llm_for_general_chat_when_available(tmp_path):
    """When LLM is available, open chat should not stay on symbolic fallback."""
    calls = []

    class FakeLLM(LLMChatBridge):
        def __init__(self):
            super().__init__(enabled=True, api_key="sk-fake")

        def chat(self, message, **kwargs):
            calls.append(message)
            return {
                "ok": True,
                "text": f"LLM:{message}",
                "model": "fake",
                "method": "llm_bridge",
            }

    ai = CompleteAI(
        seed_demo=False,
        llm=FakeLLM(),
        persist_path=str(tmp_path / "state.json"),
        autosave=False,
    )
    out = ai.handle("Write a haiku about rain")
    assert out["route"] == "llm_chat"
    assert out["llm_at_runtime"] is True
    assert out["text"].startswith("LLM:")
    assert calls[0].startswith("Write a haiku about rain")
    # Finderbay still preferred for business
    out2 = ai.handle("Show unpaid invoices for Project A")
    assert out2["route"] == "finderbay"
    assert out2["llm_at_runtime"] is False


def test_open_domain_name_and_identity():
    chat = OpenDomainChat()
    r = chat.respond("My name is Alex")
    assert "Alex" in r["text"]
    r2 = chat.respond("Are you Complete AI?")
    assert "Complete AI" in r2["text"]
