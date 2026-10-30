"""Complete AI v1.5 — tools, session, diagnostics."""

from __future__ import annotations

from cognitive_ai.chat.llm_bridge import LLMChatBridge
from cognitive_ai.chat.tools import extract_tool_calls, strip_tool_lines
from cognitive_ai.complete_ai import CompleteAI


def test_tool_line_parsing():
    text = (
        'Here is a plan.\n'
        'TOOL finderbay {"text": "Show unpaid invoices for Project A"}\n'
        'TOOL vision {"task": "count_people", "image": "./crowd_people.jpg"}\n'
        "Thanks."
    )
    calls = extract_tool_calls(text)
    assert len(calls) == 2
    assert calls[0]["name"] == "finderbay"
    assert "Project A" in calls[0]["args"]["text"]
    assert calls[1]["name"] == "vision"
    cleaned = strip_tool_lines(text)
    assert "TOOL" not in cleaned
    assert "Thanks" in cleaned


def test_tool_router_local_engines():
    ai = CompleteAI(seed_demo=True, enable_llm=False)
    st = ai.tools.run("status", {})
    assert st["ok"] is True
    assert st["name"] == "Complete AI"
    fb = ai.tools.run("finderbay", {"text": "status"})
    assert fb["ok"] is True
    vis = ai.tools.run("vision", {"task": "count_people", "image": "./crowd_people.jpg"})
    assert vis["ok"] is True
    assert vis.get("people_count", vis.get("count", 0)) >= 1
    mem = ai.tools.run("memory", {"action": "remember", "key": "lang", "value": "en"})
    assert mem["ok"] is True
    assert ai.recall("lang") == "en"


def test_session_memory_and_diagnostics():
    ai = CompleteAI(seed_demo=True, enable_llm=False)
    ai.handle("Who are you?")
    ai.handle("Show unpaid invoices for Project A")
    assert len(ai.session) >= 4  # user+assistant × 2
    assert ai.session[0].role == "user"
    diag = ai.diagnostics()
    assert diag["ok"] is True
    assert diag["passed"] == diag["total"]
    assert diag["version"].startswith("1.")
    ai.reset_session()
    assert ai.session == []


def test_llm_tool_loop_with_fake_model():
    class FakeLLM(LLMChatBridge):
        def __init__(self):
            super().__init__(enabled=True, api_key="sk-fake")
            self.n = 0

        def chat(self, message, **kwargs):
            self.n += 1
            if self.n == 1:
                return {
                    "ok": True,
                    "text": 'TOOL finderbay {"text": "status"}\n',
                    "model": "fake",
                    "method": "llm_bridge",
                }
            return {
                "ok": True,
                "text": "Status checked via local Finderbay tool.",
                "model": "fake",
                "method": "llm_bridge",
            }

    ai = CompleteAI(seed_demo=True, llm=FakeLLM())
    out = ai.handle("Please check system status using tools")
    assert out["route"] == "llm_tools"
    assert out["llm_at_runtime"] is True
    assert out["tools"]
    assert out["tools"][0]["call"]["name"] == "finderbay"
    assert "Status checked" in out["text"]


def test_api_complete_mode_health():
    from cognitive_ai.api.server import APIState

    state = APIState(mode="complete")
    assert state.complete is not None
    assert state.complete.status()["name"] == "Complete AI"
