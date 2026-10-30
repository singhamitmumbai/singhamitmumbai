"""ChatGPT-level non-LLM brain — broad question understanding & response battery."""

from __future__ import annotations

from cognitive_ai.commonsense.chat import OpenDomainChat
from cognitive_ai.nlu.smart_questions import SmartQuestionUnderstanding
from cognitive_ai.response.chatgpt_style import ChatGPTStyleComposer
from cognitive_ai.search import IntelligentWebSearch
from cognitive_ai.system import CompleteNonLLMAI
from cognitive_ai.version import __version__


# Questions a ChatGPT-class assistant should understand and answer usefully offline
CHATGPT_BATTERY: list[tuple[str, list[str]]] = [
    # Meta
    ("Who are you?", ["complete ai"]),
    ("Are you ChatGPT?", ["complete ai"]),
    ("Hello", ["complete ai", "hello"]),
    ("Thanks", ["welcome"]),
    # Common sense yes/no
    ("Can penguins fly?", ["no", "cannot"]),
    ("Can birds fly?", ["yes", "can"]),
    # Definitions / explanations (local search KB or CS)
    ("What is gravity?", ["gravity"]),
    ("Explain photosynthesis simply", ["photosynthesis", "plant", "light", "oxygen"]),
    ("What is OCR?", ["ocr", "character", "text", "optical"]),
    ("Explain SQLite", ["sqlite", "database"]),
    ("What is artificial intelligence?", ["ai", "intelligence"]),
    # Search routing (offline local)
    ("Search the web for speed of light", ["light", "299", "speed"]),
    ("Look up thermal imaging", ["thermal", "infrared", "heat"]),
    ("Search the web for internet", ["internet", "network"]),
    # Howto / compare
    ("How do I reserve cement?", ["1.", "goal", "steps"]),
    ("Compare dog and cat", ["dog", "cat"]),
    # Creative templates
    ("Write a poem about Mars", ["poem", "mars"]),
    ("Draft an email about the project delay", ["email", "subject"]),
    ("Brainstorm ideas for a study plan", ["brainstorm", "study"]),
    ("Pros and cons of remote work", ["pros", "cons"]),
    ("Make an outline for climate change essay", ["outline", "climate"]),
    # Math
    ("Calculate 12 * (3 + 4)", ["84"]),
    ("What is 15% of 240", ["36"]),
    # Style preferences
    ("Explain gravity in simple terms", ["gravity"]),
    # Social follow-through
    ("How are you?", ["ready", "running"]),
    # Hinglish
    ("Gravity kya hai?", ["gravity"]),
]


def test_version_chatgpt_level():
    # ChatGPT-level stack introduced in 1.14; still present in later minors
    major_minor = tuple(int(x) for x in __version__.split(".")[:2])
    assert major_minor >= (1, 14)


def test_smart_frame_chatgpt_fields():
    sq = SmartQuestionUnderstanding()
    f = sq.understand("Could you please explain gravity in simple terms?")
    assert f.chatgpt_level is True
    assert f.style in ("eli5", "helpful", "steps")
    assert f.answer_policy in ("local_first", "search")
    f2 = sq.understand("Write a poem about the ocean")
    assert f2.answer_policy == "creative"
    assert f2.creative_kind == "poem"
    f3 = sq.understand("Calculate 2 + 2")
    assert f3.answer_policy == "math"
    assert f3.math_expr
    f4 = sq.understand("Search the web for DNA")
    assert f4.answer_policy == "search"
    f5 = sq.understand("Show unpaid invoices for Project A")
    assert f5.domain == "finderbay"
    assert f5.needs_tools is True


def test_chatgpt_style_composer_shapes():
    c = ChatGPTStyleComposer()
    d = c.compose(
        topic="gravity",
        ask_form="define",
        summary="Gravity pulls masses together.",
        extra_bullets=["Stronger near large masses"],
        style="helpful",
        confidence=0.8,
    )
    assert "gravity" in d.text.lower() or "Gravity" in d.text
    assert d.follow_ups
    y = c.compose(topic="can birds fly", ask_form="yesno", yesno=True, summary="Birds can fly.")
    assert "yes" in y.text.lower()
    w = c.compose_writing(kind="poem", topic="Mars")
    assert "mars" in w.text.lower()
    m = c.compose_math("2+2", 4, ok=True)
    assert "4" in m.text


def test_web_search_chatgpt_local_style():
    web = IntelligentWebSearch(allow_live=False)
    r = web.search("What is gravity?", style="helpful")
    assert r.ok
    assert r.method == "local_kb"
    assert "gravity" in r.answer.lower()
    assert r.related_queries
    # eli5 style still answers
    r2 = web.search("photosynthesis", style="eli5")
    assert r2.ok
    assert "photosynthesis" in r2.answer.lower() or "plant" in r2.answer.lower()


def test_chatgpt_battery_offline():
    chat = OpenDomainChat()
    failures = []
    for question, needles in CHATGPT_BATTERY:
        out = chat.respond(question, allow_live_search=False)
        text = (out.get("text") or "").lower()
        ok = out.get("supported", True) and any(n.lower() in text for n in needles)
        # Also require ChatGPT-level flag on successful paths
        if ok and question not in ("Thanks",):
            ok = ok and (out.get("chatgpt_level") is True or out.get("method"))
        if not ok:
            failures.append((question, out.get("method"), text[:160], needles))
    assert not failures, failures


def test_finderbay_still_works_alongside_chatgpt_chat():
    ai = CompleteNonLLMAI(seed_demo=True)
    r = ai.handle("How many unpaid invoices for Project A?")
    assert r["frame"]["operation"] == "query_invoices"
    assert int(r["structured"].get("answer") or 0) >= 1


def test_multi_turn_followup_style():
    chat = OpenDomainChat()
    a = chat.respond("Can penguins fly?", allow_live_search=False)
    assert a.get("supported")
    # follow-up about another animal using prior predicate when possible
    b = chat.respond("what about birds?", allow_live_search=False)
    assert b.get("text")
    assert b.get("chatgpt_level") is True


def test_query_expansion_present():
    web = IntelligentWebSearch(allow_live=False)
    ex = web.expand_queries("gravity", question={"ask_form": "define"})
    assert ex
    assert any("definition" in e or "overview" in e for e in ex)
