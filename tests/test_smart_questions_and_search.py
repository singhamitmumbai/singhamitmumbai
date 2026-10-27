"""Smarter non-LLM question understanding + intelligent search — broad battery."""

from __future__ import annotations

from cognitive_ai.assistant import FinderbayAssistant
from cognitive_ai.commonsense.chat import OpenDomainChat
from cognitive_ai.jarvis import JarvisPipeline
from cognitive_ai.nlu.questions import QuestionAnalyzer
from cognitive_ai.nlu.smart_questions import SmartQuestionUnderstanding
from cognitive_ai.search import IntelligentWebSearch
from cognitive_ai.system import CompleteNonLLMAI
from cognitive_ai.general.agent import GeneralNonLLMAI


# Representative questions the non-LLM brain should understand/respond to
QUESTION_BATTERY = [
    # Meta / social
    ("Who are you?", ["complete ai", "finderbay"]),
    ("Hello", ["hello", "complete ai"]),
    ("How are you?", ["ready", "running"]),
    # Yes/no common sense
    ("Can penguins fly?", ["cannot", "no"]),
    # Define / explain via local search KB
    ("What is gravity?", ["gravity", "mass", "attract"]),
    ("Explain photosynthesis", ["photosynthesis", "light", "plant", "oxygen"]),
    ("Search the web for speed of light", ["light", "299", "km", "speed"]),
    ("Look up thermal imaging", ["thermal", "infrared", "heat"]),
    # Howto
    ("How do I reserve cement?", ["steps", "goal", "check"]),
    # Compare
    ("Compare dog and cat", ["dog", "cat"]),
    # Finderbay
    ("How many unpaid invoices for Project A?", None),  # numeric / invoice domain
    ("Show unpaid invoices for Project A", ["invoice"]),
    # Vision-ish wording (routing)
    ("Vision status", ["vision", "yolo", "engine"]),
]


def test_question_analyzer_smarter_forms():
    qa = QuestionAnalyzer()
    a = qa.analyze("Please tell me what gravity is?")
    assert a.is_question
    assert a.wh_type in ("what", "none") or a.ask_form in ("define", "unknown")
    b = qa.analyze("Search the web for photosynthesis")
    assert b.ask_form == "search" or "search" in b.ask_form
    c = qa.analyze("Kyun fire smoke cause karta hai?")
    # hindi-ish why may map causal/why
    assert c.is_question or c.wh_type in ("why", "none")


def test_smart_question_frame_policies():
    sq = SmartQuestionUnderstanding()
    f1 = sq.understand("Could you please tell me what gravity is?")
    assert f1.normalized
    assert f1.domain in ("science", "general", "web")
    assert f1.answer_policy in ("search", "local_first")
    f2 = sq.understand("Show unpaid invoices for Project A")
    assert f2.domain == "finderbay"
    assert f2.needs_tools is True
    f3 = sq.understand("Search the web for SQLite")
    assert f3.needs_search is True
    assert f3.answer_policy == "search"


def test_intelligent_web_search_local_first():
    web = IntelligentWebSearch(allow_live=False)
    r = web.search("What is gravity?")
    assert r.ok
    assert r.method == "local_kb"
    assert "gravity" in r.answer.lower() or "mass" in r.answer.lower()
    assert r.hits and r.hits[0].source == "local_kb"
    assert r.rewritten_query
    r2 = web.search("look up photosynthesis")
    assert r2.ok
    assert "photosynthesis" in r2.answer.lower() or "plant" in r2.answer.lower()


def test_intelligent_web_search_rewrite():
    web = IntelligentWebSearch(allow_live=False)
    q = web.rewrite_query(
        "Please search the web for artificial intelligence",
        question={"ask_form": "search", "focus": ["artificial intelligence"]},
    )
    assert "artificial intelligence" in q.lower()
    assert "please" not in q.lower()


def test_open_domain_answers_battery_offline():
    chat = OpenDomainChat()
    failures = []
    for question, needles in QUESTION_BATTERY:
        if needles is None:
            continue  # Finderbay handled elsewhere
        if "invoice" in question.lower() or question.lower().startswith("vision"):
            continue
        out = chat.respond(question, allow_live_search=False)
        text = (out.get("text") or "").lower()
        ok = out.get("supported", True) and any(n in text for n in needles)
        if not ok:
            failures.append((question, out.get("method"), text[:120], needles))
    assert not failures, failures


def test_finderbay_and_general_questions():
    ai = CompleteNonLLMAI(seed_demo=True)
    r = ai.handle("How many unpaid invoices for Project A?")
    assert r["frame"]["operation"] == "query_invoices"
    assert r["structured"].get("answer_kind") == "count"
    assert int(r["structured"].get("answer") or 0) >= 1

    g = GeneralNonLLMAI(seed_demo=True)
    out = g.handle("Can penguins fly?")
    text = (out.get("text") or "").lower()
    assert "cannot" in text or "no" in text


def test_assistant_web_search_route():
    a = FinderbayAssistant(seed_demo=True, enable_llm=False)
    out = a.handle("Search the web for gravity")
    assert out["route"] in ("web_search", "jarvis")
    text = (out.get("text") or "").lower()
    assert "gravity" in text or "mass" in text or "attract" in text


def test_jarvis_web_search_tool():
    pipe = JarvisPipeline()
    r = pipe.run("Search the web for photosynthesis")
    assert r.intent == "web.search"
    assert r.ok or "photosynthesis" in r.response.lower() or "plant" in r.response.lower()
    # Direct tool
    hit = pipe.tools.invoke("web.search", {"query": "what is sqlite"})
    assert hit.get("ok")
    assert "sqlite" in (hit.get("answer") or "").lower()


def test_full_question_matrix_response_shapes():
    """Every listed question yields a non-empty response with metadata."""
    chat = OpenDomainChat()
    ai = CompleteNonLLMAI(seed_demo=True)
    questions = [
        "Who are you?",
        "Can birds fly?",
        "What is gravity?",
        "Why can fire cause smoke?",
        "Explain OCR",
        "How to make tea?",
        "Compare car and truck",
        "Search the web for internet",
        "How many unpaid invoices for Project A?",
        "Show alerts for Project A",
        "dog:bark :: cat:?",
        "Thanks",
    ]
    for q in questions:
        if "invoice" in q.lower() or "alert" in q.lower():
            out = ai.handle(q)
            assert out.get("text"), q
            assert out.get("human_level_agi") is False
        else:
            out = chat.respond(q, allow_live_search=False)
            assert out.get("text"), q
            assert out.get("question") is not None
            assert out.get("human_level_agi") is False
