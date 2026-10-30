"""Advanced question understanding (non-LLM)."""

from cognitive_ai.nlu.questions import QuestionAnalyzer, analyze_question
from cognitive_ai.nlu.pipeline import LanguageUnderstanding
from cognitive_ai.system import CompleteNonLLMAI
from cognitive_ai.general.agent import GeneralNonLLMAI
from cognitive_ai.commonsense.chat import OpenDomainChat


def test_wh_and_ask_form_classification():
    qa = QuestionAnalyzer()
    a = qa.analyze("How many unpaid invoices are there for Project A?")
    assert a.is_question
    assert a.wh_type == "how_many"
    assert a.ask_form == "count"
    assert a.aggregation == "count"
    assert a.expected_answer == "number"
    assert any("invoice" in f for f in a.focus)

    b = qa.analyze("Why can fire cause smoke?")
    assert b.wh_type == "why"
    assert b.ask_form == "causal"

    c = qa.analyze("Can penguins fly?")
    assert c.wh_type == "yesno"
    assert c.ask_form == "yesno"
    assert c.expected_answer == "boolean"


def test_multi_clause_split():
    a = analyze_question("How many unpaid invoices for Project A, and which are overdue?")
    assert a.is_multi_clause
    assert len(a.clauses) >= 2


def test_nlu_attaches_question_analysis():
    nlu = LanguageUnderstanding()
    frame = nlu.parse("How many unpaid invoices for Project A?")
    assert frame.operation == "query_invoices"
    assert frame.intent_type == "question"
    assert frame.question is not None
    assert frame.question["ask_form"] == "count"
    assert any(c["field"] == "aggregation" and c["value"] == "count" for c in frame.conditions)


def test_count_question_gets_numeric_answer():
    ai = CompleteNonLLMAI(seed_demo=True)
    r = ai.handle("How many unpaid invoices for Project A?")
    assert r["frame"]["operation"] == "query_invoices"
    assert r["structured"].get("answer_kind") == "count"
    assert isinstance(r["structured"].get("answer"), int)
    assert r["structured"]["answer"] >= 1
    # short-ish answer
    assert r["text"].rstrip(".").replace(" ", "").isalnum() or r["structured"]["answer"] >= 1


def test_followup_what_about_project():
    ai = CompleteNonLLMAI(seed_demo=True)
    r1 = ai.handle("Show unpaid invoices for Project A")
    assert r1["frame"]["operation"] == "query_invoices"
    r2 = ai.handle("What about Project B?")
    assert r2["frame"]["operation"] == "query_invoices"
    assert r2["frame"]["project_id"] == "project_b"


def test_followup_how_many_of_those():
    ai = CompleteNonLLMAI(seed_demo=True)
    ai.handle("Show unpaid invoices for Project A")
    r = ai.handle("How many of those?")
    assert r["frame"]["operation"] == "query_invoices"
    assert any(c.get("field") == "aggregation" for c in r["frame"]["conditions"])
    assert r["structured"].get("answer_kind") == "count"


def test_open_domain_uses_question_meta():
    chat = OpenDomainChat()
    out = chat.respond("Can penguins fly?")
    assert out["question"]["wh_type"] == "yesno"
    assert "cannot" in out["text"].lower() or "no" in out["text"].lower()


def test_open_domain_where_and_define():
    chat = OpenDomainChat()
    w = chat.respond("Where is Mumbai?")
    assert w["supported"]
    assert "mumbai" in w["text"].lower() or "india" in w["text"].lower()
    d = chat.respond("What is a knife?")
    assert "knife" in d["text"].lower()


def test_general_routes_count_question_to_finderbay():
    ai = GeneralNonLLMAI(seed_demo=True)
    r = ai.handle("How many unpaid invoices for Project A?")
    assert r["route"] == "finderbay"
    assert r["frame"]["operation"] == "query_invoices"


def test_human_level_agi_still_false():
    assert GeneralNonLLMAI(seed_demo=True).status()["human_level_agi"] is False
