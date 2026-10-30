"""Vision see → analyze → understand → search loop."""

from __future__ import annotations

from cognitive_ai.assistant import FinderbayAssistant
from cognitive_ai.nlu.pipeline import LanguageUnderstanding
from cognitive_ai.system import CompleteNonLLMAI
from cognitive_ai.vision import VisionSystem
from cognitive_ai.vision.knowledge import ObjectKnowledge
from cognitive_ai.vision.search import ObjectSearch
from cognitive_ai.vision.understand import VisionUnderstanding


def test_object_knowledge_lookup_and_learn():
    kb = ObjectKnowledge()
    assert kb.known("car")
    fact = kb.lookup("automobile")  # alias
    assert fact and fact["label"] == "car"
    learned = kb.remember(
        "gantry",
        {"category": "equipment", "summary": "Overhead frame on site", "uses": ["signage"]},
        source="search",
    )
    assert learned["understood"]
    assert kb.known("gantry")


def test_search_encyclopedia_and_hypothesis():
    search = ObjectSearch()
    hit = search.search("traffic light")
    assert hit["found"] is True
    assert hit["method"] == "encyclopedia"
    assert search.knowledge.known("traffic light")
    miss = search.search("xyzzy_quantum_widget_99")
    assert miss["found"] is False
    assert miss["method"] == "hypothesis"


def test_see_analyze_understand_pipeline():
    vu = VisionUnderstanding()
    # stub path hints vehicles + people
    out = vu.perceive("traffic_cars_and_crowd_people.jpg")
    assert out["task"] == "understand_scene"
    assert out["pipeline"][0] == "modality_detect"
    assert "see" in out["pipeline"] and "understand" in out["pipeline"]
    assert out["count"] >= 1
    assert out["see"]["stage"] == "see"
    assert out["analyze"]["stage"] == "analyze"
    assert out["understand"]["stage"] == "understand"
    assert out["understand"]["understood_count"] >= 1
    assert "I see" in out["text"] or "Understood" in out["text"]
    assert out["llm"] is False


def test_unknown_object_triggers_search():
    vu = VisionUnderstanding()
    # Generic stub emits label "object" for unknown names — may need search
    out = vu.perceive("strange_widget_device.jpg")
    assert out["ok"] is True
    # explain unknown-ish construction term via search
    ex = vu.explain_object("rebar")
    assert ex["understood"] is True
    assert ex["fact"]["category"] in ("material", "learned")


def test_vision_system_understand_api():
    vs = VisionSystem()
    st = vs.status()
    assert "understanding" in st
    assert st["understanding"]["pipeline"][0] == "modality_detect"
    assert "thermal" in st["understanding"]["modalities"]
    result = vs.understand("crowd_people.jpg")
    assert result["task"] == "understand_scene"
    assert (result.get("understand") or {}).get("understood_count", 0) >= 1


def test_nlu_understand_scene_intent():
    nlu = LanguageUnderstanding()
    frame = nlu.parse("Understand the scene in image ./traffic_cars.jpg")
    assert frame.operation == "understand_scene"
    assert frame.supported is True
    cond = {c["field"]: c["value"] for c in frame.conditions}
    assert cond.get("image_path") == "./traffic_cars.jpg"


def test_end_to_end_understand_via_ai():
    ai = CompleteNonLLMAI(seed_demo=True)
    out = ai.handle("Understand the scene in image ./crowd_people.jpg")
    assert out["frame"]["operation"] == "understand_scene"
    assert out["faithful"] is True
    assert out["human_level_agi"] is False
    text = out["text"].lower()
    assert "understood" in text or "see" in text or "person" in text


def test_assistant_understand_image():
    a = FinderbayAssistant(seed_demo=True, enable_llm=False)
    out = a.understand_image("./traffic_cars.jpg")
    assert out["route"] == "vision_understand"
    assert out["structured"]["task"] == "understand_scene"
    out2 = a.handle("understand image ./crowd_people.jpg")
    assert out2["route"] == "vision_understand"
