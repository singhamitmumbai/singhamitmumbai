"""Computer vision integration — YOLO_Projects adapters (stub without ultralytics)."""

from __future__ import annotations

from pathlib import Path

from cognitive_ai.nlu.pipeline import LanguageUnderstanding
from cognitive_ai.system import CompleteNonLLMAI
from cognitive_ai.vision import VisionSystem, list_projects
from cognitive_ai.vision.adapters import VisionAdapters
from cognitive_ai.vision.yolo_engine import YOLOEngine


def test_vendor_projects_present():
    projects = list_projects()
    assert "vehicles" in projects
    assert "people_count" in projects
    assert "document_scan" in projects
    assert "objects" in projects
    assert projects["vehicles"]["installed"] is True
    assert "YOLO_Projects" in projects["vehicles"]["provenance"]
    root = Path(projects["vehicles"]["vendor_root"])
    assert (root / "LICENSE").exists()
    assert (root / "ATTRIBUTION.md").exists()


def test_yolo_stub_vehicles_and_people():
    eng = YOLOEngine()
    assert eng.engine in ("stub", "ultralytics")
    cars = eng.detect("traffic_cars.jpg", classes=["car", "truck", "bus", "motorcycle", "bicycle"])
    assert cars["count"] >= 1
    assert "car" in cars["counts_by_label"] or cars["engine"] == "ultralytics"
    people = eng.detect("crowd_people.png", classes=["person"])
    assert people["count"] >= 1 or people["engine"] == "ultralytics"


def test_adapters_status_and_tasks():
    ad = VisionAdapters(YOLOEngine())
    st = ad.status()
    assert st["llm"] is False
    assert st["task"] == "vision_status"
    assert "vehicles" in st["projects"]
    out = ad.detect_vehicles("vehicle_truck.jpg")
    assert out["task"] == "detect_vehicles"
    assert out["vehicle_count"] >= 0
    fire = ad.detect_fire("fire_smoke.jpg")
    assert fire["task"] == "detect_fire"
    assert "alert" in fire


def test_nlu_vision_intents():
    nlu = LanguageUnderstanding()
    cases = [
        ("Detect vehicles in image ./traffic_cars.jpg", "detect_vehicles"),
        ("Count people in image ./crowd_people.jpg", "count_people"),
        ("Detect fire in image ./fire_smoke.jpg", "detect_fire"),
        ("Vision status", "vision_status"),
        ("Detect objects in image ./scene.jpg", "detect_objects"),
        ("PPE compliance in image ./site.jpg", "ppe_compliance"),
        ("Estimate queue in image ./queue_people.jpg", "estimate_queue"),
    ]
    for text, op in cases:
        frame = nlu.parse(text)
        assert frame.operation == op, (text, frame.operation, frame.intent_scores)
        assert frame.supported is True
        if op != "vision_status":
            cond = {c["field"]: c["value"] for c in frame.conditions}
            assert "image_path" in cond


def test_end_to_end_vision_via_ai():
    ai = CompleteNonLLMAI(seed_demo=True)
    st = ai.handle("Vision status")
    assert st["frame"]["operation"] == "vision_status"
    assert st["faithful"] is True
    assert "vision" in st["text"].lower() or "YOLO" in st["text"]
    assert st["llm_dependency"] is False
    assert st["human_level_agi"] is False

    cars = ai.handle("Detect vehicles in image ./traffic_cars.jpg")
    assert cars["frame"]["operation"] == "detect_vehicles"
    assert cars["structured"].get("count", 0) >= 1 or cars["structured"].get("engine") == "ultralytics"
    assert "vehicle" in cars["text"].lower() or "Detected" in cars["text"]

    people = ai.handle("Count people in image ./crowd_people.jpg")
    assert people["frame"]["operation"] == "count_people"
    assert people["structured"].get("count", 0) >= 1 or people["structured"].get("engine") == "ultralytics"

    fire = ai.handle("Detect fire in image ./fire_smoke.jpg")
    assert fire["frame"]["operation"] == "detect_fire"
    assert "alert" in fire["structured"]


def test_vision_system_analyze_records_observation():
    from cognitive_ai.core.working_state import WorkingState

    vs = VisionSystem()
    state = WorkingState()
    out = vs.analyze("crowd_people.jpg", task="count_people", state=state)
    assert out["task"] == "count_people"
    assert state.observations
    assert state.observations[-1].modality == "vision"
    assert out.get("observation_id")


def test_ai_status_includes_vision():
    ai = CompleteNonLLMAI(seed_demo=False)
    s = ai.status()
    assert "vision" in s
    assert s["vision"]["llm"] is False
    assert s["version"].startswith("1.2")
