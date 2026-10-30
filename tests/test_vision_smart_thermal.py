"""Smart vision + thermal / multi-modality tests."""

from __future__ import annotations

from cognitive_ai.assistant import FinderbayAssistant
from cognitive_ai.nlu.pipeline import LanguageUnderstanding
from cognitive_ai.vision import VisionSystem, detect_modality
from cognitive_ai.vision.smart import SmartVisionReasoner
from cognitive_ai.vision.thermal import ThermalAnalyzer


def test_modality_detection_cues():
    assert detect_modality("site_thermal_flir.jpg")["modality"] == "thermal"
    assert detect_modality("panel_infrared_ir_image.png")["modality"] == "infrared"
    assert detect_modality("gate_night_vision.jpg")["modality"] == "night_vision"
    assert detect_modality("chest_xray.dcm")["modality"] == "xray"
    assert detect_modality("depth_map.png")["modality"] == "depth"
    assert detect_modality("invoice_scan.jpg")["modality"] == "document"
    assert detect_modality("ordinary_photo.jpg")["modality"] == "rgb"


def test_thermal_analyzer_stub():
    th = ThermalAnalyzer()
    out = th.analyze("motor_panel_thermal_overheat.jpg")
    assert out["task"] == "thermal_analyze"
    assert out["ok"] is True
    assert out["calibrated_celsius"] is False
    assert out["count"] >= 1
    assert "hotspot" in (out.get("summary") or "").lower() or out.get("hotspots")


def test_smart_reasoner_scene_and_risk():
    smart = SmartVisionReasoner()
    r = smart.reason(
        detections=[{"label": "fire"}, {"label": "person"}],
        understood=[],
        unknown=[],
        modality={"modality": "rgb", "confidence": 0.9},
    )
    assert r["scene_type"] in ("hazard_fire", "crowd", "general")
    assert r["max_risk"] in ("critical", "high")
    assert r["smart_summary"]


def test_understand_thermal_image_end_to_end():
    vs = VisionSystem()
    out = vs.understand("warehouse_thermal_flir_person.jpg")
    assert out["task"] == "understand_scene"
    assert out["modality"]["modality"] == "thermal"
    assert out["thermal"] is not None
    assert out["smart"]["scene_type"]
    assert "thermal" in (out.get("text") or "").lower() or out["smart"]["modality"] == "thermal"
    # hotspots / person_thermal should be understood via KB
    assert (out.get("understand") or {}).get("understood_count", 0) >= 1


def test_thermal_analyze_task():
    vs = VisionSystem()
    out = vs.analyze_thermal("electrical_panel_thermal.jpg")
    assert out["task"] == "thermal_analyze"
    assert out.get("alert") in (True, False)
    st = vs.status()
    assert st["understanding"]["thermal"] is True
    assert "thermal" in st["understanding"]["modalities"]


def test_nlu_thermal_intent():
    nlu = LanguageUnderstanding()
    frame = nlu.parse("Analyze thermal image ./panel_thermal.jpg")
    assert frame.operation in ("thermal_analyze", "understand_scene")
    cond = {c["field"]: c["value"] for c in frame.conditions}
    assert cond.get("image_path") == "./panel_thermal.jpg"


def test_assistant_thermal_and_night():
    a = FinderbayAssistant(seed_demo=True, enable_llm=False)
    t = a.analyze_thermal("./motor_thermal.jpg")
    assert t["route"] == "vision_thermal"
    u = a.understand_image("./gate_night_vision_people.jpg")
    assert u["route"] == "vision_understand"
    assert u.get("modality") == "night_vision" or (
        isinstance(u.get("structured", {}).get("modality"), dict)
        and u["structured"]["modality"]["modality"] == "night_vision"
    )
