"""Catalog of vendored YOLO / CV projects from Nawaf-Rayhan585/YOLO_Projects."""

from __future__ import annotations

from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2] / "third_party" / "YOLO_Projects"

# Map AI task names → project folder + capability description
PROJECTS: dict[str, dict[str, Any]] = {
    "vehicles": {
        "folder": "vehicle-detection-tracking",
        "script": "vehicle_tracker.py",
        "description": "Detect/track cars, motorcycles, buses, trucks (YOLO + ByteTrack)",
        "task": "detect_vehicles",
        "needs_weights": "yolo11n.pt",
    },
    "people_count": {
        "folder": "people-counter-in-out",
        "script": "people_counter.py",
        "description": "IN/OUT people counter",
        "task": "count_people",
        "needs_weights": "yolo11n.pt",
    },
    "crowd_heatmap": {
        "folder": "crowd-heatmap",
        "script": "crowd_heatmap.py",
        "description": "Crowd density heatmap + person count",
        "task": "crowd_heatmap",
        "needs_weights": "yolo11n.pt",
    },
    "queue": {
        "folder": "queue-wait-time-estimator",
        "script": "queue_estimator.py",
        "description": "Queue length + wait time estimate",
        "task": "estimate_queue",
        "needs_weights": "yolo11n.pt",
    },
    "fire": {
        "folder": "fire-smoke-detection",
        "script": None,
        "description": "Fire/smoke detection (custom-trained weights)",
        "task": "detect_fire",
        "needs_weights": "best.pt",
        "train_notebook": "train_fire_detector.ipynb",
    },
    "ppe": {
        "folder": "ppe-safety-compliance",
        "script": "ppe_compliance.py",
        "description": "PPE compliance (hard hat / vest / mask)",
        "task": "ppe_compliance",
        "needs_weights": "best.pt",
    },
    "fall": {
        "folder": "fall-detection",
        "script": "fall_detector.py",
        "description": "Pose-based fall detection",
        "task": "detect_fall",
        "needs_weights": "yolo11n-pose.pt",
    },
    "intrusion": {
        "folder": "restricted-zone-intrusion-alert",
        "script": "intrusion_alert.py",
        "description": "Restricted-zone intrusion alert",
        "task": "detect_intrusion",
        "needs_weights": "yolo11n.pt",
    },
    "parking": {
        "folder": "parking-space-occupancy",
        "script": "parking_occupancy.py",
        "description": "Parking slot free/occupied",
        "task": "parking_occupancy",
        "needs_weights": "yolo11n.pt",
    },
    "plates": {
        "folder": "license-plate-recognition",
        "script": "anpr_inference.py",
        "description": "License plate detection + OCR",
        "task": "read_plates",
        "needs_weights": "best.pt",
    },
    "document_scan": {
        "folder": "document-scanner-ocr",
        "script": "document_scanner.py",
        "description": "Classical CV document scan + OCR (no YOLO required)",
        "task": "scan_document",
        "needs_weights": None,
    },
    "face_blur": {
        "folder": "face-blur-anonymizer",
        "script": "face_anonymizer.py",
        "description": "Blur faces for privacy",
        "task": "blur_faces",
        "needs_weights": "yolo11n.pt",
    },
    "objects": {
        "folder": None,
        "script": None,
        "description": "General COCO object detection via Ultralytics YOLO",
        "task": "detect_objects",
        "needs_weights": "yolo11n.pt",
    },
}


def list_projects() -> dict[str, dict[str, Any]]:
    out = {}
    for key, meta in PROJECTS.items():
        folder = meta.get("folder")
        path = (ROOT / folder) if folder else None
        out[key] = {
            **meta,
            "installed": bool(path and path.exists()) if folder else True,
            "path": str(path) if path else None,
            "vendor_root": str(ROOT),
            "provenance": "https://github.com/Nawaf-Rayhan585/YOLO_Projects (MIT)",
        }
    return out


def project_path(key: str) -> Path | None:
    meta = PROJECTS.get(key)
    if not meta or not meta.get("folder"):
        return None
    return ROOT / meta["folder"]
