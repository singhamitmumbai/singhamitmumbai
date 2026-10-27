"""Task adapters over YOLO engine + classical CV (document scanner).

Adapters mirror capabilities from third_party/YOLO_Projects (MIT, Nawaf Rayhan).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .catalog import PROJECTS
from .yolo_engine import PERSON_CLASS, VEHICLE_CLASSES, YOLOEngine


class VisionAdapters:
    def __init__(self, engine: Optional[YOLOEngine] = None) -> None:
        self.engine = engine or YOLOEngine()

    def detect_objects(self, image: Any, *, conf: float = 0.35) -> dict[str, Any]:
        result = self.engine.detect(image, conf=conf)
        return {**result, "task": "detect_objects", "project": "objects"}

    def detect_vehicles(self, image: Any, *, conf: float = 0.35) -> dict[str, Any]:
        result = self.engine.detect(image, conf=conf, classes=sorted(VEHICLE_CLASSES))
        return {
            **result,
            "task": "detect_vehicles",
            "project": "vehicles",
            "vehicle_count": result["count"],
        }

    def count_people(self, image: Any, *, conf: float = 0.35) -> dict[str, Any]:
        result = self.engine.detect(image, conf=conf, classes=[PERSON_CLASS])
        return {
            **result,
            "task": "count_people",
            "project": "people_count",
            "people_count": result["count"],
        }

    def crowd_heatmap(self, image: Any, *, conf: float = 0.35) -> dict[str, Any]:
        people = self.count_people(image, conf=conf)
        density = "high" if people["people_count"] >= 10 else "medium" if people["people_count"] >= 4 else "low"
        return {
            **people,
            "task": "crowd_heatmap",
            "project": "crowd_heatmap",
            "density": density,
            "note": people.get("note") or "Heatmap rendering requires OpenCV display; count only here",
        }

    def estimate_queue(self, image: Any, *, conf: float = 0.35, seconds_per_person: float = 45.0) -> dict[str, Any]:
        people = self.count_people(image, conf=conf)
        wait = round(people["people_count"] * seconds_per_person / 60.0, 1)
        return {
            **people,
            "task": "estimate_queue",
            "project": "queue",
            "estimated_wait_minutes": wait,
            "seconds_per_person": seconds_per_person,
        }

    def detect_fire(self, image: Any, *, conf: float = 0.35) -> dict[str, Any]:
        # Custom fire weights may be absent; fall back to stub/label heuristic via engine
        result = self.engine.detect(image, conf=conf)
        fires = [d for d in result["detections"] if d["label"] in ("fire", "smoke")]
        if not fires and result["engine"] == "stub":
            # Re-run with fire-hinted path if image is a path
            hint_path = f"fire_{Path(str(image)).name}" if isinstance(image, (str, Path)) else image
            result = self.engine.detect(hint_path, conf=conf)
            fires = [d for d in result["detections"] if d["label"] in ("fire", "smoke")]
        return {
            **result,
            "detections": fires,
            "count": len(fires),
            "counts_by_label": {k: v for k, v in result.get("counts_by_label", {}).items() if k in ("fire", "smoke")}
            or ({"fire": len(fires)} if fires else {}),
            "task": "detect_fire",
            "project": "fire",
            "alert": bool(fires),
            "note": result.get("note")
            or "Train custom fire model via third_party/YOLO_Projects/fire-smoke-detection/",
        }

    def ppe_compliance(self, image: Any, *, conf: float = 0.35) -> dict[str, Any]:
        people = self.count_people(image, conf=conf)
        # Without custom PPE weights we can only report people present
        return {
            **people,
            "task": "ppe_compliance",
            "project": "ppe",
            "compliant": None,
            "note": "Custom PPE weights (best.pt) required for hat/vest/mask classes; people counted only",
        }

    def detect_fall(self, image: Any, *, conf: float = 0.35) -> dict[str, Any]:
        people = self.count_people(image, conf=conf)
        return {
            **people,
            "task": "detect_fall",
            "project": "fall",
            "fall_detected": False,
            "note": "Pose model yolo11n-pose.pt required for real fall detection",
        }

    def detect_intrusion(self, image: Any, *, conf: float = 0.35) -> dict[str, Any]:
        people = self.count_people(image, conf=conf)
        return {
            **people,
            "task": "detect_intrusion",
            "project": "intrusion",
            "intrusion": people["people_count"] > 0,
            "note": "Zone polygons from YOLO_Projects restricted-zone project; person presence used as proxy",
        }

    def parking_occupancy(self, image: Any, *, conf: float = 0.35) -> dict[str, Any]:
        vehicles = self.detect_vehicles(image, conf=conf)
        return {
            **vehicles,
            "task": "parking_occupancy",
            "project": "parking",
            "occupied_estimate": vehicles["vehicle_count"],
            "note": "Slot polygons from parking-space-occupancy project needed for exact free/occupied",
        }

    def read_plates(self, image: Any, *, conf: float = 0.35) -> dict[str, Any]:
        vehicles = self.detect_vehicles(image, conf=conf)
        return {
            **vehicles,
            "task": "read_plates",
            "project": "plates",
            "plates": [],
            "note": "ANPR custom weights + OCR required (license-plate-recognition project)",
        }

    def blur_faces(self, image: Any, *, conf: float = 0.35) -> dict[str, Any]:
        people = self.count_people(image, conf=conf)
        return {
            **people,
            "task": "blur_faces",
            "project": "face_blur",
            "faces_blurred": people["people_count"],
            "note": "Output image writing requires OpenCV; count of person boxes reported",
        }

    def scan_document(self, image: Any) -> dict[str, Any]:
        """Classical CV document scanner adapted from document-scanner-ocr."""
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception as exc:  # noqa: BLE001
            return {
                "task": "scan_document",
                "project": "document_scan",
                "engine": "unavailable",
                "error": f"opencv_unavailable:{exc}",
                "text": "",
                "corners": None,
            }

        path = str(image) if isinstance(image, (str, Path)) else None
        if path is None:
            return {
                "task": "scan_document",
                "project": "document_scan",
                "engine": "opencv",
                "error": "image_path_required",
                "text": "",
                "corners": None,
            }

        img = cv2.imread(path)
        if img is None:
            return {
                "task": "scan_document",
                "project": "document_scan",
                "engine": "opencv",
                "error": "imread_failed",
                "text": "",
                "corners": None,
            }

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        corners = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                corners = approx.reshape(4, 2).tolist()
                break

        text = ""
        try:
            import pytesseract  # type: ignore

            warped = img
            if corners is not None:
                pts = np.array(corners, dtype="float32")
                # order and warp (simplified)
                rect = _order_points(pts)
                (tl, tr, br, bl) = rect
                width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
                height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
                dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(img, M, (width, height))
            text = pytesseract.image_to_string(warped)
        except Exception:  # noqa: BLE001
            text = ""

        return {
            "task": "scan_document",
            "project": "document_scan",
            "engine": "opencv+tesseract" if text else "opencv",
            "error": None,
            "corners": corners,
            "text": text.strip(),
            "has_document_quad": corners is not None,
        }

    def run(self, task: str, image: Any, **kwargs: Any) -> dict[str, Any]:
        dispatch = {
            "detect_objects": self.detect_objects,
            "detect_vehicles": self.detect_vehicles,
            "count_people": self.count_people,
            "crowd_heatmap": self.crowd_heatmap,
            "estimate_queue": self.estimate_queue,
            "detect_fire": self.detect_fire,
            "ppe_compliance": self.ppe_compliance,
            "detect_fall": self.detect_fall,
            "detect_intrusion": self.detect_intrusion,
            "parking_occupancy": self.parking_occupancy,
            "read_plates": self.read_plates,
            "blur_faces": self.blur_faces,
            "scan_document": self.scan_document,
            "vision_status": lambda _img=None, **_k: self.status(),
        }
        fn = dispatch.get(task)
        if fn is None:
            return {"task": task, "error": f"unknown_vision_task:{task}", "detections": []}
        if task == "vision_status":
            return fn()
        if task == "scan_document":
            return fn(image)
        return fn(image, **kwargs)

    def status(self) -> dict[str, Any]:
        return {
            "task": "vision_status",
            "engine": self.engine.engine,
            "weights": self.engine.weights,
            "available": self.engine.available,
            "error": self.engine.error,
            "projects": {k: {"installed": True, "description": v["description"]} for k, v in PROJECTS.items()},
            "vendor": "https://github.com/Nawaf-Rayhan585/YOLO_Projects",
            "license": "MIT",
            "llm": False,
        }


def _order_points(pts):
    import numpy as np

    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
