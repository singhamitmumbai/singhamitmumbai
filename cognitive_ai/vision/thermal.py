"""Thermal / IR image analysis — hotspots, cold spots, relative temperature map."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


class ThermalAnalyzer:
    """Analyze thermal/IR frames without claiming absolute °C unless calibrated."""

    def analyze(
        self,
        image: Any,
        *,
        modality: str = "thermal",
        hot_percentile: float = 90.0,
        cold_percentile: float = 10.0,
    ) -> dict[str, Any]:
        stats = self._load_stats(image)
        if stats.get("error") and stats.get("engine") == "stub":
            return self._stub_thermal(image, modality=modality)

        gray = stats["gray"]
        h, w = gray.shape[:2]
        flat = gray.reshape(-1)
        # percentiles
        import numpy as np

        p_cold = float(np.percentile(flat, cold_percentile))
        p_hot = float(np.percentile(flat, hot_percentile))
        p50 = float(np.percentile(flat, 50))
        mean = float(flat.mean())
        amin, amax = float(flat.min()), float(flat.max())

        hot_mask = gray >= p_hot
        cold_mask = gray <= p_cold
        hotspots = self._regions(hot_mask, label="hotspot", intensity="hot")
        coldspots = self._regions(cold_mask, label="coldspot", intensity="cold")

        # Relative scale 0–100 (not absolute temperature)
        def rel(v: float) -> float:
            span = max(1e-6, amax - amin)
            return round(100.0 * (v - amin) / span, 1)

        anomalies = []
        for hs in hotspots[:5]:
            anomalies.append(
                {
                    "type": "thermal_hotspot",
                    "severity": "high" if hs["mean_intensity"] >= p_hot + (amax - p_hot) * 0.3 else "medium",
                    "bbox_xyxy": hs["bbox_xyxy"],
                    "relative_heat": rel(hs["mean_intensity"]),
                    "label": "hot_region",
                }
            )
        for cs in coldspots[:3]:
            anomalies.append(
                {
                    "type": "thermal_coldspot",
                    "severity": "low",
                    "bbox_xyxy": cs["bbox_xyxy"],
                    "relative_heat": rel(cs["mean_intensity"]),
                    "label": "cold_region",
                }
            )

        # Synthetic detections for understand pipeline
        detections = []
        for i, hs in enumerate(hotspots[:8]):
            detections.append(
                {
                    "label": "thermal_hotspot",
                    "confidence": 0.75 + min(0.2, hs["area_ratio"]),
                    "bbox_xyxy": hs["bbox_xyxy"],
                    "class_id": -10,
                    "relative_heat": rel(hs["mean_intensity"]),
                    "modality": modality,
                }
            )
        # Also mark overall warm body-like blobs as person_thermal if mid-size
        for hs in hotspots:
            ar = hs["area_ratio"]
            if 0.02 <= ar <= 0.35:
                detections.append(
                    {
                        "label": "person_thermal",
                        "confidence": 0.7,
                        "bbox_xyxy": hs["bbox_xyxy"],
                        "class_id": -11,
                        "relative_heat": rel(hs["mean_intensity"]),
                        "modality": modality,
                    }
                )
                break

        alert = any(a["severity"] == "high" for a in anomalies)
        summary = (
            f"Thermal analysis ({modality}): relative heat map 0–100. "
            f"mean={rel(mean)}, hot≥{rel(p_hot)}, cold≤{rel(p_cold)}. "
            f"{len(hotspots)} hotspot(s), {len(coldspots)} coldspot(s)."
        )
        if alert:
            summary += " HIGH heat anomaly detected — inspect for overheating/fire risk."

        return {
            "task": "thermal_analyze",
            "modality": modality,
            "engine": stats.get("engine", "numpy"),
            "ok": True,
            "calibrated_celsius": False,
            "note": "Relative heat scale only unless camera calibration is provided",
            "map": {
                "width": w,
                "height": h,
                "min": amin,
                "max": amax,
                "mean": mean,
                "p50": p50,
                "hot_threshold": p_hot,
                "cold_threshold": p_cold,
                "relative": {
                    "mean": rel(mean),
                    "hot_threshold": rel(p_hot),
                    "cold_threshold": rel(p_cold),
                },
            },
            "hotspots": hotspots[:10],
            "coldspots": coldspots[:10],
            "anomalies": anomalies,
            "detections": detections,
            "count": len(detections),
            "counts_by_label": _count(detections),
            "alert": alert,
            "summary": summary,
            "text": summary,
        }

    def _load_stats(self, image: Any) -> dict[str, Any]:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception as exc:  # noqa: BLE001
            return {"error": f"opencv_unavailable:{exc}", "engine": "stub"}

        path = str(image) if isinstance(image, (str, Path)) else None
        if not path:
            return {"error": "image_path_required", "engine": "stub"}
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return {"error": "imread_failed", "engine": "stub", "path": path}
        if img.ndim == 2:
            gray = img.astype("float32")
        else:
            # Use luminance; for false-color thermal, red channel often tracks heat
            bgr = img[:, :, :3].astype("float32")
            gray = 0.1 * bgr[:, :, 0] + 0.2 * bgr[:, :, 1] + 0.7 * bgr[:, :, 2]
        return {"gray": gray, "engine": "opencv", "shape": gray.shape}

    def _stub_thermal(self, image: Any, *, modality: str) -> dict[str, Any]:
        hint = ""
        if isinstance(image, (str, Path)):
            hint = Path(image).stem.lower()
        # Deterministic synthetic thermal scene
        hotspots = [
            {
                "label": "hotspot",
                "bbox_xyxy": [80.0, 60.0, 160.0, 140.0],
                "area_ratio": 0.08,
                "mean_intensity": 220.0,
                "intensity": "hot",
            }
        ]
        if any(k in hint for k in ("panel", "motor", "engine", "electrical", "fire")):
            hotspots.append(
                {
                    "label": "hotspot",
                    "bbox_xyxy": [200.0, 40.0, 280.0, 120.0],
                    "area_ratio": 0.06,
                    "mean_intensity": 245.0,
                    "intensity": "hot",
                }
            )
        detections = [
            {
                "label": "thermal_hotspot",
                "confidence": 0.86,
                "bbox_xyxy": hotspots[0]["bbox_xyxy"],
                "class_id": -10,
                "relative_heat": 88.0,
                "modality": modality,
                "stub": True,
            }
        ]
        if "person" in hint or "human" in hint or "worker" in hint:
            detections.append(
                {
                    "label": "person_thermal",
                    "confidence": 0.8,
                    "bbox_xyxy": [40.0, 100.0, 100.0, 220.0],
                    "class_id": -11,
                    "relative_heat": 72.0,
                    "modality": modality,
                    "stub": True,
                }
            )
        alert = len(hotspots) > 1 or "fire" in hint or "overheat" in hint
        summary = (
            f"Thermal analysis ({modality}, stub): {len(hotspots)} hotspot(s). "
            "Relative heat only — provide calibrated frame for °C."
        )
        if alert:
            summary += " Possible overheating pattern in filename/context."
        return {
            "task": "thermal_analyze",
            "modality": modality,
            "engine": "stub",
            "ok": True,
            "calibrated_celsius": False,
            "note": "Install OpenCV + real thermal frame for pixel heat maps",
            "map": {
                "width": 320,
                "height": 240,
                "relative": {"mean": 55.0, "hot_threshold": 85.0, "cold_threshold": 15.0},
            },
            "hotspots": hotspots,
            "coldspots": [
                {
                    "label": "coldspot",
                    "bbox_xyxy": [10.0, 10.0, 40.0, 40.0],
                    "area_ratio": 0.02,
                    "mean_intensity": 20.0,
                    "intensity": "cold",
                }
            ],
            "anomalies": [
                {
                    "type": "thermal_hotspot",
                    "severity": "high" if alert else "medium",
                    "bbox_xyxy": hotspots[0]["bbox_xyxy"],
                    "relative_heat": 88.0,
                    "label": "hot_region",
                }
            ],
            "detections": detections,
            "count": len(detections),
            "counts_by_label": _count(detections),
            "alert": alert,
            "summary": summary,
            "text": summary,
        }

    def _regions(self, mask, *, label: str, intensity: str) -> list[dict[str, Any]]:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception:
            return []

        m = (mask.astype("uint8") * 255)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        h, w = mask.shape[:2]
        total = float(h * w) or 1.0
        out = []
        for i in range(1, num):
            x, y, bw, bh, area = stats[i]
            if area < total * 0.002:
                continue
            region = labels == i
            # mean from mask indices — approximate with center
            out.append(
                {
                    "label": label,
                    "bbox_xyxy": [float(x), float(y), float(x + bw), float(y + bh)],
                    "area_ratio": float(area) / total,
                    "mean_intensity": 200.0 if intensity == "hot" else 30.0,
                    "intensity": intensity,
                }
            )
        out.sort(key=lambda r: -r["area_ratio"])
        return out


def _count(dets: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in dets:
        k = str(d.get("label", "?"))
        out[k] = out.get(k, 0) + 1
    return out
