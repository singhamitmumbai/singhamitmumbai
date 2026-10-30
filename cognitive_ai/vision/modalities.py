"""Image modality detection — RGB, thermal/IR, night vision, X-ray, depth, document, etc."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


MODALITIES = (
    "rgb",
    "thermal",
    "infrared",
    "night_vision",
    "xray",
    "depth",
    "ultrasound",
    "document",
    "multispectral",
    "unknown",
)


# Filename / path cues → modality
_CUES: list[tuple[str, tuple[str, ...]]] = [
    ("thermal", ("thermal", "thermography", "flir", "heatcam", "heat_map", "heatmap_ir", "lwir", "mwir")),
    ("infrared", ("infrared", "_ir_", "nir_", "swir", "ir_image", "ir-cam")),
    ("night_vision", ("nightvision", "night_vision", "nvg", "lowlight", "low_light", "starlight")),
    ("xray", ("xray", "x-ray", "radiograph", "ct_scan")),
    ("depth", ("depth", "disparity", "rgbd", "pointcloud")),
    ("ultrasound", ("ultrasound", "sonogram", "echo_")),
    ("document", ("scan", "invoice", "receipt", "document", "page_", "ocr_")),
    ("multispectral", ("multispectral", "hyperspectral", "ndvi")),
]


def detect_modality(
    image: Any,
    *,
    hint: Optional[str] = None,
    force: Optional[str] = None,
) -> dict[str, Any]:
    """Infer imaging modality from path/name cues and optional pixel stats."""
    if force and force in MODALITIES:
        return {
            "modality": force,
            "confidence": 1.0,
            "method": "forced",
            "cues": [force],
            "pixel_stats": None,
        }

    path_str = ""
    if isinstance(image, (str, Path)):
        path_str = str(image)
    elif hasattr(image, "filename"):
        path_str = str(getattr(image, "filename") or "")

    blob = f"{path_str} {hint or ''}".lower().replace("-", "_")
    cues_hit: list[str] = []
    modality = "rgb"
    confidence = 0.55

    for mod, keys in _CUES:
        matched = [k for k in keys if k in blob]
        if matched:
            modality = mod
            cues_hit = matched
            confidence = 0.82 + min(0.15, 0.03 * len(matched))
            break

    # Extension hints
    suffix = Path(path_str).suffix.lower() if path_str else ""
    if suffix in (".dcm",):  # DICOM often medical
        if modality == "rgb":
            modality = "xray"
            cues_hit.append(".dcm")
            confidence = max(confidence, 0.7)

    pixel_stats = _pixel_stats(image)
    if pixel_stats and modality in ("rgb", "unknown"):
        # Grayscale + high dynamic range + "hot" colormap cues → thermal-ish
        if pixel_stats.get("likely_thermal"):
            modality = "thermal"
            cues_hit.append("pixel_thermal_heuristic")
            confidence = max(confidence, 0.68)
        elif pixel_stats.get("likely_night"):
            modality = "night_vision"
            cues_hit.append("pixel_night_heuristic")
            confidence = max(confidence, 0.62)
        elif pixel_stats.get("likely_document"):
            modality = "document"
            cues_hit.append("pixel_document_heuristic")
            confidence = max(confidence, 0.6)

    return {
        "modality": modality if modality in MODALITIES else "unknown",
        "confidence": round(confidence, 3),
        "method": "cues+pixels" if pixel_stats else "cues",
        "cues": cues_hit,
        "path": path_str or None,
        "suffix": suffix or None,
        "pixel_stats": pixel_stats,
        "supported_pipeline": _pipeline_for(modality if modality in MODALITIES else "unknown"),
    }


def _pipeline_for(modality: str) -> list[str]:
    base = ["modality_detect", "see", "analyze", "understand", "search_if_needed"]
    if modality == "thermal":
        return ["modality_detect", "thermal_map", "hotspot_detect", "see", "analyze", "understand", "search_if_needed"]
    if modality == "infrared":
        return ["modality_detect", "ir_enhance", "see", "analyze", "understand", "search_if_needed"]
    if modality == "night_vision":
        return ["modality_detect", "lowlight_adapt", "see", "analyze", "understand", "search_if_needed"]
    if modality == "xray":
        return ["modality_detect", "contrast_stretch", "anomaly_scan", "understand", "search_if_needed"]
    if modality == "depth":
        return ["modality_detect", "depth_stats", "see", "analyze", "understand"]
    if modality == "document":
        return ["modality_detect", "scan_document", "ocr", "understand"]
    if modality == "ultrasound":
        return ["modality_detect", "speckle_stats", "region_detect", "understand", "search_if_needed"]
    if modality == "multispectral":
        return ["modality_detect", "band_stats", "see", "analyze", "understand"]
    return base


def _pixel_stats(image: Any) -> Optional[dict[str, Any]]:
    """Optional OpenCV stats; returns None if unavailable or unreadable."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return None

    path = str(image) if isinstance(image, (str, Path)) else None
    if not path:
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        # Synthetic stats from filename for stub environments
        return None

    if img.ndim == 2:
        gray = img.astype("float32")
        channels = 1
    else:
        channels = img.shape[2] if img.ndim == 3 else 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32")

    mean = float(gray.mean())
    std = float(gray.std())
    amin = float(gray.min())
    amax = float(gray.max())
    # Simple entropy proxy
    hist = np.histogram(gray, bins=32, range=(0, 255))[0].astype("float32")
    hist = hist / max(1.0, hist.sum())
    entropy = float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum())

    likely_thermal = channels == 1 and std > 35 and entropy > 3.5
    # Also common false-color thermal saved as 3-channel with high red dominance
    if img.ndim == 3 and channels >= 3:
        b, g, r = cv2.split(img[:, :, :3].astype("float32"))
        if float(r.mean()) > float(b.mean()) + 25 and std > 30:
            likely_thermal = True

    likely_night = mean < 55 and std < 40
    likely_document = mean > 180 and std < 45 and channels in (1, 3)

    return {
        "channels": channels,
        "mean": round(mean, 2),
        "std": round(std, 2),
        "min": round(amin, 2),
        "max": round(amax, 2),
        "entropy": round(entropy, 3),
        "likely_thermal": bool(likely_thermal),
        "likely_night": bool(likely_night),
        "likely_document": bool(likely_document),
        "engine": "opencv",
    }
