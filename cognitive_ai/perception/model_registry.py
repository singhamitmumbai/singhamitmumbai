"""Speech model registry: English/Hindi Vosk models with download helpers."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"


@dataclass
class SpeechModelSpec:
    name: str
    lang: str
    url: str
    dirname: str
    size_hint_mb: int
    description: str


MODELS = {
    "en-small": SpeechModelSpec(
        "en-small",
        "en",
        "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
        "vosk-model-small-en-us-0.15",
        40,
        "Small English US model (default)",
    ),
    "en-in-small": SpeechModelSpec(
        "en-in-small",
        "en-in",
        "https://alphacephei.com/vosk/models/vosk-model-small-en-in-0.4.zip",
        "vosk-model-small-en-in-0.4",
        40,
        "Small Indian English model",
    ),
    "hi-small": SpeechModelSpec(
        "hi-small",
        "hi",
        "https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip",
        "vosk-model-small-hi-0.22",
        45,
        "Small Hindi model",
    ),
}


def model_path(name: str = "en-small") -> Path:
    spec = MODELS[name]
    return MODELS_DIR / spec.dirname


def ensure_model(name: str = "en-small", *, download: bool = True) -> Path:
    spec = MODELS[name]
    target = MODELS_DIR / spec.dirname
    if target.exists():
        return target
    if not download:
        raise FileNotFoundError(f"Model {name} not found at {target}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = MODELS_DIR / f"{spec.dirname}.zip"
    urlretrieve(spec.url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(MODELS_DIR)
    zip_path.unlink(missing_ok=True)
    if not target.exists():
        raise RuntimeError(f"Download finished but {target} missing")
    return target


def list_models() -> dict[str, dict]:
    out = {}
    for name, spec in MODELS.items():
        path = model_path(name)
        out[name] = {
            "lang": spec.lang,
            "installed": path.exists(),
            "path": str(path),
            "size_hint_mb": spec.size_hint_mb,
            "description": spec.description,
            "url": spec.url,
        }
    return out
