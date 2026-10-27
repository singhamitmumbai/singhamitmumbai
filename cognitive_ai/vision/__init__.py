"""Computer vision — smart multi-modality (RGB, thermal, IR, night, …)."""

from .system import VisionSystem
from .catalog import list_projects
from .understand import VisionUnderstanding
from .knowledge import ObjectKnowledge
from .modalities import detect_modality, MODALITIES
from .thermal import ThermalAnalyzer

__all__ = [
    "VisionSystem",
    "list_projects",
    "VisionUnderstanding",
    "ObjectKnowledge",
    "detect_modality",
    "MODALITIES",
    "ThermalAnalyzer",
]
