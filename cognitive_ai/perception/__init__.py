"""Perception pipelines: speech, OCR, sensors — measurable and swappable."""

from .gateway import InputGateway
from .speech import SpeechPipeline
from .ocr import DocumentPipeline

__all__ = ["InputGateway", "SpeechPipeline", "DocumentPipeline"]
