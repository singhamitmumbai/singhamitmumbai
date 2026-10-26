"""Complete non-LLM cognitive AI — symbolic brain + Finderbay assistant.

No pretrained general-purpose LLM at runtime. Neural specialists for speech/OCR
may be plugged in with audited provenance; this package ships symbolic /
algorithmic cognition plus measurable perception stubs.
"""

from .brain import Brain
from .version import __version__

__all__ = ["Brain", "CompleteNonLLMAI", "__version__", "CognitiveOS"]


def __getattr__(name: str):
    if name == "CognitiveOS":
        from .os.system import CognitiveOS

        return CognitiveOS
    if name == "CompleteNonLLMAI":
        from .system import CompleteNonLLMAI

        return CompleteNonLLMAI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
