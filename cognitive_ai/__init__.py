"""Complete non-LLM cognitive AI — Finderbay + open-domain + speech/OCR.

No pretrained general-purpose LLM at runtime. Includes:
- Finderbay task assistant (invoices, deliveries, materials, tasks, documents)
- Open-domain common-sense chat (curated KB + rules)
- Production speech (Vosk) and OCR (Tesseract) when installed
- Symbolic Cognitive OS / brain demos

Honest limits: not human-level AGI; open-domain coverage is finite and explicit.
"""

from .brain import Brain
from .version import __version__

__all__ = [
    "Brain",
    "CompleteNonLLMAI",
    "GeneralNonLLMAI",
    "__version__",
    "CognitiveOS",
]


def __getattr__(name: str):
    if name == "CognitiveOS":
        from .os.system import CognitiveOS

        return CognitiveOS
    if name == "CompleteNonLLMAI":
        from .system import CompleteNonLLMAI

        return CompleteNonLLMAI
    if name == "GeneralNonLLMAI":
        from .general.agent import GeneralNonLLMAI

        return GeneralNonLLMAI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
