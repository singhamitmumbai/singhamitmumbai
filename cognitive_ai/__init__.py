"""Complete AI — full stack product.

Core: Finderbay tasks, vision, coding, SelfBrain agent, speech/OCR, open-domain chat
(symbolic / non-LLM). Optional: ChatGPT-class general chat via LLM bridge when enabled.

Honest limits: human_level_agi=false. LLM is OFF by default (llm_at_runtime=false).
"""

from .brain import Brain
from .version import __version__

__all__ = [
    "Brain",
    "CompleteAI",
    "CompleteNonLLMAI",
    "GeneralNonLLMAI",
    "__version__",
    "CognitiveOS",
]


def __getattr__(name: str):
    if name == "CognitiveOS":
        from .os.system import CognitiveOS

        return CognitiveOS
    if name == "CompleteAI":
        from .complete_ai import CompleteAI

        return CompleteAI
    if name == "CompleteNonLLMAI":
        from .system import CompleteNonLLMAI

        return CompleteNonLLMAI
    if name == "GeneralNonLLMAI":
        from .general.agent import GeneralNonLLMAI

        return GeneralNonLLMAI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
