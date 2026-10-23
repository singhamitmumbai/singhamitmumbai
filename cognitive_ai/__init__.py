"""Symbolic cognitive AI — no LLM dependencies.

A runnable “brain” with knowledge memory, beliefs, inference, planning,
self-model, think loop, and simple rule/fact learning, plus a Cognitive OS.
"""

from .brain import Brain
from .version import __version__

__all__ = ["Brain", "__version__", "CognitiveOS"]


def __getattr__(name: str):
    if name == "CognitiveOS":
        from .os.system import CognitiveOS

        return CognitiveOS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
