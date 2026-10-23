"""Symbolic cognitive AI — no LLM dependencies.

A runnable “brain” with knowledge memory, beliefs, inference, planning,
self-model, think loop, and simple rule/fact learning.
"""

from .brain import Brain
from .version import __version__

__all__ = ["Brain", "__version__"]
