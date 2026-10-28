"""Advanced non-LLM coding system: AST analysis, generation, edit, test.

Not an LLM coding assistant — symbolic/structural tools only.
"""

from .system import CodingSystem
from .verifier import CodeVerifier

__all__ = ["CodingSystem", "CodeVerifier"]
