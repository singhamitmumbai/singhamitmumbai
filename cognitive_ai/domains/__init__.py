"""Domain packages that plug into the shared non-LLM core.

Includes legacy symbolic demo domains (animals, blocks, kitchen) and Finderbay.
"""

from .symbolic import load_domain, load_animals, load_blocks, load_kitchen

__all__ = [
    "load_domain",
    "load_animals",
    "load_blocks",
    "load_kitchen",
]
