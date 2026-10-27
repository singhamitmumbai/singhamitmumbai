"""Proto-AGI gap stack — upgrades toward broader general intelligence.

Honest: this is NOT human-level AGI. Modules advance learning, open-world
understanding, transfer, long-horizon autonomy, multimodality, lifelong memory,
metacognition, and hard evaluation/alignment — while always reporting
`human_level_agi=false`.
"""

from .general_learning import GeneralLearner
from .stack import ProtoAGIStack, run_agi_gap_demo

__all__ = ["ProtoAGIStack", "run_agi_gap_demo", "GeneralLearner"]
