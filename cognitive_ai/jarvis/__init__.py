"""JARVIS Tools & System Blueprint — Finderbay implementation.

Maps the engineering blueprint (capture→interpret→resolve→plan→execute→verify→respond→record)
onto typed tools, jobs, provenance memory, skills, voice pipeline, SQLite+FTS5, learning, and evaluation.
"""

from .contracts import ToolContract, JobState, FactRecord
from .registry import JarvisToolRegistry
from .jobs import JobStore, JobRecord
from .pipeline import JarvisPipeline
from .skills import MaterialShortageSkill, SkillRegistry
from .eval_suite import run_jarvis_eval
from .system import JarvisSystem
from .nlu import IntentClassifier
from .sqlite_memory import SqliteMemory
from .learning import LearningPipeline
from .policy import PolicyGuard, AuthContext
from .documents import DocumentToolkit
from .dialogue import DialogueManager
from .voice import VoicePipeline

__all__ = [
    "ToolContract",
    "JobState",
    "FactRecord",
    "JarvisToolRegistry",
    "JobStore",
    "JobRecord",
    "JarvisPipeline",
    "MaterialShortageSkill",
    "SkillRegistry",
    "run_jarvis_eval",
    "JarvisSystem",
    "IntentClassifier",
    "SqliteMemory",
    "LearningPipeline",
    "PolicyGuard",
    "AuthContext",
    "DocumentToolkit",
    "DialogueManager",
    "VoicePipeline",
]
