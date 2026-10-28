"""Completeness audit — verify the full Complete AI code surface is wired."""

from __future__ import annotations

import importlib
from typing import Any

from ..version import __version__


REQUIRED_MODULES = [
    "cognitive_ai.complete_ai",
    "cognitive_ai.system",
    "cognitive_ai.general.agent",
    "cognitive_ai.chat.llm_bridge",
    "cognitive_ai.chat.tools",
    "cognitive_ai.vision.system",
    "cognitive_ai.vision.understand",
    "cognitive_ai.vision.knowledge",
    "cognitive_ai.vision.search",
    "cognitive_ai.vision.modalities",
    "cognitive_ai.vision.thermal",
    "cognitive_ai.vision.smart",
    "cognitive_ai.coding.system",
    "cognitive_ai.coding.verifier",
    "cognitive_ai.coding.generator",
    "cognitive_ai.coding.editor",
    "cognitive_ai.coding.runner",
    "cognitive_ai.agent.self_brain",
    "cognitive_ai.api.server",
    "cognitive_ai.ui.server",
    "cognitive_ai.nlu.pipeline",
    "cognitive_ai.commonsense.chat",
    "cognitive_ai.runtime.persist",
    "cognitive_ai.response.hindi",
    "cognitive_ai.assistant.finder_assistant",
    "cognitive_ai.assistant.voice",
    "cognitive_ai.assistant.computer",
    "cognitive_ai.assistant.computer_policy",
    "cognitive_ai.assistant.computer_nlu",
    "cognitive_ai.assistant.memory_hub",
    "cognitive_ai.assistant.learning_bridge",
    "cognitive_ai.jarvis.pipeline",
    "cognitive_ai.jarvis.registry",
    "cognitive_ai.jarvis.skills",
    "cognitive_ai.jarvis.provenance",
    "cognitive_ai.jarvis.voice",
    "cognitive_ai.jarvis.eval_suite",
    "cognitive_ai.jarvis.system",
    "cognitive_ai.jarvis.nlu",
    "cognitive_ai.jarvis.sqlite_memory",
    "cognitive_ai.jarvis.learning",
    "cognitive_ai.jarvis.policy",
    "cognitive_ai.jarvis.documents",
    "cognitive_ai.jarvis.dialogue",
    "cognitive_ai.jarvis.adapters",
    "cognitive_ai.intelligence.router",
    "cognitive_ai.intelligence.confidence",
    "cognitive_ai.intelligence.hardening",
    "cognitive_ai.nlu.smart_questions",
    "cognitive_ai.search.web_intel",
    "cognitive_ai.response.chatgpt_style",
    "cognitive_ai.agi_gap.stack",
    "cognitive_ai.agi_gap.world_knowledge",
    "cognitive_ai.agi_gap.continual_skills",
    "cognitive_ai.agi_gap.deep_transfer",
    "cognitive_ai.agi_gap.long_horizon",
    "cognitive_ai.agi_gap.multimodal_hub",
    "cognitive_ai.agi_gap.lifelong_memory",
    "cognitive_ai.agi_gap.metacognition",
    "cognitive_ai.agi_gap.alignment",
    "cognitive_ai.agi_gap.general_learning",
    "cognitive_ai.agi_gap.eval_suite",
    "cognitive_ai.agi_gap.scorecard",
    "cognitive_ai.benchmarks.hard_novel",
    "cognitive_ai.thinking",
    "cognitive_ai.thinking.deliberator",
    "cognitive_ai.thinking.attach",
    "cognitive_ai.runtime.full_suite",
    "cognitive_ai.benchmarks.scan_like",
    "cognitive_ai.benchmarks.transfer_bench",
    "cognitive_ai.perception.model_registry",
    "cognitive_ai.perception.ocr_layout",
    "cognitive_ai.perception.speech_streaming",
    "cognitive_ai.domains.finderbay.auth",
    "cognitive_ai.domains.finderbay.payments",
    "cognitive_ai.domains.finderbay.contracts",
]


REQUIRED_ATTRS = {
    "cognitive_ai.complete_ai": ["CompleteAI", "create_complete_ai"],
    "cognitive_ai.chat.tools": ["ToolRouter", "extract_tool_calls"],
    "cognitive_ai.chat.llm_bridge": ["LLMChatBridge", "llm_bridge_from_env"],
    "cognitive_ai.vision.system": ["VisionSystem"],
    "cognitive_ai.vision.understand": ["VisionUnderstanding"],
    "cognitive_ai.vision.knowledge": ["ObjectKnowledge"],
    "cognitive_ai.vision.modalities": ["detect_modality", "MODALITIES"],
    "cognitive_ai.vision.thermal": ["ThermalAnalyzer"],
    "cognitive_ai.vision.smart": ["SmartVisionReasoner"],
    "cognitive_ai.coding.system": ["CodingSystem"],
    "cognitive_ai.coding.verifier": ["CodeVerifier"],
    "cognitive_ai.coding.generator": ["generate_function", "parse_generate_request", "generate_enum"],
    "cognitive_ai.coding.editor": ["CodeEditor"],
    "cognitive_ai.coding.runner": ["CodeRunner"],
    "cognitive_ai.assistant.finder_assistant": ["FinderbayAssistant"],
    "cognitive_ai.jarvis.pipeline": ["JarvisPipeline"],
    "cognitive_ai.jarvis.skills": ["MaterialShortageSkill", "SkillRegistry"],
    "cognitive_ai.jarvis.eval_suite": ["run_jarvis_eval"],
    "cognitive_ai.jarvis.system": ["JarvisSystem"],
    "cognitive_ai.jarvis.nlu": ["IntentClassifier"],
    "cognitive_ai.jarvis.sqlite_memory": ["SqliteMemory"],
    "cognitive_ai.jarvis.learning": ["LearningPipeline"],
    "cognitive_ai.intelligence.router": ["SmartRouter", "RouteDecision"],
    "cognitive_ai.intelligence.confidence": ["score_response", "ConfidenceScore"],
    "cognitive_ai.intelligence.hardening": ["SelfCheck", "harden_text"],
    "cognitive_ai.nlu.smart_questions": ["SmartQuestionUnderstanding"],
    "cognitive_ai.search.web_intel": ["IntelligentWebSearch", "SearchResult"],
    "cognitive_ai.response.chatgpt_style": ["ChatGPTStyleComposer", "ChatGPTStyleAnswer"],
    "cognitive_ai.agi_gap.stack": ["ProtoAGIStack", "run_agi_gap_demo"],
    "cognitive_ai.agi_gap.world_knowledge": ["WorldKnowledgeHub", "OpenWorldFrame"],
    "cognitive_ai.agi_gap.continual_skills": ["GrowingSkillLibrary"],
    "cognitive_ai.agi_gap.deep_transfer": ["DeepTransferEngine"],
    "cognitive_ai.agi_gap.long_horizon": ["LongHorizonAgent"],
    "cognitive_ai.agi_gap.multimodal_hub": ["AlwaysOnMultimodalHub"],
    "cognitive_ai.agi_gap.lifelong_memory": ["LifelongMemory"],
    "cognitive_ai.agi_gap.metacognition": ["MetaCognition"],
    "cognitive_ai.agi_gap.alignment": ["AlignmentGuard"],
    "cognitive_ai.agi_gap.general_learning": ["GeneralLearner"],
    "cognitive_ai.agi_gap.eval_suite": ["run_agi_grade_eval"],
    "cognitive_ai.agi_gap.scorecard": ["build_scorecard", "AGIGapScorecard"],
    "cognitive_ai.benchmarks.hard_novel": ["run_hard_novel_benchmark"],
    "cognitive_ai.thinking": [
        "ThinkingSystem",
        "Deliberation",
        "ThoughtStep",
        "think_about",
        "deliberate",
        "merge_thinking",
        "think_or_abstain",
    ],
    "cognitive_ai.thinking.deliberator": ["ThinkingSystem", "Deliberation", "think_about"],
    "cognitive_ai.thinking.attach": ["deliberate", "merge_thinking", "think_or_abstain", "get_thinker"],
    "cognitive_ai.runtime.full_suite": ["run_full_suite"],
    "cognitive_ai.benchmarks.scan_like": ["run_scan_benchmark"],
    "cognitive_ai.benchmarks.transfer_bench": ["run_transfer_benchmark"],
    "cognitive_ai.perception.model_registry": ["list_models", "MODELS", "ensure_model"],
    "cognitive_ai.perception.ocr_layout": ["LayoutOCR", "DocumentResult"],
    "cognitive_ai.perception.speech_streaming": ["StreamingASR", "mic_stream_loop"],
    "cognitive_ai.domains.finderbay.auth": ["AuthService"],
    "cognitive_ai.domains.finderbay.payments": ["PaymentService"],
    "cognitive_ai.domains.finderbay.contracts": ["ContractService"],
    "cognitive_ai.assistant.computer": ["ComputerControl", "ALLOWED"],
    "cognitive_ai.assistant.computer_policy": ["policy_snapshot", "resolve_in_sandbox", "APP_ALLOWLIST"],
    "cognitive_ai.assistant.computer_nlu": ["parse_computer_command"],
}


def run_completeness_check() -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for mod_name in REQUIRED_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            missing = [
                a for a in REQUIRED_ATTRS.get(mod_name, []) if not hasattr(mod, a)
            ]
            results.append(
                {
                    "module": mod_name,
                    "ok": not missing,
                    "missing_attrs": missing,
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append({"module": mod_name, "ok": False, "error": str(exc)})

    # Smoke-construct Complete AI
    try:
        from ..complete_ai import CompleteAI

        ai = CompleteAI(seed_demo=False, enable_llm=False)
        diag = ai.diagnostics()
        results.append(
            {
                "module": "CompleteAI.diagnostics",
                "ok": bool(diag.get("ok")),
                "detail": {"passed": diag.get("passed"), "total": diag.get("total")},
            }
        )
        export = ai.export_session(fmt="markdown")
        results.append(
            {
                "module": "CompleteAI.export_session",
                "ok": isinstance(export, str) and "Complete AI" in export,
            }
        )
    except Exception as exc:  # noqa: BLE001
        results.append({"module": "CompleteAI.smoke", "ok": False, "error": str(exc)})

    passed = sum(1 for r in results if r.get("ok"))
    return {
        "product": "Complete AI",
        "version": __version__,
        "ok": passed == len(results),
        "passed": passed,
        "total": len(results),
        "results": results,
        "human_level_agi": False,
        "complete_code": passed == len(results),
    }
