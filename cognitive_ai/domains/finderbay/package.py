"""Finderbay domain package bundle: ontology, vocabulary, templates, eval cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cognitive_ai.capabilities.spec import FINDERBAY_CAPABILITIES
from cognitive_ai.evaluation.harness import EvalCase
from cognitive_ai.memory.ontology import finderbay_ontology
from cognitive_ai.memory.store import MemorySystem
from .seed import seed_memory


RESPONSE_TEMPLATES = {
    "invoice_summary": "{count} invoices matched, totaling {total}.",
    "pending_summary": "Found {pending} confirmed pending delivery item(s).",
    "draft_not_sent": "A follow-up draft was prepared and was not sent.",
}


@dataclass
class FinderbayPackage:
    name: str = "finderbay"
    version: str = "1.0.0"

    def ontology(self):
        return finderbay_ontology()

    def capabilities(self):
        return FINDERBAY_CAPABILITIES

    def seed(self, memory: MemorySystem | None = None) -> MemorySystem:
        return seed_memory(memory)

    def templates(self) -> dict[str, str]:
        return dict(RESPONSE_TEMPLATES)

    def evaluation_cases(self) -> list[EvalCase]:
        return [
            EvalCase(
                case_id="inv_unpaid_gt_1lakh_excl_disputed",
                task="query_invoices",
                input_text=(
                    "Show unpaid invoices above ₹1 lakh for Project A, excluding disputed ones."
                ),
                # inv_42 150k + inv_43 250k matched; inv_44 disputed excluded;
                # inv_45 unknown dispute reported separately
                expected={"count": 2, "unresolved_count": 1},
                tags=["language", "composition", "invoices"],
            ),
            EvalCase(
                case_id="pending_followup_no_send",
                task="prepare_followup",
                input_text=(
                    "From this week's delivery records, find the items still pending for "
                    "Project A and prepare a follow-up. Do not send it."
                ),
                expected={
                    "pending_count": 1,
                    "uncertain_count": 1,
                    "draft_prepared": True,
                    "sent": False,
                    "contains": "not sent",
                },
                tags=["planning", "verification", "deliveries"],
            ),
            EvalCase(
                case_id="kal_ambiguity",
                task="clarify",
                input_text="Dikhao pending deliveries for Project A kal",
                expected={"contains": "kal"},
                tags=["hinglish", "ambiguity"],
            ),
            EvalCase(
                case_id="status",
                task="status",
                input_text="What are your capabilities?",
                expected={"contains": "non-LLM"},
                tags=["meta"],
            ),
        ]

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "capabilities": [c.name for c in self.capabilities()],
            "ontology_concepts": list(self.ontology().concepts.keys()),
            "templates": list(self.templates().keys()),
            "eval_cases": len(self.evaluation_cases()),
        }
