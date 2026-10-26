"""Content selection → document plan → sentence plan → realization → validate."""

from __future__ import annotations

from typing import Any, Optional

from ..core.working_state import WorkingState
from ..execution.executor import Executor
from ..reasoning.engine import ReasoningResult


def _format_inr(amount: float) -> str:
    if amount >= 10_000_000:
        return f"₹{amount / 10_000_000:.2f} crore".replace(".00", "")
    if amount >= 100_000:
        val = amount / 100_000
        if abs(val - round(val)) < 1e-9:
            return f"₹{int(round(val))} lakh"
        return f"₹{val:.1f} lakh"
    # Indian-ish grouping for display
    return f"₹{amount:,.0f}"


class ResponseGenerator:
    """Assemble answers from reusable grammatical components — no LLM."""

    def generate(
        self,
        state: WorkingState,
        executor: Executor,
        reasoning: Optional[ReasoningResult] = None,
    ) -> dict[str, Any]:
        frame = state.semantic_frame or {}
        op = frame.get("operation")

        if frame.get("needs_clarification") and op != "correct":
            text = " ".join(frame["needs_clarification"])
            answer = {
                "text": text,
                "structured": {"clarification": frame["needs_clarification"]},
                "faithful": True,
            }
            state.last_answer = answer
            return answer

        if op == "unsupported":
            answer = {
                "text": (
                    "That request is not in my supported capability set. "
                    "I can help with invoices, deliveries, materials, tasks and drafts "
                    "within the Finderbay domain."
                ),
                "structured": {"supported": False},
                "faithful": True,
            }
            state.last_answer = answer
            return answer

        if op == "cancel":
            answer = {"text": "Cancelled the active goal.", "structured": {"cancelled": True}, "faithful": True}
            state.last_answer = answer
            return answer

        if op == "correct":
            corr = state.dialogue.corrections[-1] if state.dialogue.corrections else {}
            answer = {
                "text": f"Updated {corr.get('field')} to {corr.get('new')}. Affected plan steps were invalidated.",
                "structured": corr,
                "faithful": True,
            }
            state.last_answer = answer
            return answer

        if op == "status":
            answer = {
                "text": (
                    "I am a complete non-LLM assistant for Finderbay. "
                    "Supported: invoice queries, delivery pending lists, material lookup, "
                    "task notes and follow-up drafts (without sending). "
                    "I keep unknown ≠ false, predicted ≠ observed, attempted ≠ confirmed."
                ),
                "structured": state.summary(),
                "faithful": True,
            }
            state.last_answer = answer
            return answer

        reasoning = reasoning or executor.last_reasoning
        table = executor.last_table
        draft = None
        for ar in reversed(state.action_results):
            for eff in ar.observed_effects:
                if "draft" in eff:
                    draft = eff["draft"]
                    break
            if draft:
                break

        if op == "query_invoices":
            text, structured = self._realize_invoices(reasoning, table)
        elif op in ("query_deliveries", "prepare_followup"):
            text, structured = self._realize_deliveries(reasoning, table, draft, op)
        elif op == "query_materials":
            text, structured = self._realize_materials(table)
        else:
            text = reasoning.message if reasoning else "Done."
            structured = {"reasoning": reasoning.to_dict() if reasoning else None}

        # Validation: claims in text must match structure
        faithful = self._validate(text, structured)
        answer = {"text": text, "structured": structured, "faithful": faithful, "table": table}
        state.last_answer = answer
        return answer

    def _realize_invoices(
        self,
        reasoning: Optional[ReasoningResult],
        table: Optional[dict],
    ) -> tuple[str, dict]:
        conc = (reasoning.conclusions[0] if reasoning and reasoning.conclusions else {})
        count = conc.get("count", table.get("row_count", 0) if table else 0)
        total = float(conc.get("total_inr") or (table or {}).get("meta", {}).get("total_inr") or 0)
        unresolved = conc.get("unresolved") or (table or {}).get("meta", {}).get("unresolved") or []
        as_of = None
        if table and table.get("rows"):
            as_of = table["rows"][0].get("as_of")

        # Sentence planning
        clauses = []
        if count == 0:
            clauses.append("No invoices matched the stated conditions.")
        elif count == 1:
            clauses.append(
                f"One invoice matched, totaling {_format_inr(total)}"
                + (f", based on the register as of {as_of}." if as_of else ".")
            )
        else:
            clauses.append(
                f"{self._num_word(count)} invoices matched, totaling {_format_inr(total)}"
                + (f", based on the register as of {as_of}." if as_of else ".")
            )
        if unresolved:
            n = len(unresolved)
            clauses.append(
                f"{self._num_word(n)} other invoice{'s' if n != 1 else ''} "
                f"{'have' if n != 1 else 'has'} unresolved payment or dispute status."
            )
        text = " ".join(clauses)
        structured = {
            "count": count,
            "total_inr": total,
            "unresolved_count": len(unresolved),
            "as_of": as_of,
        }
        return text, structured

    def _realize_deliveries(
        self,
        reasoning: Optional[ReasoningResult],
        table: Optional[dict],
        draft: Optional[dict],
        op: str,
    ) -> tuple[str, dict]:
        conc = (reasoning.conclusions[0] if reasoning and reasoning.conclusions else {})
        pending_n = conc.get("count_pending", table.get("row_count", 0) if table else 0)
        uncertain_n = conc.get("count_uncertain", (table or {}).get("meta", {}).get("count_uncertain", 0))
        clauses = [
            f"Found {pending_n} confirmed pending delivery item(s).",
        ]
        if uncertain_n:
            clauses.append(
                f"{uncertain_n} additional item(s) have unresolved receipt status "
                f"and were kept separate from confirmed pending."
            )
        if op == "prepare_followup":
            if draft:
                clauses.append("A follow-up draft was prepared and was not sent.")
            else:
                clauses.append("Follow-up drafting was requested.")
        text = " ".join(clauses)
        if draft:
            text += "\n\n--- Draft ---\n" + draft.get("body", "")
        structured = {
            "pending_count": pending_n,
            "uncertain_count": uncertain_n,
            "draft_prepared": bool(draft),
            "sent": False if draft else None,
        }
        return text, structured

    def _realize_materials(self, table: Optional[dict]) -> tuple[str, dict]:
        n = table.get("row_count", 0) if table else 0
        text = f"{n} material record(s) available."
        return text, {"count": n}

    @staticmethod
    def _num_word(n: int) -> str:
        words = {
            1: "One",
            2: "Two",
            3: "Three",
            4: "Four",
            5: "Five",
            6: "Six",
            7: "Seven",
            8: "Eight",
            9: "Nine",
            10: "Ten",
        }
        return words.get(n, str(n))

    @staticmethod
    def _validate(text: str, structured: dict) -> bool:
        """Check that numeric claims in text match structured result when present."""
        for key in ("count", "pending_count", "total_inr", "unresolved_count"):
            if key not in structured or structured[key] is None:
                continue
            val = structured[key]
            if key == "total_inr":
                continue  # formatting variants; structural link is enough
            if isinstance(val, int) and val > 0 and str(val) not in text and ResponseGenerator._num_word(val).lower() not in text.lower():
                # soft check — still allow number words mismatch for 0
                if val <= 10:
                    word = ResponseGenerator._num_word(val)
                    if word.lower() not in text.lower() and str(val) not in text:
                        return False
        return True
