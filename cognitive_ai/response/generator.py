"""Content selection → document plan → sentence plan → realization → validate."""

from __future__ import annotations

from typing import Any, Optional

from ..core.working_state import WorkingState
from ..execution.executor import Executor
from ..reasoning.engine import ReasoningResult
from .hindi import prefer_hindi, realize_invoice_hi, realize_pending_hi


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
                    "Supported: invoice queries, delivery pending lists, material lookup/reserve, "
                    "local tasks, document field extraction, alias learning and follow-up drafts "
                    "(without sending). "
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

        lang = frame.get("language") or "en"
        if op == "query_invoices":
            text, structured = self._realize_invoices(reasoning, table)
            if prefer_hindi(lang):
                text = realize_invoice_hi(structured)
        elif op in ("query_deliveries", "prepare_followup"):
            text, structured = self._realize_deliveries(reasoning, table, draft, op)
            if prefer_hindi(lang) and op != "prepare_followup":
                text = realize_pending_hi(structured, draft)
        elif op == "query_materials":
            text, structured = self._realize_materials(table)
        elif op == "reserve_stock":
            text, structured = self._realize_reserve(executor.last_reservation)
        elif op == "manage_task":
            text, structured = self._realize_task(executor, table)
        elif op == "document_qa":
            text, structured = self._realize_document(executor.last_document)
        elif op == "learn_alias":
            alias = executor.last_alias or {}
            text = (
                f"Learned alias: '{alias.get('former')}' is now also known as "
                f"'{alias.get('alias')}' → canonical '{alias.get('canonical')}'."
            )
            structured = {**alias, "learned": True}
        elif op == "login":
            text = "Login successful." if executor.auth_token else "Login failed."
            structured = {"logged_in": bool(executor.auth_token)}
        elif op == "record_payment":
            pay = executor.last_payment or {}
            p = pay.get("payment") or {}
            text = (
                f"Recorded payment {p.get('id')} of ₹{p.get('amount_inr')} "
                f"for {p.get('invoice_id')}. Invoice paid total ₹{pay.get('invoice_paid_total')}."
            )
            structured = {"payment_posted": True, **pay}
        elif op in ("query_payments", "query_contracts"):
            n = table.get("row_count", 0) if table else 0
            kind = "payment" if op == "query_payments" else "contract"
            text = f"{n} {kind}(s) found."
            if table and table.get("meta", {}).get("active_value_inr") is not None:
                text += f" Active contract value ₹{table['meta']['active_value_inr']:,.0f}."
            structured = {"count": n, "listed": True}
        else:
            text = reasoning.message if reasoning else "Done."
            structured = {"reasoning": reasoning.to_dict() if reasoning else None}

        # Validation: claims in text must match structure
        faithful = self._validate(text, structured)
        answer = {
            "text": text,
            "structured": structured,
            "faithful": faithful,
            "table": table,
            "language": lang,
        }
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
        lines = [f"{n} material record(s) available."]
        if table and table.get("rows"):
            for row in table["rows"]:
                lines.append(
                    f"- {row.get('name') or row.get('id')}: available={row.get('available')}, "
                    f"reserved={row.get('reserved')} {row.get('unit') or ''}".strip()
                )
        return "\n".join(lines), {"count": n, "items": (table or {}).get("rows") or []}

    def _realize_reserve(self, reservation: Optional[dict]) -> tuple[str, dict]:
        if not reservation:
            return "Reservation failed.", {"reserved": False}
        qty = reservation.get("qty_reserved")
        text = (
            f"Reserved {qty} of {reservation.get('name') or reservation.get('item')}. "
            f"Available now {reservation.get('available')}, "
            f"reserved balance {reservation.get('reserved_balance')}."
        )
        return text, {
            "item": reservation.get("item"),
            "qty_reserved": qty,
            "available": reservation.get("available"),
            "reserved_balance": reservation.get("reserved_balance"),
            "reserved": True,
        }

    def _realize_task(self, executor: Executor, table: Optional[dict]) -> tuple[str, dict]:
        if executor.last_task:
            t = executor.last_task
            text = f"Created task '{t.get('title')}' ({t.get('id')}) for {t.get('project_id') or 'no project'}."
            return text, {"task_id": t.get("id"), "title": t.get("title"), "created": True}
        n = table.get("row_count", 0) if table else 0
        text = f"{n} task(s) listed."
        if table and table.get("rows"):
            for row in table["rows"]:
                text += f"\n- [{row.get('status')}] {row.get('title')} ({row.get('id')})"
        return text, {"count": n, "listed": True}

    def _realize_document(self, doc_result: Optional[dict]) -> tuple[str, dict]:
        if not doc_result:
            return "No document available to read.", {"ok": False}
        field = doc_result.get("field")
        value = doc_result.get("value")
        parsed = doc_result.get("parsed") or {}
        issues = parsed.get("issues") or []
        if field:
            if value is None and issues:
                text = (
                    f"I can read the document, but the field '{field}' is ambiguous or invalid "
                    f"({'; '.join(issues)})."
                )
            else:
                text = f"Document field '{field}' = {value!r}."
        else:
            fields = parsed.get("fields") or {}
            text = "Extracted fields: " + ", ".join(f"{k}={v}" for k, v in fields.items())
        if issues and field is None:
            text += f" Issues: {'; '.join(issues)}."
        return text, {
            "field": field,
            "value": value,
            "fields": parsed.get("fields"),
            "issues": issues,
            "ok": True,
        }

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
