"""Explicit transition model T(s'|s,a) for known domain dynamics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Transition:
    name: str
    preconditions: Callable[[dict], bool]
    effects: Callable[[dict, dict], dict]
    description: str = ""


class WorldModel:
    """Predict how situations change under actions; start with explicit transitions."""

    def __init__(self) -> None:
        self.transitions: dict[str, Transition] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(
            Transition(
                name="reserve_stock",
                description="Move quantity from available to reserved",
                preconditions=lambda s: (
                    float(s.get("available", 0)) >= float(s.get("_qty", 0))
                    and s.get("authorized", True)
                ),
                effects=lambda s, args: {
                    **s,
                    "available": float(s.get("available", 0)) - float(args["quantity"]),
                    "reserved": float(s.get("reserved", 0)) + float(args["quantity"]),
                },
            )
        )
        self.register(
            Transition(
                name="receive_material",
                description="Increase available stock on receipt",
                preconditions=lambda s: float(s.get("_qty", 0)) > 0,
                effects=lambda s, args: {
                    **s,
                    "available": float(s.get("available", 0)) + float(args["quantity"]),
                },
            )
        )
        self.register(
            Transition(
                name="cancel_reservation",
                description="Release reserved stock back to available",
                preconditions=lambda s: float(s.get("reserved", 0)) >= float(s.get("_qty", 0)),
                effects=lambda s, args: {
                    **s,
                    "available": float(s.get("available", 0)) + float(args["quantity"]),
                    "reserved": float(s.get("reserved", 0)) - float(args["quantity"]),
                },
            )
        )
        self.register(
            Transition(
                name="mark_invoice_paid",
                description="Payment settles an invoice",
                preconditions=lambda s: s.get("payment_status") in ("unpaid", "partial"),
                effects=lambda s, args: {**s, "payment_status": "paid", "paid_amount": args.get("amount")},
            )
        )
        self.register(
            Transition(
                name="mark_damaged",
                description="Reduce usable available stock due to damage",
                preconditions=lambda s: float(s.get("available", 0)) >= float(s.get("_qty", 0)),
                effects=lambda s, args: {
                    **s,
                    "available": float(s.get("available", 0)) - float(args["quantity"]),
                    "damaged": float(s.get("damaged", 0)) + float(args["quantity"]),
                },
            )
        )
        self.register(
            Transition(
                name="post_payment_partial",
                description="Partial payment reduces outstanding without full settlement",
                preconditions=lambda s: s.get("payment_status") in ("unpaid", "partial"),
                effects=lambda s, args: {
                    **s,
                    "payment_status": "partial",
                    "paid_total": float(s.get("paid_total", 0)) + float(args.get("amount") or 0),
                },
            )
        )

    def register(self, transition: Transition) -> None:
        self.transitions[transition.name] = transition

    def predict(self, state: dict[str, Any], action: str, args: dict[str, Any]) -> dict[str, Any]:
        """Return predicted next state (does not mutate reality)."""
        tr = self.transitions.get(action)
        if not tr:
            return {"error": f"unknown_transition:{action}", "state": state}
        probe = {**deepcopy(state), "_qty": args.get("quantity", 0)}
        if not tr.preconditions(probe):
            return {"error": "preconditions_unmet", "state": state, "action": action}
        nxt = tr.effects(deepcopy(state), args)
        return {"state": nxt, "action": action, "predicted": True}

    def observation_model(self, state: dict[str, Any]) -> dict[str, Any]:
        """O(o|s): what observations are expected from a state."""
        return {
            "expected_fields": list(state.keys()),
            "payment_status": state.get("payment_status"),
            "available": state.get("available"),
            "reserved": state.get("reserved"),
        }
