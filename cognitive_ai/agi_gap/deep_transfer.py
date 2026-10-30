"""Deep(er) skill transfer — schema mapping + analogy across domains."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from ..general.transfer import SkillTransfer, TransferResult
from ..memory.store import MemorySystem, Procedure


# Cross-domain schema alignments (source_role → target_role)
DOMAIN_SCHEMAS: dict[str, dict[str, str]] = {
    "finderbay": {
        "project": "project",
        "invoice": "record",
        "material": "resource",
        "warehouse": "location",
        "vendor": "agent",
    },
    "kitchen": {
        "recipe": "procedure",
        "ingredient": "resource",
        "pantry": "location",
        "chef": "agent",
        "dish": "goal",
    },
    "hospital": {
        "patient": "record",
        "medicine": "resource",
        "ward": "location",
        "doctor": "agent",
        "treatment": "procedure",
    },
    "school": {
        "student": "record",
        "course": "procedure",
        "classroom": "location",
        "teacher": "agent",
        "assignment": "goal",
    },
    "coding": {
        "function": "procedure",
        "module": "resource",
        "repo": "location",
        "developer": "agent",
        "test": "verify",
    },
    "lab": {
        "protocol": "procedure",
        "reagent": "resource",
        "bench": "location",
        "scientist": "agent",
        "result": "goal",
    },
    "logistics": {
        "route_plan": "procedure",
        "cargo": "resource",
        "depot": "location",
        "driver": "agent",
        "delivery": "goal",
    },
    "retail": {
        "checkout": "procedure",
        "sku": "resource",
        "store": "location",
        "cashier": "agent",
        "sale": "goal",
    },
    "farm": {
        "planting": "procedure",
        "seed": "resource",
        "field": "location",
        "farmer": "agent",
        "harvest": "goal",
    },
    "construction": {
        "build_plan": "procedure",
        "material": "resource",
        "site": "location",
        "foreman": "agent",
        "structure": "goal",
    },
}


@dataclass
class DeepTransferResult:
    success: bool
    message: str
    mapped_steps: list[str] = field(default_factory=list)
    schema_map: dict[str, str] = field(default_factory=dict)
    analogy: dict[str, str] = field(default_factory=dict)
    new_skill: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeepTransferEngine:
    """Transfer procedures across domains via schema roles + string/entity remap."""

    def __init__(self, memory: Optional[MemorySystem] = None) -> None:
        self.memory = memory or MemorySystem()
        self.shallow = SkillTransfer(self.memory)

    def transfer_project(
        self,
        procedure_name: str,
        *,
        from_project: str,
        to_project: str,
        quantity_scale: float = 1.0,
    ) -> TransferResult:
        return self.shallow.transfer(
            procedure_name,
            from_project=from_project,
            to_project=to_project,
            quantity_scale=quantity_scale,
        )

    def transfer_domain(
        self,
        procedure_name: str,
        *,
        from_domain: str,
        to_domain: str,
        entity_map: Optional[dict[str, str]] = None,
    ) -> DeepTransferResult:
        proc = self.memory.procedures.get(procedure_name)
        if not proc:
            return DeepTransferResult(False, f"Unknown procedure '{procedure_name}'")
        src = DOMAIN_SCHEMAS.get(from_domain, {})
        dst = DOMAIN_SCHEMAS.get(to_domain, {})
        if not src or not dst:
            return DeepTransferResult(
                False,
                f"Need known domains. Have: {sorted(DOMAIN_SCHEMAS)}",
                details={"from_domain": from_domain, "to_domain": to_domain},
            )
        # Invert source roles → find matching target entities by role
        role_map: dict[str, str] = {}
        for ent, role in src.items():
            # find target entity with same role
            for tent, trole in dst.items():
                if trole == role:
                    role_map[ent] = tent
                    break
        entity_map = {**(role_map), **(entity_map or {})}

        mapped = []
        for step in proc.steps:
            s = step
            # longer keys first
            for old in sorted(entity_map.keys(), key=len, reverse=True):
                new = entity_map[old]
                s = re.sub(rf"\b{re.escape(old)}\b", new, s, flags=re.I)
            mapped.append(s)

        new_name = f"{procedure_name}__xfer_{from_domain}_to_{to_domain}"
        self.memory.add_procedure(
            Procedure(
                name=new_name,
                steps=mapped,
                applicability=[f"domain:{to_domain}", "deep_transfer"],
                version="deep_xfer_v1",
            )
        )
        return DeepTransferResult(
            True,
            f"Deep-transferred '{procedure_name}' from {from_domain} → {to_domain} as '{new_name}'.",
            mapped_steps=mapped,
            schema_map=entity_map,
            analogy={f"{a}≃{b}": DOMAIN_SCHEMAS[from_domain].get(a, "?") for a, b in entity_map.items()},
            new_skill=new_name,
            details={"from_domain": from_domain, "to_domain": to_domain},
        )

    def analogize(self, a: str, b: str, c: str) -> dict[str, Any]:
        """a:b :: c:? using domain role tables when possible."""
        # If a,b in same domain schema values/keys
        for domain, table in DOMAIN_SCHEMAS.items():
            if a in table and b in table.values():
                # a maps to role, find c's domain peer with same role as b? simplified
                role_a = table[a]
                # find key in other domains with role_a, paired with c's domain
                pass
            if a in table and c in table:
                role = table[a]
                # find entity in same domain with complementary? use role of b if b is role name
                for ent, role2 in table.items():
                    if ent != a and (b == role2 or b == ent):
                        return {
                            "answer": ent if b == role2 else role2,
                            "method": "schema_role",
                            "domain": domain,
                            "human_level_agi": False,
                        }
        # Cross-domain: same role
        role = None
        for table in DOMAIN_SCHEMAS.values():
            if a in table:
                role = table[a]
                break
        if role:
            for domain, table in DOMAIN_SCHEMAS.items():
                if c in table and table[c] == role:
                    # find peer for b's role in c's domain
                    b_role = None
                    for t in DOMAIN_SCHEMAS.values():
                        if b in t:
                            b_role = t[b]
                            break
                    if b_role:
                        for ent, r in table.items():
                            if r == b_role:
                                return {
                                    "answer": ent,
                                    "method": "cross_domain_role",
                                    "domain": domain,
                                    "role": b_role,
                                    "human_level_agi": False,
                                }
        return {
            "answer": None,
            "method": "unsolved",
            "hint": "Provide entities present in DOMAIN_SCHEMAS",
            "human_level_agi": False,
        }

    def compose_transfer(
        self,
        procedure_name: str,
        *,
        chain: list[tuple[str, str]],
    ) -> DeepTransferResult:
        """Multi-hop domain transfer: A→B→C… composing schema maps."""
        current = procedure_name
        mapped: list[str] = []
        schema_acc: dict[str, str] = {}
        last: Optional[DeepTransferResult] = None
        for frm, to in chain:
            last = self.transfer_domain(current, from_domain=frm, to_domain=to)
            if not last.success:
                return last
            current = last.new_skill or current
            mapped = last.mapped_steps
            schema_acc.update(last.schema_map)
        return DeepTransferResult(
            True,
            f"Composed transfer chain {chain} → '{current}'.",
            mapped_steps=mapped,
            schema_map=schema_acc,
            new_skill=current,
            details={"chain": chain},
        )

    def structural_overlap(self, domain_a: str, domain_b: str) -> dict[str, Any]:
        a = DOMAIN_SCHEMAS.get(domain_a) or {}
        b = DOMAIN_SCHEMAS.get(domain_b) or {}
        roles_a = set(a.values())
        roles_b = set(b.values())
        shared = sorted(roles_a & roles_b)
        return {
            "domain_a": domain_a,
            "domain_b": domain_b,
            "shared_roles": shared,
            "overlap": len(shared) / max(1, len(roles_a | roles_b)),
            "human_level_agi": False,
        }
