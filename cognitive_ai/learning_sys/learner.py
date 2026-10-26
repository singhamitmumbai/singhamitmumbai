"""Learning mechanisms with verification — never self-label from unverified predictions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from ..core.records import Claim, EvidenceStatus, LearningExample
from ..core.ids import new_id
from ..memory.store import MemorySystem, Procedure


class LearningSystem:
    def __init__(self, memory: MemorySystem, dataset_dir: Optional[Path | str] = None) -> None:
        self.memory = memory
        self.dataset_dir = Path(dataset_dir) if dataset_dir else Path("data/finderbay/examples")
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.candidates: list[dict[str, Any]] = []
        self.releases: list[dict[str, Any]] = []

    def learn_fact(
        self,
        subject: str,
        relationship: str,
        value: Any,
        *,
        source: str,
        reviewer: str = "user",
    ) -> Claim:
        claim = Claim(
            subject=subject,
            relationship=relationship,
            value=value,
            source=source,
            evidence_status=EvidenceStatus.ASSERTED,
            notes=f"reviewed_by:{reviewer}",
        )
        self.memory.remember_claim(claim)
        if relationship == "alias":
            self.memory.learn_alias(str(value), subject, source=source)
        return claim

    def propose_procedure(
        self,
        name: str,
        steps: list[str],
        examples: list[LearningExample],
        counterexamples: list[LearningExample],
    ) -> dict[str, Any]:
        """Candidate skill: test on independent cases before release."""
        # Toy success: procedure accepted if any positive examples exist and
        # counterexamples do not share identical input hashes with positives.
        pos_hashes = {e.content_hash or self._hash(e.input_data) for e in examples}
        neg_hashes = {e.content_hash or self._hash(e.input_data) for e in counterexamples}
        overlap = pos_hashes & neg_hashes
        candidate = {
            "name": name,
            "steps": steps,
            "status": "rejected" if overlap or not examples else "candidate",
            "positive": len(examples),
            "counterexamples": len(counterexamples),
            "id": new_id("cand"),
        }
        self.candidates.append(candidate)
        if candidate["status"] == "candidate":
            self.memory.add_procedure(
                Procedure(name=name, steps=steps, applicability=["reviewed"], version="candidate")
            )
        return candidate

    def add_example(self, example: LearningExample) -> Path:
        if not example.content_hash:
            example.content_hash = self._hash(example.input_data)
        # Never accept model prediction as its own label without reviewer
        if example.reviewer in ("model", "self", "unverified"):
            raise ValueError("Unverified model predictions cannot become training labels.")
        path = self.dataset_dir / f"{example.example_id}.json"
        path.write_text(json.dumps(example.to_dict(), indent=2, default=str), encoding="utf-8")
        return path

    def continual_cycle(self, train_fn=None) -> dict[str, Any]:
        """Collect → verify → version → train → test → release → monitor → rollback."""
        cycle = {
            "stages": [
                "collect_evidence",
                "verify_labels",
                "version_dataset",
                "train_candidate",
                "test_new_and_old",
                "release",
                "monitor",
                "retain_rollback",
            ],
            "released": False,
            "message": "Cycle scaffold ready; bind train_fn for statistical models.",
        }
        if train_fn is not None:
            metrics = train_fn()
            cycle["metrics"] = metrics
            if metrics.get("regression", 0) <= metrics.get("max_regression", 0.02):
                cycle["released"] = True
                self.releases.append(metrics)
            else:
                cycle["message"] = "Rollback retained — regression exceeded budget."
        return cycle

    @staticmethod
    def _hash(data: Any) -> str:
        blob = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]
