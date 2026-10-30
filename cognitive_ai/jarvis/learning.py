"""Offline learning pipeline (blueprint §12) — corrections → dataset → evaluate → activate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..core.ids import new_id
from ..core.records import utc_now
from .nlu import IntentClassifier, INTENT_EXAMPLES


@dataclass
class LearningExample:
    input_text: str
    label: str
    kind: str = "intent"  # intent|slot|retrieval|workflow
    notes: str = ""
    source_permission: str = "user_correction"
    example_id: str = field(default_factory=lambda: new_id("lex"))
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelRelease:
    release_id: str
    kind: str
    metrics: dict[str, Any]
    dataset_hash: str
    code_revision: str = "local"
    active: bool = False
    created_at: str = field(default_factory=utc_now)
    previous_release_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LearningPipeline:
    """Collect corrections → versioned dataset → offline train → compare → activate."""

    def __init__(self, store_dir: str | Path | None = None) -> None:
        self.store_dir = Path(store_dir) if store_dir else None
        if self.store_dir:
            self.store_dir.mkdir(parents=True, exist_ok=True)
        self.examples: list[LearningExample] = []
        self.releases: list[ModelRelease] = []
        self.active_classifier: IntentClassifier = IntentClassifier()
        self.candidate_classifier: Optional[IntentClassifier] = None
        self._shadow_mode = True  # candidate does not change live outcomes

    def collect_correction(
        self,
        input_text: str,
        wrong_label: str,
        right_label: str,
        *,
        kind: str = "intent",
        notes: str = "",
    ) -> LearningExample:
        ex = LearningExample(
            input_text=input_text,
            label=right_label,
            kind=kind,
            notes=f"wrong={wrong_label}; {notes}".strip("; "),
        )
        self.examples.append(ex)
        if self.store_dir:
            path = self.store_dir / "corrections.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ex.to_dict()) + "\n")
        return ex

    def dataset_hash(self) -> str:
        blob = json.dumps([e.to_dict() for e in self.examples], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def train_offline(self) -> IntentClassifier:
        """Merge corrections into intent examples and refit classifier offline."""
        merged = {k: list(v) for k, v in INTENT_EXAMPLES.items()}
        for ex in self.examples:
            if ex.kind != "intent":
                continue
            merged.setdefault(ex.label, []).append(ex.input_text)
        self.candidate_classifier = IntentClassifier(merged)
        self._shadow_mode = True
        return self.candidate_classifier

    def evaluate(self, held_out: list[tuple[str, str]]) -> dict[str, Any]:
        """held_out: list of (text, expected_intent)."""
        active = self.active_classifier
        cand = self.candidate_classifier or active

        def score(clf: IntentClassifier) -> dict[str, Any]:
            ok = 0
            for text, label in held_out:
                pred = clf.predict(text)
                if pred.intent == label:
                    ok += 1
            total = max(1, len(held_out))
            return {"correct": ok, "total": len(held_out), "accuracy": ok / total}

        active_m = score(active)
        cand_m = score(cand)
        improved = cand_m["accuracy"] >= active_m["accuracy"]
        return {
            "active": active_m,
            "candidate": cand_m,
            "improved": improved,
            "shadow_mode": self._shadow_mode,
            "dataset_hash": self.dataset_hash(),
            "example_count": len(self.examples),
        }

    def activate_if_improved(self, held_out: list[tuple[str, str]], *, code_revision: str = "local") -> ModelRelease:
        metrics = self.evaluate(held_out)
        prev = self.releases[-1].release_id if self.releases else None
        release = ModelRelease(
            release_id=new_id("rel"),
            kind="intent_classifier",
            metrics=metrics,
            dataset_hash=self.dataset_hash(),
            code_revision=code_revision,
            active=False,
            previous_release_id=prev,
        )
        if metrics["improved"] and self.candidate_classifier is not None:
            self.active_classifier = self.candidate_classifier
            self._shadow_mode = False
            release.active = True
        self.releases.append(release)
        if self.store_dir:
            (self.store_dir / "releases.jsonl").open("a", encoding="utf-8").write(
                json.dumps(release.to_dict()) + "\n"
            )
        return release

    def rollback(self) -> Optional[ModelRelease]:
        if len(self.releases) < 2:
            return None
        # Refit from base + examples excluding last activation batch is complex;
        # restore base classifier and mark rollback.
        self.active_classifier = IntentClassifier()
        self.candidate_classifier = None
        self._shadow_mode = True
        rb = ModelRelease(
            release_id=new_id("rel"),
            kind="intent_classifier",
            metrics={"rolled_back": True},
            dataset_hash=self.dataset_hash(),
            active=True,
            previous_release_id=self.releases[-1].release_id,
        )
        self.releases.append(rb)
        return rb

    def status(self) -> dict[str, Any]:
        return {
            "examples": len(self.examples),
            "releases": len(self.releases),
            "shadow_mode": self._shadow_mode,
            "dataset_hash": self.dataset_hash(),
            "active_release": next((r.to_dict() for r in reversed(self.releases) if r.active), None),
        }
