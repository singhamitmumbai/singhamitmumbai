#!/usr/bin/env python3
"""Offline demo of computer vision wired into CompleteNonLLMAI (no LLM)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognitive_ai.system import CompleteNonLLMAI
from cognitive_ai.vision import VisionSystem, list_projects


def main() -> int:
    print("=" * 60)
    print("Computer vision demo — YOLO_Projects + CompleteNonLLMAI")
    print("=" * 60)
    vs = VisionSystem()
    print("\n--- Vision status ---")
    print(json.dumps(vs.status(), indent=2, default=str)[:1500])
    print(f"\nVendored projects: {len(list_projects())}")

    ai = CompleteNonLLMAI(seed_demo=True)
    for q in (
        "Vision status",
        "Count people in image ./crowd_people.jpg",
        "Detect vehicles in image ./traffic_cars.jpg",
        "Detect fire in image ./fire_smoke.jpg",
        "Detect objects in image ./scene_photo.jpg",
    ):
        print("-" * 40)
        print("USER:", q)
        out = ai.handle(q)
        print("AI:  ", out["text"])
        print(" op= ", (out.get("frame") or {}).get("operation"))

    print("\nDemo complete. No LLM was used.")
    print("For real CNN weights: pip install -e '.[vision]'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
