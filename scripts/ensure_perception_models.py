#!/usr/bin/env python3
"""Bootstrap offline speech model registry entries (optional download).

Usage:
  python scripts/ensure_perception_models.py
  python scripts/ensure_perception_models.py --download hi-small
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    p = argparse.ArgumentParser(description="Ensure perception model registry is ready")
    p.add_argument("--download", nargs="*", default=[], help="Model names to download if missing")
    p.add_argument("--list", action="store_true", help="List registered models")
    args = p.parse_args()

    from cognitive_ai.perception.model_registry import list_models, ensure_model

    models = list_models()
    print(json.dumps({"models": models, "human_level_agi": False}, indent=2, default=str))
    if args.list and not args.download:
        return 0
    for name in args.download:
        try:
            path = ensure_model(name, download=True)
            print(json.dumps({"ensured": name, "path": str(path), "ok": True}))
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"ensured": name, "ok": False, "error": str(exc)}), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
