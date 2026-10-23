#!/usr/bin/env python3
"""One-shot launcher for the Cognitive OS shell."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo without install
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognitive_ai.os.shell import launch_os  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Boot the Cognitive OS")
    p.add_argument("--root", default=None, help="OS data directory")
    p.add_argument("--domain", default="animals")
    p.add_argument("--sched", action="store_true", help="Enable background cognition")
    p.add_argument(
        "--demo",
        action="store_true",
        help="Non-interactive demo session",
    )
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    script = None
    if args.demo:
        script = [
            "status",
            "reason",
            "prove tweety can fly",
            "why tweety can fly",
            "goal Feed tweety | tweety fed true",
            "ps",
            "think 4",
            "ask What do you know?",
            "mem 8",
            "dmesg 12",
            "status",
            "shutdown",
        ]

    return launch_os(
        root=args.root,
        domain=args.domain,
        auto_sched=args.sched,
        script=script,
    )


if __name__ == "__main__":
    raise SystemExit(main())
