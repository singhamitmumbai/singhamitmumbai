"""python -m cognitive_ai.os"""

from .shell import launch_os
import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(prog="cognitive_ai.os", description="Boot Cognitive OS")
    p.add_argument("--root", default=None, help="OS store directory (default ~/.cognitive_os)")
    p.add_argument("--domain", default="animals", help="Initial knowledge domain")
    p.add_argument("--sched", action="store_true", help="Start background cognition on boot")
    p.add_argument(
        "--demo",
        action="store_true",
        help="Run a short non-interactive OS demo session then exit",
    )
    args = p.parse_args(argv)

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
            "ask What are you trying to do?",
            "ask Who are you?",
            "mem 10",
            "dmesg 15",
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
