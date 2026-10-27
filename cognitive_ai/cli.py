"""Interactive REPL and CLI for the symbolic cognitive brain."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Optional

from .brain import Brain
from .knowledge import Rule, Triple
from .version import __version__


HELP = """
Commands:
  help                         Show this help
  domain <name>                Load domain: animals | blocks | kitchen
  assert <s> <p> <o>           Assert a fact (also believed)
  rule <name> | <ants> => <cons>
                               Add rule. ants/cons are triples separated by ','
                               Example: rule r1 | (?x is_a bird), (?x has wings) => (?x can fly)
  goal <desc> | <s> <p> <o>    Set a goal
  think [N]                    Run N think cycles (default 5)
  reason                       Forward-chain once
  prove <s> <p> <o>            Backward-chain prove
  ask <question>               Introspective / factual question
  why <s> <p> <o>              Explain a conclusion
  query [<s> [<p> [<o>]]]      Query facts (* wildcards)
  dump [path]                  Dump mind state (optional JSON path)
  save <path>                  Persist knowledge base
  load <path>                  Load knowledge base
  lessons                      Show recent learning lessons
  thoughts                     Show recent thoughts
  quit / exit                  Leave REPL
""".strip()


def _parse_triples(blob: str) -> list[Triple]:
    """Parse '(a b c), (d e f)' into Triples."""
    import re

    triples: list[Triple] = []
    for m in re.finditer(r"\((\S+)\s+(\S+)\s+(\S+)\)", blob):
        triples.append(Triple(m.group(1), m.group(2), m.group(3)))
    return triples


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cognitive_ai",
        description="Symbolic cognitive AI (no LLM) — REPL, demo, and Cognitive OS.",
    )
    p.add_argument("--version", action="version", version=f"cognitive_ai {__version__}")
    p.add_argument(
        "command",
        nargs="?",
        default=None,
        choices=[
            "os",
            "demo",
            "repl",
            "chat",
            "ui",
            "finderbay",
            "complete",
            "general",
            "eval",
            "bench",
            "api",
            "blueprint",
            "agent",
            "code",
            "vision",
        ],
        help="Optional subcommand: os, chat/ui, demo, repl, finderbay/complete, general, eval, bench, api, blueprint, agent, code, vision",
    )
    p.add_argument(
        "--image",
        type=str,
        default=None,
        help="Image/video path for vision subcommand",
    )
    p.add_argument(
        "--task",
        type=str,
        default="detect_objects",
        help="Vision task: detect_objects|detect_vehicles|count_people|detect_fire|scan_document|vision_status|...",
    )
    p.add_argument(
        "--ask",
        type=str,
        default=None,
        help="One-shot question for the Finderbay complete system",
    )
    p.add_argument("--host", default="0.0.0.0", help="Chat UI bind host")
    p.add_argument("--port", type=int, default=8765, help="Chat UI port")
    p.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optional path to JSON/SQLite knowledge store",
    )
    p.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Load a built-in domain at startup",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Run the offline demo then exit",
    )
    p.add_argument(
        "--os-demo",
        action="store_true",
        help="When launching OS: run a scripted OS demo session",
    )
    p.add_argument(
        "--root",
        type=str,
        default=None,
        help="Cognitive OS store directory (default ~/.cognitive_os)",
    )
    p.add_argument(
        "--sched",
        action="store_true",
        help="Start background cognition scheduler on OS boot",
    )
    p.add_argument(
        "--think",
        type=int,
        default=None,
        metavar="N",
        help="Run N think cycles non-interactively then dump summary",
    )
    p.add_argument(
        "--goal",
        type=str,
        default=None,
        help='Goal as "description|s p o"',
    )
    p.add_argument(
        "--repl",
        action="store_true",
        help="Force interactive REPL",
    )
    return p


def run_demo(brain: Optional[Brain] = None) -> int:
    """Offline demo: animals domain, reason, plan, self-explain."""
    brain = brain or Brain(name="DemoBrain")
    print("=" * 60)
    print("Symbolic Cognitive AI — offline demo (zero network / LLM)")
    print("=" * 60)

    brain.load_domain("animals")
    print(f"\nLoaded domain: {brain.kb.domain} ({len(brain.kb)} facts, {len(brain.kb.rules())} rules)")

    print("\n--- Forward reasoning ---")
    trace = brain.reason()
    print(trace["explanation"])
    for d in trace["derived"][:12]:
        print(f"  derived: {d}")

    print("\n--- Prove: (tweety can fly)? ---")
    proof = brain.prove("tweety", "can", "fly")
    print(proof["explanation"])
    print("success:", proof["success"])

    print("\n--- Why ---")
    print(brain.why("tweety", "can", "fly"))

    print("\n--- Set goal: feed tweety ---")
    # Ensure tweety needs food via inheritance then feed
    brain.reason()
    brain.set_goal("Feed tweety", "tweety", "fed", "true", priority=1.0)
    report = brain.think(steps=4)
    print(f"Think stopped: {report.stopped_reason}")
    for c in report.cycles:
        print(f"  cycle {c.step}: plan={c.plan_summary}")
        for a in c.actions:
            print(f"    {a}")
        for u in c.utterances:
            print(f"    say: {u}")

    print("\n--- Self-understanding ---")
    for q in (
        "Who are you?",
        "What do you know?",
        "What are you trying to do?",
        "How do you think?",
        "What are your limits?",
    ):
        print(f"\nQ: {q}")
        print(f"A: {brain.ask(q)}")

    print("\n--- Mind summary ---")
    summary = brain.kb.summary()
    print(json.dumps(summary, indent=2))
    print(f"Open goals: {len(brain.planner.open_goals())}")
    print(f"Thoughts recorded: {len(brain.self_model.thoughts)}")
    print("\nDemo complete. No LLM was used.")
    return 0


def handle_line(brain: Brain, line: str) -> bool:
    """Handle one REPL line. Returns False to exit."""
    line = line.strip()
    if not line:
        return True
    try:
        parts = shlex.split(line)
    except ValueError as e:
        print(f"parse error: {e}")
        return True
    cmd = parts[0].lower()
    args = parts[1:]

    if cmd in ("quit", "exit", "q"):
        return False
    if cmd == "help":
        print(HELP)
        return True
    if cmd == "domain":
        if not args:
            print("usage: domain <animals|blocks|kitchen>")
            return True
        brain.load_domain(args[0])
        print(f"Loaded domain '{args[0]}' ({len(brain.kb)} facts).")
        return True
    if cmd == "assert":
        if len(args) < 3:
            print("usage: assert <s> <p> <o>")
            return True
        brain.assert_fact(args[0], args[1], args[2], learn=True)
        print(f"OK ({args[0]} {args[1]} {args[2]})")
        return True
    if cmd == "rule":
        raw = line[len(parts[0]) :].strip()
        if "|" not in raw or "=>" not in raw:
            print("usage: rule <name> | (a b c), ... => (d e f), ...")
            return True
        name_part, rest = raw.split("|", 1)
        name = name_part.strip() or "user_rule"
        ants_s, cons_s = rest.split("=>", 1)
        ants = _parse_triples(ants_s)
        cons = _parse_triples(cons_s)
        if not ants or not cons:
            print("Need at least one antecedent and one consequent triple.")
            return True
        rule = Rule(name=name, antecedents=ants, consequents=cons, learned=True)
        brain.add_rule(rule)
        brain.learner.learn_rule(rule)
        print(f"Rule '{name}' added.")
        return True
    if cmd == "goal":
        raw = line[len(parts[0]) :].strip()
        if "|" not in raw:
            print('usage: goal <description> | <s> <p> <o>')
            return True
        desc, trip = raw.split("|", 1)
        tparts = trip.split()
        if len(tparts) < 3:
            print("Need subject predicate object after |")
            return True
        g = brain.set_goal(desc.strip(), tparts[0], tparts[1], tparts[2])
        print(f"Goal {g.goal_id}: {g.description}")
        return True
    if cmd == "think":
        n = int(args[0]) if args else 5
        report = brain.think(steps=n)
        print(f"Stopped: {report.stopped_reason}")
        for c in report.cycles:
            print(f"[cycle {c.step}] {c.plan_summary}")
            for u in c.utterances:
                print(f"  {u}")
        return True
    if cmd == "reason":
        print(json.dumps(brain.reason(), indent=2))
        return True
    if cmd == "prove":
        if len(args) < 3:
            print("usage: prove <s> <p> <o>")
            return True
        print(json.dumps(brain.prove(args[0], args[1], args[2]), indent=2))
        return True
    if cmd == "ask":
        q = line[len(parts[0]) :].strip().strip('"').strip("'")
        print(brain.ask(q))
        return True
    if cmd == "why":
        if len(args) < 3:
            print("usage: why <s> <p> <o>")
            return True
        print(brain.why(args[0], args[1], args[2]))
        return True
    if cmd == "query":
        s = args[0] if len(args) > 0 else "*"
        p = args[1] if len(args) > 1 else "*"
        o = args[2] if len(args) > 2 else "*"
        hits = brain.kb.query(s, p, o)
        if not hits:
            print("(no matches)")
        for t in hits:
            print(t)
        return True
    if cmd == "dump":
        data = brain.dump_mind()
        if args:
            Path(args[0]).write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"Wrote {args[0]}")
        else:
            # Compact view
            print(json.dumps({
                "knowledge": data["knowledge"],
                "goals": data["goals"],
                "working_memory": {
                    "situation": data["working_memory"]["situation"],
                    "belief_count": len(data["working_memory"]["beliefs"]),
                    "open_questions": data["working_memory"]["open_questions"],
                },
                "recent_thoughts": data["self_model"]["recent_thoughts"][-8:],
            }, indent=2))
        return True
    if cmd == "save":
        if not args:
            print("usage: save <path>")
            return True
        path = brain.save(args[0])
        print(f"Saved {path}")
        return True
    if cmd == "load":
        if not args:
            print("usage: load <path>")
            return True
        brain.load(args[0])
        print(f"Loaded {args[0]} ({len(brain.kb)} facts)")
        return True
    if cmd == "lessons":
        for lesson in brain.learner.recent_lessons():
            print(f"- [{lesson.kind}] {lesson.content}")
        return True
    if cmd == "thoughts":
        for t in brain.self_model.recent_thoughts(15):
            print(f"- [{t.kind}] {t.content}")
        return True

    print(f"Unknown command '{cmd}'. Type 'help'.")
    return True


def repl(brain: Brain) -> int:
    print(f"cognitive_ai {__version__} — symbolic brain REPL (no LLM)")
    print("Type 'help' for commands, 'demo' via: python -m cognitive_ai --demo")
    if brain.kb.domain and brain.kb.domain != "empty":
        print(f"Domain: {brain.kb.domain} ({len(brain.kb)} facts)")
    while True:
        try:
            line = input("brain> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not handle_line(brain, line):
            break
    return 0


def run_finderbay_demo() -> int:
    """End-to-end Finderbay complete non-LLM AI demo (blueprint §20)."""
    from .system import CompleteNonLLMAI

    ai = CompleteNonLLMAI(seed_demo=True)
    print("=" * 60)
    print(f"Complete Non-LLM AI — Finderbay demo (v{ai.version})")
    print("No pretrained general-purpose LLM at runtime.")
    print("=" * 60)
    print(json.dumps(ai.status(), indent=2, default=str))

    queries = [
        "Show unpaid invoices above ₹1 lakh for Project A, excluding disputed ones.",
        (
            "From this week's delivery records, find the items still pending for "
            "Project A and prepare a follow-up. Do not send it."
        ),
        "Dikhao pending deliveries for Project A kal",
        "Show material stock and inventory",
        "Reserve 15 bags of cement",
        "Create task for Project A: Order safety helmets",
        "List tasks for Project A",
        "What is the amount due on the invoice scan document?",
        "Warehouse B is now called Central Store.",
        "Show overdue invoices for Project A",
        "What are your capabilities?",
    ]
    for q in queries:
        print("\n" + "-" * 60)
        print(f"USER: {q}")
        result = ai.handle(q)
        print(f"FRAME: {result['frame']['operation']} conf={result['frame']['confidence']:.2f}")
        print(f"PLAN:  {result['plan']['outcome']} — {result['plan']['explanation']}")
        print(f"AI:    {result['text']}")
        if result.get("structured"):
            print(f"DATA:  {json.dumps(result['structured'], default=str)}")

    print("\n--- Independent evaluation suite ---")
    report = ai.evaluate()
    print(json.dumps(report.to_dict(), indent=2, default=str))
    print(f"\nAccuracy: {report.accuracy:.0%} ({report.passed}/{report.passed + report.failed})")
    print("Demo complete. No LLM was used.")
    return 0 if report.failed == 0 else 1


def finderbay_repl() -> int:
    from .system import CompleteNonLLMAI

    ai = CompleteNonLLMAI(seed_demo=True)
    print(f"Finderbay CompleteNonLLMAI {ai.version} — type 'quit' to exit, 'eval' to run suite")
    while True:
        try:
            line = input("finderbay> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        line = line.strip()
        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            break
        if line.lower() == "eval":
            print(json.dumps(ai.evaluate().to_dict(), indent=2, default=str))
            continue
        if line.lower() == "status":
            print(json.dumps(ai.status(), indent=2, default=str))
            continue
        result = ai.handle(line)
        print(result["text"])
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)

    # Complete Finderbay system
    if args.command in ("finderbay", "complete"):
        if args.ask:
            from .system import CompleteNonLLMAI

            ai = CompleteNonLLMAI(seed_demo=True)
            result = ai.handle(args.ask)
            print(result["text"])
            return 0
        if args.demo or not sys.stdin.isatty():
            return run_finderbay_demo()
        return finderbay_repl()

    if args.command == "general":
        from .general.agent import GeneralNonLLMAI

        ai = GeneralNonLLMAI(seed_demo=True)
        if args.ask:
            print(ai.handle(args.ask)["text"])
            return 0
        if args.demo or not sys.stdin.isatty():
            print("=" * 60)
            print(f"General Non-LLM AI demo (v{ai.version})")
            print(json.dumps(ai.status(), indent=2, default=str)[:2000])
            for q in (
                "Hello",
                "Can penguins fly?",
                "What is a knife used for?",
                "dog:bark :: cat:?",
                "Show unpaid invoices above ₹1 lakh for Project A, excluding disputed ones.",
                "Are you ChatGPT?",
            ):
                print("-" * 40)
                print("USER:", q)
                print("AI:  ", ai.ask(q))
            print("Demo complete. No LLM was used.")
            return 0
        print(f"GeneralNonLLMAI {ai.version} — Finderbay + open-domain. Type quit to exit.")
        while True:
            try:
                line = input("general> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if line.strip().lower() in ("quit", "exit", "q"):
                break
            if line.strip().lower() == "status":
                print(json.dumps(ai.status(), indent=2, default=str))
                continue
            print(ai.ask(line))
        return 0

    if args.command == "eval":
        from .system import CompleteNonLLMAI

        report = CompleteNonLLMAI(seed_demo=True).evaluate()
        print(json.dumps(report.to_dict(), indent=2, default=str))
        return 0 if report.failed == 0 else 1

    if args.command == "bench":
        from .benchmarks import run_scan_benchmark, run_transfer_benchmark
        from .commonsense.eval_suite import run_commonsense_eval
        from .nlu.pipeline import LanguageUnderstanding
        from .perception.model_registry import list_models

        payload = {
            "commonsense": run_commonsense_eval(),
            "scan": run_scan_benchmark(LanguageUnderstanding().parse),
            "transfer": run_transfer_benchmark(),
            "speech_models": list_models(),
        }
        print(json.dumps(payload, indent=2, default=str))
        ok = (
            payload["commonsense"]["failed"] == 0
            and payload["transfer"]["passed"]
            and payload["scan"]["by_split"]["train_like"]["passed"] >= 1
        )
        return 0 if ok else 1

    if args.command == "api":
        from .api.server import serve_api
        from .version import __version__

        host = args.host if args.host != "0.0.0.0" else "127.0.0.1"
        port = 8787 if args.port == 8765 else args.port
        mode = "finderbay" if (args.domain or "") in ("finderbay", "complete") else "general"
        httpd = serve_api(host, port, mode=mode)
        print(f"API http://{host}:{port}/  (v{__version__}, mode={mode}, human_level_agi=false)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()
        return 0

    if args.command == "blueprint":
        from .blueprint.audit import run_blueprint_audit

        print(json.dumps(run_blueprint_audit(), indent=2, default=str))
        return 0

    if args.command == "agent":
        from .agent import FinderbayAgentSystem

        system = FinderbayAgentSystem(seed_demo=True)
        if args.ask:
            # Treat --ask as a batch of tasks separated by ' || '
            tasks = [t.strip() for t in args.ask.split("||") if t.strip()]
            report = system.auto(tasks)
            print(json.dumps(report, indent=2, default=str))
            return 0 if report.get("failed", 0) == 0 else 1
        if args.demo:
            report = system.auto(
                [
                    "Show unpaid invoices for Project A",
                    "Show alerts for Project A",
                    "Show purchase orders for Project A",
                    "generate function add(a, b)",
                ]
            )
            print(json.dumps({"status": system.status(), "report": report}, indent=2, default=str))
            return 0
        print(json.dumps(system.status(), indent=2, default=str))
        return 0

    if args.command == "code":
        from .coding import CodingSystem

        coding = CodingSystem()
        if args.ask:
            print(json.dumps(coding.handle(args.ask), indent=2, default=str))
            return 0
        print(json.dumps(coding.status(), indent=2, default=str))
        return 0

    if args.command == "vision":
        from .system import CompleteNonLLMAI
        from .vision import VisionSystem, list_projects

        if args.task == "vision_status" or (not args.image and not args.ask):
            vs = VisionSystem()
            payload = {"status": vs.status(), "projects": list_projects()}
            print(json.dumps(payload, indent=2, default=str))
            return 0
        if args.ask:
            ai = CompleteNonLLMAI(seed_demo=True)
            print(json.dumps(ai.handle(args.ask), indent=2, default=str)[:4000])
            print()
            print(ai.handle(args.ask)["text"])
            return 0
        if not args.image:
            print("vision: provide --image PATH or --task vision_status", file=sys.stderr)
            return 2
        ai = CompleteNonLLMAI(seed_demo=False)
        ask = f"{args.task.replace('_', ' ')} in image {args.image}"
        result = ai.handle(ask)
        print(result["text"])
        print(json.dumps(result.get("structured") or {}, indent=2, default=str)[:3000])
        return 0

    # Chat UI launcher
    if args.command in ("chat", "ui"):
        from .ui.server import serve_chat

        serve_chat(
            host=args.host,
            port=args.port,
            domain=args.domain or "finderbay",
        )
        return 0

    # Cognitive OS launcher
    if args.command == "os":
        from .os.shell import launch_os

        script = None
        if args.os_demo or args.demo:
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
            domain=args.domain or "animals",
            auto_sched=args.sched,
            script=script,
        )

    brain = Brain(storage=args.storage)
    if args.domain:
        brain.load_domain(args.domain)

    if args.demo or args.command == "demo":
        return run_demo(brain)

    if args.goal:
        if not args.domain:
            brain.load_domain("animals")
        if "|" not in args.goal:
            print("--goal must be 'description|s p o'", file=sys.stderr)
            return 2
        desc, trip = args.goal.split("|", 1)
        parts = trip.split()
        if len(parts) < 3:
            print("--goal needs s p o after |", file=sys.stderr)
            return 2
        brain.set_goal(desc.strip(), parts[0], parts[1], parts[2])

    if args.think is not None:
        if not args.domain and brain.kb.domain == "empty":
            brain.load_domain("animals")
        report = brain.think(steps=args.think)
        print(json.dumps(report.to_dict(), indent=2))
        return 0

    # Default: REPL (or demo if stdin is not a TTY and no flags)
    if args.repl or args.command == "repl" or sys.stdin.isatty():
        return repl(brain)

    # Non-interactive stdin: treat lines as commands
    for line in sys.stdin:
        if not handle_line(brain, line):
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
