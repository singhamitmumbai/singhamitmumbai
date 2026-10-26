"""Cognitive OS shell — interactive terminal UI."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import Optional, TextIO

from ..knowledge import Rule, Triple
from .system import CognitiveOS

HELP = """
Cognitive OS commands
─────────────────────
  boot [domain]              Boot brain + services (optional domain)
  shutdown                   Persist knowledge and halt
  status                     System / brain / scheduler status
  help                       This help

  think [N]                  Run N think cycles (default 3)
  ask <question>             Ask the brain (introspection / facts)
  assert <s> <p> <o>         Assert a fact into knowledge + beliefs
  retract <s> <p> <o>        Remove a fact
  goal <desc> | <s> <p> <o>  Set a goal process
  kill <goal_id>             Abandon a goal
  ps                         List goals (process table)
  mem [N]                    Dump knowledge / beliefs
  why <s> <p> <o>            Explain a conclusion
  reason                     Forward-chain once
  prove <s> <p> <o>          Backward-chain prove
  query [s [p [o]]]          Query triples (* wildcards)
  domain <name>              Switch domain: animals|blocks|kitchen

  sched start|stop|pause|resume|status|tick [N]
                             Background cognition scheduler
  dmesg [N]                  Show console log
  save                       Persist knowledge to OS store
  load                       Reload knowledge from OS store
  dump [path]                Dump full mind JSON
  lessons                    Recent learning lessons
  thoughts                   Recent thoughts
  clear                      Clear console buffer

  echo <text>                Print text
  version                    Show version
  exit / quit / halt         Shutdown and leave the shell
""".strip()


def _parse_triples(blob: str) -> list[Triple]:
    import re

    return [
        Triple(m.group(1), m.group(2), m.group(3))
        for m in re.finditer(r"\((\S+)\s+(\S+)\s+(\S+)\)", blob)
    ]


class OSShell:
    """Line-oriented shell bound to a CognitiveOS instance."""

    def __init__(self, cos: CognitiveOS, out: Optional[TextIO] = None) -> None:
        self.cos = cos
        self.out = out or sys.stdout
        self.running = True

    def emit(self, text: str = "") -> None:
        print(text, file=self.out)

    def handle(self, line: str) -> bool:
        """Dispatch one command. Returns False to exit the shell."""
        line = line.strip()
        if not line:
            return True
        self.cos.console.info(f"$ {line}", source="shell")
        try:
            parts = shlex.split(line)
        except ValueError as e:
            self.emit(f"parse error: {e}")
            return True

        cmd = parts[0].lower()
        args = parts[1:]

        # exit family
        if cmd in ("exit", "quit", "halt", "q"):
            self.emit(self.cos.shutdown(persist=True))
            self.running = False
            return False

        handlers = {
            "help": self._help,
            "?": self._help,
            "boot": self._boot,
            "shutdown": self._shutdown_only,
            "status": self._status,
            "think": self._think,
            "ask": self._ask,
            "assert": self._assert,
            "retract": self._retract,
            "goal": self._goal,
            "kill": self._kill,
            "ps": self._ps,
            "mem": self._mem,
            "why": self._why,
            "reason": self._reason,
            "prove": self._prove,
            "query": self._query,
            "domain": self._domain,
            "sched": self._sched,
            "dmesg": self._dmesg,
            "save": self._save,
            "load": self._load,
            "dump": self._dump,
            "lessons": self._lessons,
            "thoughts": self._thoughts,
            "clear": self._clear,
            "echo": self._echo,
            "version": self._version,
            "rule": self._rule,
        }
        fn = handlers.get(cmd)
        if fn is None:
            self.emit(f"unknown command '{cmd}'. Type 'help'.")
            return True
        try:
            fn(line, args)
        except Exception as exc:  # noqa: BLE001
            self.cos.console.error(str(exc), source="shell")
            self.emit(f"error: {exc}")
        return True

    # --- commands -------------------------------------------------------------

    def _help(self, line: str, args: list[str]) -> None:
        self.emit(HELP)

    def _boot(self, line: str, args: list[str]) -> None:
        domain = args[0] if args else None
        self.emit(self.cos.boot(domain=domain))

    def _shutdown_only(self, line: str, args: list[str]) -> None:
        self.emit(self.cos.shutdown(persist=True))

    def _status(self, line: str, args: list[str]) -> None:
        self._need_boot()
        self.emit(self.cos.status_text())

    def _think(self, line: str, args: list[str]) -> None:
        self._need_boot()
        n = int(args[0]) if args else 3
        report = self.cos.brain.think(steps=n)
        self.emit(f"think → {report.stopped_reason} ({len(report.cycles)} cycles)")
        for c in report.cycles:
            self.emit(f"  [{c.step}] {c.plan_summary}")
            for a in c.actions:
                self.emit(f"    act: {a}")
            for u in c.utterances:
                self.emit(f"    say: {u}")
            for r in c.reflections:
                self.emit(f"    note: {r}")

    def _ask(self, line: str, args: list[str]) -> None:
        self._need_boot()
        q = line.split(None, 1)
        if len(q) < 2:
            self.emit("usage: ask <question>")
            return
        self.emit(self.cos.brain.ask(q[1].strip().strip('"').strip("'")))

    def _assert(self, line: str, args: list[str]) -> None:
        self._need_boot()
        if len(args) < 3:
            self.emit("usage: assert <s> <p> <o>")
            return
        self.cos.brain.assert_fact(args[0], args[1], args[2], learn=True)
        self.cos.console.info(f"asserted ({args[0]} {args[1]} {args[2]})", source="mem")
        self.emit(f"ok ({args[0]} {args[1]} {args[2]})")

    def _retract(self, line: str, args: list[str]) -> None:
        self._need_boot()
        if len(args) < 3:
            self.emit("usage: retract <s> <p> <o>")
            return
        ok = self.cos.brain.retract_fact(args[0], args[1], args[2])
        self.emit("ok" if ok else "not found")

    def _goal(self, line: str, args: list[str]) -> None:
        self._need_boot()
        raw = line.split(None, 1)[1] if " " in line else ""
        if "|" not in raw:
            self.emit('usage: goal <description> | <s> <p> <o>')
            return
        desc, trip = raw.split("|", 1)
        tparts = trip.split()
        if len(tparts) < 3:
            self.emit("need subject predicate object after |")
            return
        g = self.cos.brain.set_goal(desc.strip(), tparts[0], tparts[1], tparts[2])
        self.cos.console.info(f"goal {g.goal_id}: {g.description}", source="sched")
        self.emit(f"started goal {g.goal_id}: {g.description} → {g.target}")

    def _kill(self, line: str, args: list[str]) -> None:
        self._need_boot()
        if not args:
            self.emit("usage: kill <goal_id>")
            return
        ok = self.cos.brain.abandon_goal(args[0])
        self.emit("abandoned" if ok else "no such goal")

    def _ps(self, line: str, args: list[str]) -> None:
        self._need_boot()
        self.emit(self.cos.ps_text())

    def _mem(self, line: str, args: list[str]) -> None:
        self._need_boot()
        n = int(args[0]) if args else 25
        self.emit(self.cos.mem_text(limit=n))

    def _why(self, line: str, args: list[str]) -> None:
        self._need_boot()
        if len(args) < 3:
            self.emit("usage: why <s> <p> <o>")
            return
        self.emit(self.cos.brain.why(args[0], args[1], args[2]))

    def _reason(self, line: str, args: list[str]) -> None:
        self._need_boot()
        trace = self.cos.brain.reason()
        self.emit(trace["explanation"])
        for d in trace.get("derived", [])[:20]:
            self.emit(f"  {d}")

    def _prove(self, line: str, args: list[str]) -> None:
        self._need_boot()
        if len(args) < 3:
            self.emit("usage: prove <s> <p> <o>")
            return
        proof = self.cos.brain.prove(args[0], args[1], args[2])
        self.emit(f"success={proof['success']}: {proof['explanation']}")

    def _query(self, line: str, args: list[str]) -> None:
        self._need_boot()
        s = args[0] if len(args) > 0 else "*"
        p = args[1] if len(args) > 1 else "*"
        o = args[2] if len(args) > 2 else "*"
        hits = self.cos.brain.kb.query(s, p, o)
        if not hits:
            self.emit("(no matches)")
            return
        for t in hits:
            self.emit(str(t))

    def _domain(self, line: str, args: list[str]) -> None:
        self._need_boot()
        if not args:
            self.emit(f"current domain: {self.cos.brain.kb.domain}")
            return
        self.cos.brain.load_domain(args[0])
        self.emit(f"domain → {args[0]} ({len(self.cos.brain.kb)} facts)")

    def _sched(self, line: str, args: list[str]) -> None:
        self._need_boot()
        if not args:
            self.emit(json.dumps(self.cos.scheduler.status(), indent=2))
            return
        sub = args[0].lower()
        if sub == "start":
            self.cos.scheduler.start()
            self.emit("scheduler started")
        elif sub == "stop":
            self.cos.scheduler.stop()
            self.emit("scheduler stopped")
        elif sub == "pause":
            self.cos.scheduler.pause()
            self.emit("scheduler paused")
        elif sub == "resume":
            self.cos.scheduler.resume()
            self.emit("scheduler resumed")
        elif sub == "status":
            self.emit(json.dumps(self.cos.scheduler.status(), indent=2))
        elif sub == "tick":
            n = int(args[1]) if len(args) > 1 else 1
            report = self.cos.scheduler.tick_once(n)
            reason = getattr(report, "stopped_reason", report)
            self.emit(f"tick → {reason}")
        else:
            self.emit("usage: sched start|stop|pause|resume|status|tick [N]")

    def _dmesg(self, line: str, args: list[str]) -> None:
        n = int(args[0]) if args else 20
        self.emit(self.cos.dmesg_text(n))

    def _save(self, line: str, args: list[str]) -> None:
        self._need_boot()
        path = self.cos.store.save_knowledge(self.cos.brain)
        self.cos._persist_state()
        self.emit(f"saved → {path}")

    def _load(self, line: str, args: list[str]) -> None:
        ok = self.cos.store.load_knowledge(self.cos.brain)
        self.emit(
            f"loaded ({len(self.cos.brain.kb)} facts)"
            if ok
            else "no knowledge file on disk"
        )

    def _dump(self, line: str, args: list[str]) -> None:
        self._need_boot()
        if args:
            Path(args[0]).write_text(
                self.cos.brain.dump_mind_json(), encoding="utf-8"
            )
            self.emit(f"wrote {args[0]}")
        else:
            path = self.cos.store.dump_mind(self.cos.brain)
            self.emit(f"wrote {path}")

    def _lessons(self, line: str, args: list[str]) -> None:
        self._need_boot()
        lessons = self.cos.brain.learner.recent_lessons()
        if not lessons:
            self.emit("(no lessons)")
            return
        for lesson in lessons:
            self.emit(f"- [{lesson.kind}] {lesson.content}")

    def _thoughts(self, line: str, args: list[str]) -> None:
        self._need_boot()
        thoughts = self.cos.brain.self_model.recent_thoughts(20)
        if not thoughts:
            self.emit("(no thoughts)")
            return
        for t in thoughts:
            self.emit(f"- [{t.kind}] {t.content}")

    def _clear(self, line: str, args: list[str]) -> None:
        self.cos.console.clear()
        self.emit("console cleared")

    def _echo(self, line: str, args: list[str]) -> None:
        self.emit(" ".join(args))

    def _version(self, line: str, args: list[str]) -> None:
        self.emit(f"Cognitive OS {self.cos.version}")

    def _rule(self, line: str, args: list[str]) -> None:
        self._need_boot()
        raw = line.split(None, 1)[1] if " " in line else ""
        if "|" not in raw or "=>" not in raw:
            self.emit("usage: rule <name> | (a b c), ... => (d e f), ...")
            return
        name_part, rest = raw.split("|", 1)
        name = name_part.strip() or "user_rule"
        ants_s, cons_s = rest.split("=>", 1)
        ants = _parse_triples(ants_s)
        cons = _parse_triples(cons_s)
        if not ants or not cons:
            self.emit("need antecedents and consequents")
            return
        rule = Rule(name=name, antecedents=ants, consequents=cons, learned=True)
        self.cos.brain.add_rule(rule)
        self.cos.brain.learner.learn_rule(rule)
        self.emit(f"rule '{name}' installed")

    def _need_boot(self) -> None:
        if not self.cos.booted:
            splash = self.cos.boot()
            self.emit(splash)

    def repl(self, auto_boot: bool = True) -> int:
        self.emit(self.cos.welcome())
        if auto_boot and not self.cos.booted:
            self.emit(self.cos.boot())
        while self.running:
            try:
                line = input("cos> ")
            except (EOFError, KeyboardInterrupt):
                self.emit("")
                self.emit(self.cos.shutdown(persist=True))
                break
            if not self.handle(line):
                break
        return 0

    def run_script(self, lines: list[str], auto_boot: bool = True) -> int:
        """Non-interactive session for demos/tests."""
        self.emit(self.cos.welcome())
        if auto_boot and not self.cos.booted:
            self.emit(self.cos.boot())
        for line in lines:
            self.emit(f"cos> {line}")
            if not self.handle(line):
                break
        if self.cos.booted:
            self.emit(self.cos.shutdown(persist=True))
        return 0


def launch_os(
    root: Optional[str] = None,
    domain: str = "animals",
    auto_sched: bool = False,
    script: Optional[list[str]] = None,
) -> int:
    cos = CognitiveOS(root=root, domain=domain, auto_sched=auto_sched)
    shell = OSShell(cos)
    if script is not None:
        return shell.run_script(script)
    return shell.repl()
