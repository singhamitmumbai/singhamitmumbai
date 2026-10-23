"""Boot banner and welcome text for the Cognitive OS."""

from __future__ import annotations

from ..version import __version__

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     ██████╗ ██████╗  ██████╗ ███╗   ██╗ ██████╗ ███████╗     ║
║    ██╔════╝██╔═══██╗██╔════╝ ████╗  ██║██╔═══██╗██╔════╝     ║
║    ██║     ██║   ██║██║  ███╗██╔██╗ ██║██║   ██║███████╗     ║
║    ██║     ██║   ██║██║   ██║██║╚██╗██║██║   ██║╚════██║     ║
║    ╚██████╗╚██████╔╝╚██████╔╝██║ ╚████║╚██████╔╝███████║     ║
║     ╚═════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ ╚══════╝     ║
║                                                              ║
║              C O G N I T I V E   O S                         ║
║         symbolic brain · no LLM · offline                    ║
╚══════════════════════════════════════════════════════════════╝
""".strip(
    "\n"
)


def welcome(version: str = __version__, root: str = "") -> str:
    lines = [
        BANNER,
        f"  version {version}  ·  store: {root or '(memory)'}",
        "",
        "  Type 'help' for commands.  'boot' if not yet started.",
        "  Cognition is rule-based — no network or chat models.",
        "",
    ]
    return "\n".join(lines)


def boot_splash(services: list[str]) -> str:
    rows = ["[boot] initializing Cognitive OS…", "[boot] loading symbolic brain…"]
    for name in services:
        rows.append(f"[boot] service online: {name}")
    rows.append("[boot] ready.")
    return "\n".join(rows)
