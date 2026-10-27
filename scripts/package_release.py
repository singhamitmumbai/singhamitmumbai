#!/usr/bin/env python3
"""Build a distributable zip of the Complete Non-LLM AI source tree."""

from __future__ import annotations

import argparse
import datetime as dt
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXCLUDE_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "egg-info",
    "cognitive_ai.egg-info",
    "dist",
    "build",
    "runs",
    "wandb",
    "generated",
    "node_modules",
}

# Large optional ASR packs — not required for core/vision stub demos
EXCLUDE_PATH_PREFIXES = (
    "models/vosk-model-",
)

EXCLUDE_SUFFIXES = (
    ".pyc",
    ".pyo",
    ".pt",  # YOLO weights downloaded locally
    ".sqlite",
)


def should_skip(rel: Path) -> bool:
    parts = rel.parts
    if any(p in EXCLUDE_DIR_NAMES for p in parts):
        return True
    if any(p.endswith(".egg-info") for p in parts):
        return True
    s = rel.as_posix()
    if any(s.startswith(pref) for pref in EXCLUDE_PATH_PREFIXES):
        return True
    if rel.suffix in EXCLUDE_SUFFIXES:
        return True
    return False


def build_zip(out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    bytes_in = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            if should_skip(rel):
                continue
            zf.write(path, arcname=f"Complete_Non_LLM_AI/{rel.as_posix()}")
            count += 1
            bytes_in += path.stat().st_size
        # Manifest inside zip
        from cognitive_ai.version import __version__

        manifest = (
            f"name=Complete_Non_LLM_AI\n"
            f"version={__version__}\n"
            f"built_at={dt.datetime.utcnow().isoformat()}Z\n"
            f"files={count}\n"
            f"llm_at_runtime=false\n"
            f"human_level_agi=false\n"
            f"includes=cognitive_ai,third_party/YOLO_Projects,tests,docs,scripts\n"
            f"excludes=.git,vosk models,*.pt,__pycache__\n"
        )
        zf.writestr("Complete_Non_LLM_AI/RELEASE.txt", manifest)
    size = out.stat().st_size
    return {"path": str(out), "files": count, "raw_bytes": bytes_in, "zip_bytes": size}


def main() -> int:
    p = argparse.ArgumentParser(description="Package Complete Non-LLM AI as a zip")
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output zip path (default: dist/Complete_Non_LLM_AI_vVERSION.zip)",
    )
    args = p.parse_args()
    from cognitive_ai.version import __version__

    out = Path(args.output) if args.output else ROOT / "dist" / f"Complete_Non_LLM_AI_v{__version__}.zip"
    info = build_zip(out)
    print(f"Wrote {info['path']}")
    print(f"  files={info['files']}  zip={info['zip_bytes']:,} bytes  source≈{info['raw_bytes']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
