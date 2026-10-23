#!/usr/bin/env python3
"""Launch Cognitive OS chat UI and print a public Cloudflare tunnel URL.

Solves cloud-VM access: laptop localhost / Try Live often cannot reach the agent.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def ensure_cloudflared(path: Path) -> Path:
    if path.exists() and path.stat().st_mode & 0o111:
        return path
    import urllib.request

    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    print(f"Downloading cloudflared…")
    urllib.request.urlretrieve(url, path)
    path.chmod(0o755)
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Chat UI + public tunnel")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--domain", default="animals")
    args = p.parse_args(argv)

    # Start chat server
    chat = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cognitive_ai",
            "chat",
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--domain",
            args.domain,
        ],
        cwd=str(ROOT),
    )
    time.sleep(1.2)
    if chat.poll() is not None:
        print("Chat UI failed to start", file=sys.stderr)
        return 1

    cf = ensure_cloudflared(Path("/tmp/cloudflared"))
    tunnel = subprocess.Popen(
        [str(cf), "tunnel", "--url", f"http://127.0.0.1:{args.port}", "--protocol", "http2"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    public = None
    assert tunnel.stdout is not None
    deadline = time.time() + 45
    while time.time() < deadline:
        line = tunnel.stdout.readline()
        if not line and tunnel.poll() is not None:
            break
        if line:
            sys.stdout.write(line)
            sys.stdout.flush()
            m = re.search(r"(https://[a-zA-Z0-9-]+\.trycloudflare\.com)", line)
            if m:
                public = m.group(1)
                break

    if not public:
        print("Could not obtain public tunnel URL", file=sys.stderr)
        chat.terminate()
        tunnel.terminate()
        return 1

    print("\n" + "=" * 60)
    print("Open this URL on your laptop (not localhost):")
    print(f"  {public}")
    print("=" * 60 + "\n")
    # Keep writing tunnel logs
    try:
        while True:
            line = tunnel.stdout.readline()
            if not line and tunnel.poll() is not None:
                break
            if line:
                sys.stdout.write(line)
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        tunnel.terminate()
        chat.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
