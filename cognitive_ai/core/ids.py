"""Stable identifier helpers."""

from __future__ import annotations

from uuid import uuid4


def new_id(prefix: str = "") -> str:
    """Return a short unique id, optionally prefixed (e.g. claim_a1b2c3d4)."""
    stem = uuid4().hex[:10]
    return f"{prefix}_{stem}" if prefix else stem
