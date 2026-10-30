"""Runtime monitoring: latency, failures, coverage, resource use."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RequestMetrics:
    operation: str
    latency_ms: float
    success: bool
    abstained: bool = False
    error: str = ""


@dataclass
class RuntimeMonitor:
    requests: list[RequestMetrics] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def record(
        self,
        operation: str,
        latency_ms: float,
        *,
        success: bool,
        abstained: bool = False,
        error: str = "",
    ) -> None:
        self.requests.append(
            RequestMetrics(
                operation=operation,
                latency_ms=latency_ms,
                success=success,
                abstained=abstained,
                error=error,
            )
        )

    def summary(self) -> dict[str, Any]:
        n = len(self.requests)
        if not n:
            return {
                "requests": 0,
                "success_rate": None,
                "coverage": None,
                "p50_latency_ms": None,
                "p95_latency_ms": None,
                "uptime_s": time.time() - self.started_at,
            }
        successes = sum(1 for r in self.requests if r.success)
        abstained = sum(1 for r in self.requests if r.abstained)
        latencies = sorted(r.latency_ms for r in self.requests)

        def pct(p: float) -> float:
            idx = min(len(latencies) - 1, max(0, int(round(p * (len(latencies) - 1)))))
            return latencies[idx]

        return {
            "requests": n,
            "success_rate": successes / n,
            "coverage": (n - abstained) / n,  # answered vs rejected/abstained
            "abstain_rate": abstained / n,
            "p50_latency_ms": pct(0.5),
            "p95_latency_ms": pct(0.95),
            "errors": [r.error for r in self.requests if r.error][-10:],
            "uptime_s": time.time() - self.started_at,
        }
