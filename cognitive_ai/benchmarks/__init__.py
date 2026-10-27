"""Harder generalization benchmarks (SCAN-style + skill transfer + novel AGI-gap)."""

from .hard_novel import run_hard_novel_benchmark
from .scan_like import run_scan_benchmark
from .transfer_bench import run_transfer_benchmark

__all__ = ["run_scan_benchmark", "run_transfer_benchmark", "run_hard_novel_benchmark"]
