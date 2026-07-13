"""Regression tests for PyTorch memory analysis."""

from __future__ import annotations

import pytest

from stormlog.analyzer import MemoryAnalyzer
from stormlog.profiler import MemorySnapshot, ProfileResult


def _make_result(execution_time: float, timestamp: float) -> ProfileResult:
    snapshot = MemorySnapshot(
        timestamp=timestamp,
        allocated_memory=0,
        reserved_memory=0,
        max_memory_allocated=0,
        max_memory_reserved=0,
        active_memory=0,
        inactive_memory=0,
        cpu_memory=0,
    )
    return ProfileResult(
        function_name="fast_function",
        execution_time=execution_time,
        memory_before=snapshot,
        memory_after=snapshot,
        memory_peak=snapshot,
        memory_allocated=0,
        memory_freed=0,
        tensors_created=0,
        tensors_deleted=0,
    )


@pytest.mark.parametrize(
    "method_name",
    ["generate_performance_insights", "generate_optimization_report"],
)
def test_zero_execution_times_do_not_crash_performance_analysis(
    method_name: str,
) -> None:
    analyzer = MemoryAnalyzer()
    results = [_make_result(0.0, float(index)) for index in range(5)]

    method = getattr(analyzer, method_name)

    assert method(results) is not None
