"""Regression tests for PyTorch memory visualization."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib
import plotly.graph_objects as go
import pytest

matplotlib.use("Agg")

from stormlog.profiler import MemorySnapshot, ProfileResult
from stormlog.visualizer import MemoryVisualizer


def _make_snapshot(timestamp: float, allocated_memory: int = 0) -> MemorySnapshot:
    return MemorySnapshot(
        timestamp=timestamp,
        allocated_memory=allocated_memory,
        reserved_memory=allocated_memory,
        max_memory_allocated=allocated_memory,
        max_memory_reserved=allocated_memory,
        active_memory=allocated_memory,
        inactive_memory=0,
        cpu_memory=0,
    )


def _make_result() -> ProfileResult:
    before = _make_snapshot(1.0)
    after = _make_snapshot(2.0, allocated_memory=1024)
    return ProfileResult(
        function_name="profiled_function",
        execution_time=1.0,
        memory_before=before,
        memory_after=after,
        memory_peak=after,
        memory_allocated=1024,
        memory_freed=0,
        tensors_created=0,
        tensors_deleted=0,
    )


def test_csv_export_returns_written_results_path(tmp_path: Path) -> None:
    returned_path = MemoryVisualizer().export_data(
        results=[_make_result()],
        snapshots=[_make_snapshot(0.0)],
        format="csv",
        save_path=str(tmp_path / "profile"),
    )

    assert Path(returned_path).exists()
    assert "_results_" in Path(returned_path).name


def test_csv_export_returns_written_snapshots_path(tmp_path: Path) -> None:
    returned_path = MemoryVisualizer().export_data(
        snapshots=[_make_snapshot(0.0)],
        format="csv",
        save_path=str(tmp_path / "profile"),
    )

    assert Path(returned_path).exists()
    assert "_snapshots_" in Path(returned_path).name


def test_csv_export_rejects_empty_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No data available for CSV export"):
        MemoryVisualizer().export_data(
            format="csv",
            save_path=str(tmp_path / "profile"),
        )


def test_memory_timeline_sorts_combined_data_by_timestamp() -> None:
    figure = cast(
        go.Figure,
        MemoryVisualizer().plot_memory_timeline(
            results=[_make_result()],
            snapshots=[_make_snapshot(float(index)) for index in range(5)],
            interactive=True,
        ),
    )

    timestamps = list(figure.data[0].x)

    assert timestamps == sorted(timestamps)
