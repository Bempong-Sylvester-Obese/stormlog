"""Tests for JAX visualizer."""

from typing import Any
from unittest import mock

import pytest

from stormlog.jax.profiler import MemorySnapshot, ProfileResult
from stormlog.jax.visualizer import MemoryVisualizer
from tests.jax_test_helpers import fake_jax_runtime  # noqa: F401

pytestmark = pytest.mark.usefixtures("fake_jax_runtime")


def test_visualizer_init() -> None:
    viz = MemoryVisualizer(style="default")
    assert viz.style == "default"


@mock.patch("stormlog.jax.visualizer.plt")
def test_plot_memory_timeline_matplotlib(mock_plt: Any) -> None:
    viz = MemoryVisualizer()
    snapshots = [
        MemorySnapshot(1, "s1", 100, 100, 0, 0, {}),
        MemorySnapshot(2, "s2", 200, 200, 0, 0, {}),
    ]
    res = ProfileResult(0, 5, 200, 150, 100, snapshots, {})

    viz.plot_memory_timeline(res, interactive=False)

    mock_plt.figure.assert_called_once()
    mock_plt.plot.assert_called_once()
    mock_plt.show.assert_called_once()


@mock.patch("stormlog.jax.visualizer.go", create=True)
def test_plot_memory_timeline_plotly(mock_go: Any) -> None:
    # Set PLOTLY_AVAILABLE to True for the test
    with mock.patch("stormlog.jax.visualizer.PLOTLY_AVAILABLE", True):
        viz = MemoryVisualizer()
        snapshots = [
            MemorySnapshot(1, "s1", 100, 100, 0, 0, {}),
        ]
        res = ProfileResult(0, 5, 100, 100, 100, snapshots, {})

        viz.plot_memory_timeline(res, interactive=True)

        mock_go.Figure.assert_called_once()
        mock_go.Scatter.assert_called_once()


def test_plot_memory_timeline_no_data() -> None:
    viz = MemoryVisualizer()
    res = ProfileResult(0, 0, 0, 0, 0, [], {})

    # Should not throw, just log warning
    viz.plot_memory_timeline(res)


@mock.patch("stormlog.jax.visualizer.plt")
def test_plot_memory_timeline_save_path(mock_plt: Any, tmp_path: Any) -> None:
    viz = MemoryVisualizer()
    snapshots = [MemorySnapshot(1, "s1", 100, 100, 0, 0, {})]
    res = ProfileResult(0, 5, 200, 150, 100, snapshots, {})
    viz.plot_memory_timeline(res, save_path=str(tmp_path / "test.png"))
    mock_plt.savefig.assert_called_once()


@mock.patch("stormlog.jax.visualizer.plt")
def test_plot_memory_timeline_fallback(mock_plt: Any) -> None:
    viz = MemoryVisualizer()
    res = mock.Mock()
    res.snapshots = []
    res.memory_usage = [100, 200]
    res.timestamps = [1, 2]
    viz.plot_memory_timeline(res)
    mock_plt.plot.assert_called_once()


@mock.patch("stormlog.jax.visualizer.plt")
def test_plot_function_comparison(mock_plt: Any, tmp_path: Any) -> None:
    viz = MemoryVisualizer()
    profiles = {
        "func1": {"peak_memory_bytes": 1024 * 1024},
        "func2": {"peak_memory_bytes": 2048 * 1024},
    }
    viz.plot_function_comparison(profiles, save_path=str(tmp_path / "test.png"))
    mock_plt.bar.assert_called_once()
    mock_plt.savefig.assert_called_once()


def test_plot_function_comparison_empty() -> None:
    viz = MemoryVisualizer()
    viz.plot_function_comparison({})
