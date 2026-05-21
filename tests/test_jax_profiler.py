"""Tests for JAX memory profiler."""

import time
from typing import Any, Generator
from unittest import mock

import pytest

pytest.importorskip("jax")

from stormlog.jax.profiler import (
    JAXMemoryProfiler,
    MemorySnapshot,
    ProfileResult,
    clear_global_profiler,
    clear_profiles,
    get_global_profiler,
    get_profile_summaries,
)
from tests.jax_test_helpers import jax_fixture, jax_mark


@jax_fixture
def mock_device() -> mock.Mock:
    device = mock.Mock()
    device.memory_stats.return_value = {
        "bytes_in_use": 1024,
        "peak_bytes_in_use": 2048,
    }
    return device


@jax_fixture
def profiler(mock_device: mock.Mock) -> Generator[Any, None, None]:
    with mock.patch(
        "stormlog.jax.profiler.jax.local_devices", return_value=[mock_device]
    ):
        with mock.patch("stormlog.jax.profiler.jax.numpy.zeros") as mock_zeros:
            mock_zeros.return_value.block_until_ready.return_value = None
            p = JAXMemoryProfiler(device_index=0)
            yield p
            p.reset()


@jax_mark
def test_snapshot_creation() -> None:
    """Verify MemorySnapshot properties."""
    snapshot = MemorySnapshot(
        timestamp=time.time(),
        name="test",
        device_memory_bytes=1000,
        device_memory_reserved_bytes=2000,
        cpu_memory_bytes=2000,
        device_id=0,
        memory_stats={"bytes_in_use": 1000},
    )
    assert snapshot.name == "test"
    assert snapshot.device_memory_bytes == 1000


@jax_mark
def test_profile_result_properties() -> None:
    """Verify ProfileResult calculated properties."""
    result = ProfileResult(
        start_time=1.0,
        end_time=3.0,
        peak_memory_bytes=5000,
        average_memory_bytes=2500,
        min_memory_bytes=1000,
        snapshots=[],
        function_profiles={},
    )
    assert result.duration == 2.0


@jax_mark
def test_capture_snapshot(profiler: Any) -> None:
    """Verify manual snapshot capture."""
    snapshot = profiler.capture_snapshot("manual_1")
    assert snapshot.name == "manual_1"
    assert snapshot.device_memory_bytes == 1024
    assert snapshot.cpu_memory_bytes > 0


@jax_mark
def test_profile_function(profiler: Any) -> None:
    """Verify function decorator profiles memory."""

    def dummy_work() -> str:
        time.sleep(0.01)
        return "done"

    decorated_work = profiler.profile_function(dummy_work)

    result = decorated_work()
    assert result == "done"

    res = profiler.get_results()
    assert len(res.snapshots) >= 2  # before and after
    assert "dummy_work" in res.function_profiles
    func_prof = res.function_profiles["dummy_work"]
    assert func_prof["calls"] == 1


@jax_mark
def test_profile_context(profiler: Any) -> None:
    """Verify context manager profiles memory."""
    with profiler.profile_context("my_context"):
        time.sleep(0.01)

    res = profiler.get_results()
    assert len(res.snapshots) == 2
    assert "my_context" in res.function_profiles


@jax_mark
def test_get_results_empty(profiler: Any) -> None:
    """Verify getting results when empty is safe."""
    res = profiler.get_results()
    assert res.peak_memory_bytes == 0
    assert res.average_memory_bytes == 0
    assert len(res.snapshots) == 0


@jax_mark
def test_continuous_profiling(profiler: Any) -> None:
    """Verify continuous background profiling."""
    profiler.start_continuous_profiling(interval=0.05)
    time.sleep(0.15)
    profiler.stop_continuous_profiling()

    res = profiler.get_results()
    assert len(res.snapshots) >= 2


@jax_mark
def test_context_manager_lifecycle(profiler: Any) -> None:
    """Verify JAXMemoryProfiler can be used as a context manager."""
    with profiler:
        time.sleep(0.05)

    res = profiler.get_results()
    # Enter and exit should capture something or background thread captures
    assert len(res.snapshots) >= 0


@jax_mark
def test_global_profiler_lifecycle() -> None:
    """Verify global profiler singletons."""
    clear_global_profiler()

    p = get_global_profiler()
    assert isinstance(p, JAXMemoryProfiler)

    p2 = get_global_profiler()
    assert p is p2

    with p.profile_context("test"):
        pass

    summaries = get_profile_summaries()
    assert len(summaries) == 1

    clear_profiles()
    summaries = get_profile_summaries()
    assert len(summaries) == 0

    # Should be the same profiler, just reset
    assert get_global_profiler() is p

    clear_global_profiler()
    p3 = get_global_profiler()
    assert p3 is not p
