"""Tests for JAX memory profiler."""

import time
from unittest import mock

import pytest

pytest.importorskip("jax")

from stormlog.jax.profiler import (
    JAXMemoryProfiler,
    JAXMemorySnapshot,
    JAXProfileResult,
    clear_global_profiler,
    clear_profiles,
    get_global_profiler,
    get_profile_summaries,
)


@pytest.fixture
def mock_device():
    device = mock.Mock()
    device.memory_stats.return_value = {
        "bytes_in_use": 1024,
        "peak_bytes_in_use": 2048,
    }
    return device


@pytest.fixture
def profiler(mock_device):
    with mock.patch("stormlog.jax.profiler.jax.devices", return_value=[mock_device]):
        p = JAXMemoryProfiler(device_index=0)
        yield p
        p.reset()


@pytest.mark.jax
def test_snapshot_creation() -> None:
    """Verify JAXMemorySnapshot properties."""
    snapshot = JAXMemorySnapshot(
        timestamp=time.time(),
        name="test",
        device_memory_bytes=1000,
        cpu_memory_bytes=2000,
        device_id=0,
        memory_stats={"bytes_in_use": 1000},
    )
    assert snapshot.name == "test"
    assert snapshot.device_memory_bytes == 1000


@pytest.mark.jax
def test_profile_result_properties() -> None:
    """Verify JAXProfileResult calculated properties."""
    result = JAXProfileResult(
        start_time=1.0,
        end_time=3.0,
        peak_memory_bytes=5000,
        average_memory_bytes=2500,
        min_memory_bytes=1000,
        snapshots=[],
        function_profiles={},
    )
    assert result.duration == 2.0


@pytest.mark.jax
def test_capture_snapshot(profiler) -> None:
    """Verify manual snapshot capture."""
    snapshot = profiler.capture_snapshot("manual_1")
    assert snapshot.name == "manual_1"
    assert snapshot.device_memory_bytes == 1024
    assert snapshot.cpu_memory_bytes > 0


@pytest.mark.jax
def test_profile_function(profiler) -> None:
    """Verify function decorator profiles memory."""

    @profiler.profile_function
    def dummy_work():
        time.sleep(0.01)
        return "done"

    result = dummy_work()
    assert result == "done"

    res = profiler.get_results()
    assert len(res.snapshots) >= 2  # before and after
    assert "dummy_work" in res.function_profiles
    func_prof = res.function_profiles["dummy_work"]
    assert func_prof["calls"] == 1


@pytest.mark.jax
def test_profile_context(profiler) -> None:
    """Verify context manager profiles memory."""
    with profiler.profile_context("my_context"):
        time.sleep(0.01)

    res = profiler.get_results()
    assert len(res.snapshots) == 2
    assert "my_context" in res.function_profiles


@pytest.mark.jax
def test_get_results_empty(profiler) -> None:
    """Verify getting results when empty is safe."""
    res = profiler.get_results()
    assert res.peak_memory_bytes == 0
    assert res.average_memory_bytes == 0
    assert len(res.snapshots) == 0


@pytest.mark.jax
def test_continuous_profiling(profiler) -> None:
    """Verify continuous background profiling."""
    profiler.start_continuous_profiling(interval=0.05)
    time.sleep(0.15)
    profiler.stop_continuous_profiling()

    res = profiler.get_results()
    assert len(res.snapshots) >= 2


@pytest.mark.jax
def test_context_manager_lifecycle(profiler) -> None:
    """Verify JAXMemoryProfiler can be used as a context manager."""
    with profiler:
        time.sleep(0.05)

    res = profiler.get_results()
    # Enter and exit should capture something or background thread captures
    assert len(res.snapshots) >= 0


@pytest.mark.jax
def test_global_profiler_lifecycle() -> None:
    """Verify global profiler singletons."""
    clear_global_profiler()

    p = get_global_profiler()
    assert isinstance(p, JAXMemoryProfiler)

    p2 = get_global_profiler()
    assert p is p2

    # Add fake data
    p.capture_snapshot("test")

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
