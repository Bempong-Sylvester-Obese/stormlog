"""Tests targeting tracker missing lines for coverage."""

import time
from typing import Any
from unittest import mock

import pytest

from stormlog.jax.tracker import (
    COLLECTOR_HEALTH_HEALTHY,
    COLLECTOR_HEALTH_UNHEALTHY,
    MemoryTracker,
)
from tests.jax_test_helpers import fake_jax_runtime  # noqa: F401

pytestmark = pytest.mark.usefixtures("fake_jax_runtime")


def test_transition_to_failure_and_success() -> None:
    tracker = MemoryTracker(enable_logging=True)

    # Trigger failure
    tracker._transition_to_failure(time.time(), Exception("test error"))
    assert tracker._collector_health.status == COLLECTOR_HEALTH_UNHEALTHY
    assert tracker._collector_health.consecutive_failures == 1

    # Trigger recovery
    tracker._transition_to_success(time.time())
    assert tracker._collector_health.status == COLLECTOR_HEALTH_HEALTHY
    assert tracker._collector_health.consecutive_failures == 0


def test_run_tracking_iteration_failure() -> None:
    tracker = MemoryTracker()
    with mock.patch.object(
        tracker, "_get_current_memory_bytes", side_effect=Exception("error")
    ):
        tracker._run_tracking_iteration()
        assert tracker._collector_health.status == COLLECTOR_HEALTH_UNHEALTHY


def test_unavailable_device_memory_does_not_emit_zero_sample() -> None:
    tracker = MemoryTracker()
    tracker._device_memory_available = False
    tracker._device_memory_unavailable_reason = "backend has no allocator stats"

    tracker._run_tracking_iteration()

    result = tracker.get_tracking_results()
    assert result.memory_usage == []
    assert result.device_memory_available is False
    assert result.memory_source == "unavailable"


def test_transient_initial_memory_stats_failure_recovers() -> None:
    device = mock.Mock()
    device.memory_stats.side_effect = [
        RuntimeError("runtime warming up"),
        {"bytes_in_use": 1024, "bytes_limit": 8192},
        {"bytes_in_use": 2048, "bytes_limit": 8192},
    ]

    with mock.patch("stormlog.jax.tracker.jax.local_devices", return_value=[device]):
        tracker = MemoryTracker()

    assert tracker._device_memory_available is False

    tracker._run_tracking_iteration()

    result = tracker.get_tracking_results()
    assert device.memory_stats.call_count == 3
    assert result.memory_usage == [2048]
    assert result.device_memory_available is True
    assert tracker._collector_health.status == COLLECTOR_HEALTH_HEALTHY


def test_tracking_loop_exception() -> None:
    tracker = MemoryTracker(enable_logging=True)
    tracker._stop_event = mock.Mock()
    tracker._stop_event.is_set.side_effect = [False, True]

    with mock.patch.object(
        tracker, "_run_tracking_iteration", side_effect=Exception("loop error")
    ):
        tracker._tracking_loop()
        assert tracker._stop_event.wait.called


def test_append_event_drop() -> None:
    tracker = MemoryTracker(max_history=5)
    for i in range(10):
        tracker._append_event(
            timestamp=time.time(), memory_bytes=100, event_type="sample"
        )

    # Should drop old events
    assert len(tracker._events) == 5
    assert tracker._history_dropped_events == 5


def test_trigger_alert() -> None:
    tracker = MemoryTracker(alert_threshold_mb=100)
    callback_calls = 0

    def callback(a: Any) -> None:
        nonlocal callback_calls
        callback_calls += 1

    tracker.add_alert_callback(callback)
    tracker._trigger_alert(200, time.time())
    assert len(tracker._alerts) == 1
    assert callback_calls == 1

    tracker.remove_alert_callback(callback)
    tracker._trigger_alert(300, time.time())
    assert len(tracker._alerts) == 2
    assert callback_calls == 1


def test_get_statistics_when_unhealthy() -> None:
    tracker = MemoryTracker()
    tracker._transition_to_failure(time.time(), Exception("error"))
    stats = tracker.get_statistics()
    assert stats["collector_health_status"] == COLLECTOR_HEALTH_UNHEALTHY


def test_start_tracking_already_running() -> None:
    tracker = MemoryTracker()
    tracker.tracking = True
    # Should not raise, just return
    tracker.start_tracking()


def test_stop_tracking_not_running() -> None:
    tracker = MemoryTracker()
    # Should not raise, returns empty result
    res = tracker.stop_tracking()
    assert res.duration == 0


def test_format_results_no_telemetry() -> None:
    tracker = MemoryTracker()
    res = tracker.stop_tracking()
    # Empty results should format without error
    assert isinstance(res.memory_usage, list)
