"""Tests targeting tracker missing lines for coverage."""

import time
from typing import Any
from unittest import mock

import pytest

pytest.importorskip("jax")

from stormlog.jax.tracker import (
    COLLECTOR_HEALTH_HEALTHY,
    COLLECTOR_HEALTH_UNHEALTHY,
    MemoryTracker,
)


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
