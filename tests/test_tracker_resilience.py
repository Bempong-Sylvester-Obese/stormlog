from __future__ import annotations

from collections import deque
from pathlib import Path

import pytest

import stormlog.tracker as tracker_mod
from stormlog.collector_health import (
    COLLECTOR_HEALTH_DEGRADED,
    COLLECTOR_HEALTH_HEALTHY,
    COLLECTOR_HEALTH_UNHEALTHY,
)
from stormlog.device_collectors import DeviceMemorySample, DeviceMemorySampleResult
from stormlog.telemetry_sink import TelemetrySinkConfig


def _sample(
    *,
    allocated: int,
    reserved: int,
    used: int | None = None,
    total: int | None = 4096,
    free: int | None = None,
    active: int | None = 512,
    inactive: int | None = 256,
) -> DeviceMemorySample:
    resolved_used = max(allocated, reserved) if used is None else used
    resolved_free = (
        total - resolved_used if free is None and total is not None else free
    )
    return DeviceMemorySample(
        allocated_bytes=allocated,
        reserved_bytes=reserved,
        used_bytes=resolved_used,
        free_bytes=resolved_free,
        total_bytes=total,
        active_bytes=active,
        inactive_bytes=inactive,
        device_id=0,
    )


class _SequencedCollector:
    def __init__(self, results: list[DeviceMemorySampleResult]) -> None:
        self._results = deque(results)
        self._last = results[-1]

    def name(self) -> str:
        return "cuda"

    def is_available(self) -> bool:
        return True

    def capabilities(self) -> dict[str, object]:
        return {
            "backend": "cuda",
            "supports_device_total": True,
            "supports_device_free": True,
            "sampling_source": "test.collector",
            "telemetry_collector": "stormlog.cuda_tracker",
        }

    def sample(self) -> DeviceMemorySample:
        result = self.sample_with_diagnostics()
        if result.sample is None:
            raise RuntimeError(result.core_error or "collector unavailable")
        return result.sample

    def sample_with_diagnostics(self) -> DeviceMemorySampleResult:
        if self._results:
            self._last = self._results.popleft()
        return self._last


class _NoOpThread:
    def __init__(self, *args: object, **kwargs: object) -> None:
        _ = args
        self.daemon = bool(kwargs.get("daemon", False))

    def start(self) -> None:
        return None

    def join(self, timeout: float | None = None) -> None:
        _ = timeout


class _SequencedStopEvent:
    def __init__(self, waits: list[bool]) -> None:
        self._waits = deque(waits)

    def wait(self, timeout: float | None = None) -> bool:
        _ = timeout
        if self._waits:
            return self._waits.popleft()
        return True

    def set(self) -> None:
        return None

    def clear(self) -> None:
        return None


class _FailingFlushSink:
    def __init__(self) -> None:
        self.flush_calls = 0
        self.close_calls = 0

    def append(self, record: dict[str, object]) -> None:
        _ = record
        return None

    def flush(self, *, force: bool = False) -> None:
        _ = force
        self.flush_calls += 1
        raise OSError("disk full")

    def close(self) -> None:
        self.close_calls += 1


def _build_tracker(
    monkeypatch: pytest.MonkeyPatch,
    collector: _SequencedCollector,
    **kwargs: object,
) -> tracker_mod.MemoryTracker:
    monkeypatch.setattr(
        tracker_mod.MemoryTracker,
        "_setup_device",
        lambda self, _device: tracker_mod.torch.device("cuda:0"),
    )
    monkeypatch.setattr(
        tracker_mod,
        "build_device_memory_collector",
        lambda _device: collector,
    )
    monkeypatch.setattr(
        tracker_mod,
        "get_gpu_info",
        lambda _device: {"total_memory": 4096},
    )
    return tracker_mod.MemoryTracker(
        sampling_interval=0.01,
        enable_alerts=False,
        **kwargs,
    )


def test_memory_tracker_recovers_after_transient_collector_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _SequencedCollector(
        [
            DeviceMemorySampleResult(sample=_sample(allocated=128, reserved=256)),
            DeviceMemorySampleResult(
                sample=None,
                errors={"core_metrics": "collector unavailable"},
                core_error="collector unavailable",
            ),
            DeviceMemorySampleResult(sample=_sample(allocated=256, reserved=256)),
        ]
    )
    tracker = _build_tracker(monkeypatch, collector)
    current_time = {"value": 10.0}
    monkeypatch.setattr(tracker_mod.time, "time", lambda: current_time["value"])

    last_allocated = tracker._run_tracking_iteration(0)
    assert last_allocated == 0
    assert tracker.get_events()[-1].event_type == "collector_degraded"
    assert tracker.get_statistics()["collector_health_status"] == (
        COLLECTOR_HEALTH_UNHEALTHY
    )
    assert tracker.get_statistics()["collector_next_retry_epoch_s"] == pytest.approx(
        11.0
    )

    current_time["value"] = 10.5
    skipped_allocated = tracker._run_tracking_iteration(last_allocated)
    assert skipped_allocated == last_allocated
    assert [event.event_type for event in tracker.get_events()].count(
        "collector_degraded"
    ) == 1

    current_time["value"] = 11.1
    recovered_allocated = tracker._run_tracking_iteration(last_allocated)
    assert recovered_allocated == 256
    event_types = [event.event_type for event in tracker.get_events()]
    assert event_types.count("collector_degraded") == 1
    assert "collector_recovered" in event_types
    assert tracker.get_statistics()["collector_health_status"] == (
        COLLECTOR_HEALTH_HEALTHY
    )
    assert tracker.get_statistics()["telemetry_partial"] is False


def test_memory_tracker_keeps_retrying_during_persistent_collector_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = DeviceMemorySampleResult(
        sample=None,
        errors={"core_metrics": "collector unavailable"},
        core_error="collector unavailable",
    )
    collector = _SequencedCollector(
        [
            DeviceMemorySampleResult(sample=_sample(allocated=64, reserved=64)),
            failure,
            failure,
            failure,
        ]
    )
    tracker = _build_tracker(monkeypatch, collector)
    tracker._collector_retry_backoff_initial_s = 0.1
    tracker._collector_retry_backoff_cap_s = 0.4
    current_time = {"value": 20.0}
    monkeypatch.setattr(tracker_mod.time, "time", lambda: current_time["value"])

    last_allocated = tracker._run_tracking_iteration(64)
    assert last_allocated == 64

    current_time["value"] = 20.11
    tracker._run_tracking_iteration(last_allocated)
    current_time["value"] = 20.32
    tracker._run_tracking_iteration(last_allocated)

    stats = tracker.get_statistics()
    assert stats["collector_health_status"] == COLLECTOR_HEALTH_UNHEALTHY
    assert stats["telemetry_partial"] is True
    assert stats["collector_consecutive_failures"] == 3
    assert stats["collector_next_retry_epoch_s"] == pytest.approx(20.72)
    assert [event.event_type for event in tracker.get_events()] == [
        "collector_degraded"
    ]


def test_memory_tracker_emits_partial_sample_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial = DeviceMemorySampleResult(
        sample=_sample(
            allocated=128,
            reserved=128,
            total=None,
            free=None,
            active=None,
            inactive=None,
        ),
        partial_fields=(
            "device_total_bytes",
            "device_free_bytes",
            "allocator_active_bytes",
            "allocator_inactive_bytes",
        ),
        errors={
            "device_total_bytes": "total unavailable",
            "allocator_active_bytes": "stats unavailable",
        },
    )
    collector = _SequencedCollector(
        [
            DeviceMemorySampleResult(sample=_sample(allocated=128, reserved=128)),
            partial,
        ]
    )
    tracker = _build_tracker(monkeypatch, collector)
    tracker.stats["peak_memory"] = 128
    current_time = {"value": 30.0}
    monkeypatch.setattr(tracker_mod.time, "time", lambda: current_time["value"])

    last_allocated = tracker._run_tracking_iteration(128)

    assert last_allocated == 128
    events = tracker.get_events()
    assert [event.event_type for event in events] == ["collector_degraded", "sample"]
    assert events[-1].metadata is not None
    assert events[-1].metadata["collector_health_status"] == COLLECTOR_HEALTH_DEGRADED
    assert events[-1].metadata["telemetry_partial"] is True
    assert events[-1].metadata["collector_partial_fields"] == [
        "device_total_bytes",
        "device_free_bytes",
        "allocator_active_bytes",
        "allocator_inactive_bytes",
    ]
    assert tracker.get_statistics()["collector_health_status"] == (
        COLLECTOR_HEALTH_DEGRADED
    )


def test_memory_tracker_export_preserves_health_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    partial = DeviceMemorySampleResult(
        sample=_sample(
            allocated=256,
            reserved=256,
            total=None,
            free=None,
            active=None,
            inactive=None,
        ),
        partial_fields=("device_total_bytes", "device_free_bytes"),
        errors={"device_total_bytes": "total unavailable"},
    )
    collector = _SequencedCollector(
        [
            DeviceMemorySampleResult(sample=_sample(allocated=256, reserved=256)),
            partial,
        ]
    )
    tracker = _build_tracker(monkeypatch, collector)
    tracker.stats["peak_memory"] = 256
    current_time = {"value": 40.0}
    monkeypatch.setattr(tracker_mod.time, "time", lambda: current_time["value"])

    tracker._run_tracking_iteration(256)
    output_path = tmp_path / "tracker.json"
    tracker.export_events(str(output_path), format="json")
    payload = output_path.read_text(encoding="utf-8")

    assert "collector_health_status" in payload
    assert "collector_degraded" in payload
    assert "device_total_bytes" in payload


def test_memory_tracker_streams_events_to_append_only_sink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    collector = _SequencedCollector(
        [DeviceMemorySampleResult(sample=_sample(allocated=128, reserved=256))]
    )
    tracker = _build_tracker(
        monkeypatch,
        collector,
        telemetry_sink_config=TelemetrySinkConfig(
            root_dir=tmp_path / "sink",
            flush_every_events=1,
            flush_every_seconds=1.0,
            rollover_max_bytes=1024,
            retention_max_total_bytes=1024 * 1024,
        ),
    )

    tracker._run_tracking_iteration(0)
    tracker._close_telemetry_sink()

    segment = tmp_path / "sink" / "segment-000001.jsonl"
    lines = [line for line in segment.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 2
    payload = tracker_mod.json.loads(lines[-1])
    assert payload["collector"] == "stormlog.cuda_tracker"
    assert payload["event_type"] == "allocation"
    assert payload["allocator_allocated_bytes"] == 128


def test_memory_tracker_disables_failing_sink_and_keeps_tracking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _SequencedCollector(
        [DeviceMemorySampleResult(sample=_sample(allocated=128, reserved=256))]
    )
    tracker = _build_tracker(monkeypatch, collector)
    sink = _FailingFlushSink()
    tracker._telemetry_sink = sink
    tracker._stop_event = _SequencedStopEvent([False, False, True])
    iteration_inputs: list[int] = []

    def _run_iteration(last_allocated: int) -> int:
        iteration_inputs.append(last_allocated)
        return last_allocated + 1

    monkeypatch.setattr(tracker, "_run_tracking_iteration", _run_iteration)
    monkeypatch.setattr(tracker_mod.time, "sleep", lambda _: None)

    tracker._tracking_loop()

    assert iteration_inputs == [0, 1]
    assert sink.flush_calls == 1
    assert sink.close_calls == 1
    assert tracker._telemetry_sink is None


def test_memory_tracker_start_tracking_resets_collector_session_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _SequencedCollector(
        [DeviceMemorySampleResult(sample=_sample(allocated=128, reserved=256))]
    )
    tracker = _build_tracker(monkeypatch, collector)
    monkeypatch.setattr(tracker_mod.threading, "Thread", _NoOpThread)
    tracker._set_collector_health(
        status=COLLECTOR_HEALTH_UNHEALTHY,
        telemetry_partial=True,
        last_error="collector unavailable",
        consecutive_failures=3,
        next_retry_epoch_s=42.0,
    )
    tracker._last_observed_sample = _sample(allocated=512, reserved=768)
    tracker.stats["last_memory_check"] = 99.0

    tracker.start_tracking()

    stats = tracker.get_statistics()
    assert stats["collector_health_status"] == COLLECTOR_HEALTH_HEALTHY
    assert stats["collector_last_error"] is None
    assert stats["collector_consecutive_failures"] == 0
    assert stats["collector_next_retry_epoch_s"] is None
    assert stats["current_memory_allocated"] is None
    assert tracker.stats["last_memory_check"] == 0
    assert tracker.get_events()[-1].event_type == "start"
    assert tracker.get_events()[-1].memory_allocated == 0


def test_memory_tracker_hides_stale_current_stats_when_unhealthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _SequencedCollector(
        [
            DeviceMemorySampleResult(sample=_sample(allocated=256, reserved=512)),
            DeviceMemorySampleResult(
                sample=None,
                errors={"core_metrics": "collector unavailable"},
                core_error="collector unavailable",
            ),
        ]
    )
    tracker = _build_tracker(monkeypatch, collector)
    current_time = {"value": 10.0}
    monkeypatch.setattr(tracker_mod.time, "time", lambda: current_time["value"])

    tracker._run_tracking_iteration(0)
    stats = tracker.get_statistics()

    assert tracker._last_observed_sample is not None
    assert stats["collector_health_status"] == COLLECTOR_HEALTH_UNHEALTHY
    assert stats["current_memory_allocated"] is None
    assert stats["current_memory_reserved"] is None
    assert stats["memory_utilization_percent"] is None
