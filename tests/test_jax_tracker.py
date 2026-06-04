"""Tests for JAX memory tracker."""

import time
from typing import Any, Generator, Optional
from unittest import mock

import pytest

pytest.importorskip("jax")
from stormlog.jax.tracker import JAXMemoryTracker
from stormlog.oom_flight_recorder import OOMExceptionClassification
from stormlog.session import SESSION_STATUS_COMPLETED
from tests.jax_test_helpers import jax_fixture, jax_mark


@jax_fixture
def mock_device() -> mock.Mock:
    device = mock.Mock()
    device.memory_stats.return_value = {
        "bytes_in_use": 1024,
        "peak_bytes_in_use": 2048,
        "bytes_limit": 10240,
    }
    return device


@jax_fixture
def mock_jax_devices(mock_device: Any) -> Generator[None, None, None]:
    with mock.patch(
        "stormlog.jax.tracker.jax.local_devices", return_value=[mock_device]
    ):
        with mock.patch("stormlog.jax.tracker.jax.numpy.zeros"):
            yield


def _wait_for_condition(predicate: Any, message: str, timeout: float = 1.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(message)


def _wait_for_retained_samples(tracker: JAXMemoryTracker) -> None:
    _wait_for_condition(
        lambda: tracker.get_statistics().get("history_retained_samples", 0) > 0,
        "tracker did not collect a sample",
    )


@jax_mark
def test_tracker_init(mock_jax_devices: Any) -> None:
    """Verify tracker initialization and validation."""
    tracker = JAXMemoryTracker(sampling_interval=0.1, alert_threshold_mb=100)
    assert tracker.sampling_interval == 0.1
    assert tracker.alert_threshold_mb == 100

    with pytest.raises(ValueError, match="sampling_interval must be > 0"):
        JAXMemoryTracker(sampling_interval=-1.0)

    with pytest.raises(ValueError, match="max_history must be >= 1"):
        JAXMemoryTracker(max_history=0)


@jax_mark
def test_tracker_lifecycle(mock_jax_devices: Any) -> None:
    """Verify start and stop lifecycle."""
    tracker = JAXMemoryTracker(sampling_interval=0.01)

    tracker.start_tracking()
    try:
        _wait_for_retained_samples(tracker)
    finally:
        result = tracker.stop_tracking()

    assert result.duration > 0
    assert result.samples_collected > 0
    assert result.peak_memory_bytes == 1024

    # Session lifecycle
    summary = result.session_summary
    assert summary is not None
    assert summary.status == SESSION_STATUS_COMPLETED
    assert summary.source == "stormlog.jax.tracker"


@jax_mark
def test_tracker_idempotent_start(mock_jax_devices: Any) -> None:
    """Verify double start is safe."""
    tracker = JAXMemoryTracker(sampling_interval=0.01)
    tracker.start_tracking()
    tracker.start_tracking()  # Should not raise or spawn second thread
    try:
        _wait_for_retained_samples(tracker)
    finally:
        result = tracker.stop_tracking()
    assert result.samples_collected > 0


@jax_mark
def test_tracker_stop_without_start(mock_jax_devices: Any) -> None:
    """Verify stopping an unstarted tracker is safe."""
    tracker = JAXMemoryTracker(sampling_interval=0.01)
    result = tracker.stop_tracking()
    assert result.samples_collected == 0
    assert result.duration == 0


@jax_mark
def test_telemetry_event_structure(mock_jax_devices: Any) -> None:
    """Verify built telemetry events have expected fields."""
    tracker = JAXMemoryTracker(sampling_interval=0.01)
    tracker.start_tracking()
    try:
        _wait_for_retained_samples(tracker)
    finally:
        result = tracker.stop_tracking()

    assert len(result.telemetry_events) > 0
    event = result.telemetry_events[0]

    assert event["collector"] == "stormlog.jax.memory_tracker"
    assert event["allocator_allocated_bytes"] == 1024
    assert event["device_total_bytes"] == 10240


@jax_mark
def test_alerts(mock_jax_devices: Any) -> None:
    """Verify alerting logic."""
    tracker = JAXMemoryTracker(
        sampling_interval=0.01, alert_threshold_mb=0.0001
    )  # 100 bytes threshold

    alert_triggered = False

    def on_alert(alert: dict[str, Any]) -> None:
        nonlocal alert_triggered
        alert_triggered = True

    tracker.add_alert_callback(on_alert)

    tracker.start_tracking()
    try:
        _wait_for_condition(lambda: alert_triggered, "tracker did not trigger alert")
    finally:
        result = tracker.stop_tracking()

    assert alert_triggered
    assert result.alert_count > 0


@jax_mark
def test_phase_tracking(mock_jax_devices: Any) -> None:
    """Verify phase tracking API works."""
    tracker = JAXMemoryTracker(sampling_interval=0.01)

    with pytest.raises(RuntimeError):
        # Phase tracking requires an active session
        tracker.enter_phase("test_phase")

    tracker.start_tracking()

    with tracker.phase("my_phase"):
        time.sleep(0.05)

    result = tracker.stop_tracking()
    events = result.telemetry_events

    phase_events = [
        e for e in events if e.get("event_type") in {"phase_enter", "phase_exit"}
    ]
    assert [e.get("event_type") for e in phase_events] == [
        "phase_enter",
        "phase_exit",
    ]


@jax_mark
@mock.patch("stormlog.jax.tracker.jax.profiler.save_device_memory_profile")
def test_device_profile_export(mock_save: Any, mock_jax_devices: Any) -> None:
    """Verify JAX device profile export."""
    tracker = JAXMemoryTracker(sampling_interval=0.01, save_device_profile_on_stop=True)
    tracker.start_tracking()
    time.sleep(0.05)
    result = tracker.stop_tracking()

    mock_save.assert_called_once()
    assert result.device_memory_profile_path is not None
    assert result.device_memory_profile_path.endswith(".prof")


@jax_mark
def test_handle_exception_non_oom(mock_jax_devices: Any) -> None:
    """handle_exception returns None for non-OOM exceptions."""
    tracker = JAXMemoryTracker(enable_oom_flight_recorder=True)
    result = tracker.handle_exception(ValueError("not an OOM"), context="test")
    assert result is None


@jax_mark
def test_handle_exception_oom_disabled_recorder(mock_jax_devices: Any) -> None:
    """handle_exception returns None when flight recorder is disabled."""
    tracker = JAXMemoryTracker(enable_oom_flight_recorder=False)
    with mock.patch(
        "stormlog.jax.tracker.classify_oom_exception",
        return_value=OOMExceptionClassification(True, "test_oom"),
    ):
        result = tracker.handle_exception(RuntimeError("out of memory"), context="test")
    assert result is None


@jax_mark
def test_handle_exception_oom_dumps_bundle(
    mock_jax_devices: Any, tmp_path: Any
) -> None:
    """handle_exception triggers a dump bundle on OOM."""
    tracker = JAXMemoryTracker(
        enable_oom_flight_recorder=True,
        oom_dump_dir=str(tmp_path),
    )
    tracker._last_successful_memory_bytes = 1024
    tracker._last_reserved_bytes = 1536

    with mock.patch.object(
        tracker._oom_flight_recorder,
        "dump",
        return_value=str(tmp_path / "oom_dump_bundle"),
    ) as mock_dump:
        with mock.patch(
            "stormlog.jax.tracker.classify_oom_exception",
            return_value=OOMExceptionClassification(
                True, "message_pattern:out of memory"
            ),
        ):
            with mock.patch.object(
                tracker,
                "save_device_memory_profile_to_dir",
                return_value=str(tmp_path / "oom_dump_bundle" / "profile.prof"),
            ):
                # Create the manifest so enrichment works
                bundle = tmp_path / "oom_dump_bundle"
                bundle.mkdir()
                manifest = bundle / "manifest.json"
                manifest.write_text('{"files": []}', encoding="utf-8")

                result = tracker.handle_exception(
                    RuntimeError("out of memory"), context="test_op"
                )

    assert result is not None
    mock_dump.assert_called_once()
    dump_kwargs = mock_dump.call_args.kwargs
    assert dump_kwargs["reason"] == "message_pattern:out of memory"
    assert dump_kwargs["backend"] == "jax"
    assert dump_kwargs["context"] == "test_op"
    assert dump_kwargs["metadata"]["sample_allocated_bytes"] == 1024
    assert dump_kwargs["metadata"]["sample_reserved_bytes"] == 1536
    assert "allocator_reserved_approximate" not in dump_kwargs["metadata"]


@jax_mark
def test_handle_exception_enriches_manifest_with_profile(
    mock_jax_devices: Any, tmp_path: Any
) -> None:
    """handle_exception appends device profile to OOM manifest."""
    tracker = JAXMemoryTracker(
        enable_oom_flight_recorder=True,
        oom_dump_dir=str(tmp_path),
    )

    bundle = tmp_path / "oom_dump_bundle"
    bundle.mkdir()
    manifest = bundle / "manifest.json"
    manifest.write_text('{"files": ["events.json"]}', encoding="utf-8")

    with mock.patch.object(
        tracker._oom_flight_recorder,
        "dump",
        return_value=str(bundle),
    ):
        with mock.patch(
            "stormlog.jax.tracker.classify_oom_exception",
            return_value=OOMExceptionClassification(True, "test_oom"),
        ):
            profile_path = str(bundle / "jax-device-memory.prof")
            with mock.patch.object(
                tracker,
                "save_device_memory_profile_to_dir",
                return_value=profile_path,
            ):
                tracker.handle_exception(RuntimeError("OOM!"), context="training")

    assert tracker.last_oom_dump_path == str(bundle)
    assert "jax_device_profile" in manifest.read_text(encoding="utf-8")


@jax_mark
def test_capture_oom_context_manager(mock_jax_devices: Any, tmp_path: Any) -> None:
    """capture_oom context manager intercepts OOM and re-raises."""
    tracker = JAXMemoryTracker(
        enable_oom_flight_recorder=True,
        oom_dump_dir=str(tmp_path),
    )

    dumped = []

    def fake_handle_exception(
        exc: BaseException,
        context: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        dumped.append(True)
        return str(tmp_path / "test_dump")

    with mock.patch.object(
        tracker, "handle_exception", side_effect=fake_handle_exception
    ):
        try:
            with tracker.capture_oom("test_block"):
                raise RuntimeError("out of memory")
        except RuntimeError as exc:
            assert "out of memory" in str(exc)
        else:
            raise AssertionError("Expected RuntimeError")

    assert len(dumped) == 1


@jax_mark
def test_capture_oom_passes_non_oom_through(
    mock_jax_devices: Any,
) -> None:
    """capture_oom still raises non-OOM exceptions after handle_exception."""
    tracker = JAXMemoryTracker(enable_oom_flight_recorder=True)
    called = []
    with mock.patch.object(
        tracker, "handle_exception", side_effect=lambda *a, **kw: called.append(True)
    ):
        try:
            with tracker.capture_oom("test_block"):
                raise ValueError("not memory related")
        except ValueError as exc:
            assert "not memory" in str(exc)
        else:
            raise AssertionError("Expected ValueError")

    assert len(called) == 1


@jax_mark
def test_get_statistics_includes_oom_fields(mock_jax_devices: Any) -> None:
    """get_statistics includes OOM recorder fields."""
    tracker = JAXMemoryTracker(enable_oom_flight_recorder=True)
    stats = tracker.get_statistics()
    assert stats["oom_flight_recorder_enabled"] is True
    assert stats["last_oom_dump_path"] is None
