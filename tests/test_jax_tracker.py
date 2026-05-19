"""Tests for JAX memory tracker."""

import time
from unittest import mock

import pytest

pytest.importorskip("jax")
from stormlog.jax.tracker import JAXMemoryTracker
from stormlog.session import SESSION_STATUS_COMPLETED


@pytest.fixture
def mock_device():
    device = mock.Mock()
    device.memory_stats.return_value = {
        "bytes_in_use": 1024,
        "peak_bytes_in_use": 2048,
        "bytes_limit": 10240,
    }
    return device


@pytest.fixture
def mock_jax_devices(mock_device):
    with mock.patch("stormlog.jax.tracker.jax.devices", return_value=[mock_device]):
        with mock.patch("stormlog.jax.tracker.jax.numpy.zeros"):
            yield


@pytest.mark.jax
def test_tracker_init(mock_jax_devices) -> None:
    """Verify tracker initialization and validation."""
    tracker = JAXMemoryTracker(sampling_interval=0.1, alert_threshold_mb=100)
    assert tracker.sampling_interval == 0.1
    assert tracker.alert_threshold_mb == 100

    with pytest.raises(ValueError, match="Sampling interval must be > 0"):
        JAXMemoryTracker(sampling_interval=-1.0)

    with pytest.raises(ValueError, match="Max history must be >= 1"):
        JAXMemoryTracker(max_history=0)


@pytest.mark.jax
def test_tracker_lifecycle(mock_jax_devices) -> None:
    """Verify start and stop lifecycle."""
    tracker = JAXMemoryTracker(sampling_interval=0.01)

    tracker.start_tracking()
    time.sleep(0.05)
    result = tracker.stop_tracking()

    assert result.duration > 0
    assert result.samples_collected > 0
    assert result.peak_memory_bytes == 1024

    # Session lifecycle
    summary = result.session_summary
    assert summary is not None
    assert summary.status == SESSION_STATUS_COMPLETED
    assert summary.source == "stormlog.jax.tracker"


@pytest.mark.jax
def test_tracker_idempotent_start(mock_jax_devices) -> None:
    """Verify double start is safe."""
    tracker = JAXMemoryTracker(sampling_interval=0.01)
    tracker.start_tracking()
    tracker.start_tracking()  # Should not raise or spawn second thread
    time.sleep(0.05)
    result = tracker.stop_tracking()
    assert result.samples_collected > 0


@pytest.mark.jax
def test_tracker_stop_without_start(mock_jax_devices) -> None:
    """Verify stopping an unstarted tracker is safe."""
    tracker = JAXMemoryTracker(sampling_interval=0.01)
    result = tracker.stop_tracking()
    assert result.samples_collected == 0
    assert result.duration == 0


@pytest.mark.jax
def test_telemetry_event_structure(mock_jax_devices) -> None:
    """Verify built telemetry events have expected fields."""
    tracker = JAXMemoryTracker(sampling_interval=0.01)
    tracker.start_tracking()
    time.sleep(0.05)
    result = tracker.stop_tracking()

    assert len(result.telemetry_events) > 0
    event = result.telemetry_events[0]

    assert event["collector"] == "stormlog.jax.memory_tracker"
    assert event["allocator_allocated_bytes"] == 1024
    assert event["device_total_bytes"] == 10240


@pytest.mark.jax
def test_alerts(mock_jax_devices) -> None:
    """Verify alerting logic."""
    tracker = JAXMemoryTracker(
        sampling_interval=0.01, alert_threshold_mb=0.0001
    )  # 100 bytes threshold

    alert_triggered = False

    def on_alert(alert):
        nonlocal alert_triggered
        alert_triggered = True

    tracker.add_alert_callback(on_alert)

    tracker.start_tracking()
    time.sleep(0.05)
    result = tracker.stop_tracking()

    assert alert_triggered
    assert result.alert_count > 0


@pytest.mark.jax
def test_phase_tracking(mock_jax_devices) -> None:
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

    phase_events = [e for e in events if e.get("event_type") == "phase_change"]
    assert len(phase_events) >= 2  # enter and exit


@pytest.mark.jax
@mock.patch("stormlog.jax.tracker.jax.profiler.save_device_memory_profile")
def test_device_profile_export(mock_save, mock_jax_devices) -> None:
    """Verify JAX device profile export."""
    tracker = JAXMemoryTracker(sampling_interval=0.01, save_device_profile_on_stop=True)
    tracker.start_tracking()
    time.sleep(0.05)
    result = tracker.stop_tracking()

    mock_save.assert_called_once()
    assert result.device_memory_profile_path is not None
    assert result.device_memory_profile_path.endswith(".prof")
