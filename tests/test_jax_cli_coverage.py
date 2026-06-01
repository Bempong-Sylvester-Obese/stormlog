"""Tests for targeting missing lines in JAX CLI coverage."""

import argparse
from typing import Any
from unittest import mock

import pytest

from stormlog.jax import cli


def test_cmd_analyze_no_file(capsys: Any) -> None:
    args = argparse.Namespace(input="nonexistent_file.json")
    with mock.patch("stormlog.jax.cli.Path.exists", return_value=False):
        assert cli.cmd_analyze(args) == 1


def test_cmd_analyze_bad_json(capsys: Any) -> None:
    args = argparse.Namespace(input="bad.json")
    with (
        mock.patch("stormlog.jax.cli.Path.exists", return_value=True),
        mock.patch("stormlog.jax.cli.Path.open", side_effect=Exception("parse error")),
    ):
        assert cli.cmd_analyze(args) == 1


def test_cmd_analyze_success_no_plot(capsys: Any, tmp_path: Any) -> None:
    args = argparse.Namespace(input="ok.json", plot=None, output=str(tmp_path))
    data = {
        "peak_memory": 100,
        "average_memory": 50,
        "duration": 5,
        "alerts": 0,
        "memory_usage": [1, 2],
        "timestamps": [1.0, 2.0],
    }
    with (
        mock.patch("stormlog.jax.cli.Path.exists", return_value=True),
        mock.patch("builtins.open", mock.mock_open(read_data="{}")),
        mock.patch("json.load", return_value=data),
        mock.patch("stormlog.jax.cli.Path.open", mock.mock_open(read_data="{}")),
    ):

        assert cli.cmd_analyze(args) == 0


def test_cmd_diagnose_bad_args(capsys: Any) -> None:
    args = argparse.Namespace(duration=-1, interval=0.5)
    assert cli.cmd_diagnose(args) == 1

    args = argparse.Namespace(duration=5, interval=-1)
    assert cli.cmd_diagnose(args) == 1


def test_cmd_diagnose_jax_not_available(capsys: Any) -> None:
    args = argparse.Namespace()
    with mock.patch("stormlog.jax.cli.JAX_AVAILABLE", False):
        assert cli.cmd_diagnose(args) == 1


def test_cmd_diagnose_wandb_none(capsys: Any) -> None:
    args = argparse.Namespace(duration=5, interval=0.5)
    with mock.patch("stormlog.jax.cli._resolve_wandb_config", return_value=None):
        assert cli.cmd_diagnose(args) == 1


def test_cmd_diagnose_oserror(capsys: Any) -> None:
    args = argparse.Namespace(duration=5, interval=0.5, output=None, device=0)
    config = mock.Mock()
    config.enabled = False
    with (
        mock.patch("stormlog.jax.cli._resolve_wandb_config", return_value=config),
        mock.patch("stormlog.jax.cli.run_diagnose", side_effect=OSError("test error")),
    ):
        assert cli.cmd_diagnose(args) == 1


def test_cmd_diagnose_success(capsys: Any, tmp_path: Any) -> None:
    args = argparse.Namespace(duration=5, interval=0.5, output=None, device=0)
    config = mock.Mock()
    config.enabled = False
    with (
        mock.patch("stormlog.jax.cli._resolve_wandb_config", return_value=config),
        mock.patch("stormlog.jax.cli.run_diagnose", return_value=(tmp_path, 2)),
    ):
        # Test finding manifest
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text('{"risk_detected": true}')

        summary_file = tmp_path / "diagnostic_summary.json"
        summary_file.write_text('{"risk_flags": {"test_risk": true}}')

        assert cli.cmd_diagnose(args) == 2

        # Test 0 exit code
        with mock.patch("stormlog.jax.cli.run_diagnose", return_value=(tmp_path, 0)):
            assert cli.cmd_diagnose(args) == 0

        # Test wandb success
        config.enabled = True
        with (
            mock.patch("stormlog.jax.cli.WANDB_AVAILABLE", True),
            mock.patch("stormlog.jax.cli.run_diagnose", return_value=(tmp_path, 0)),
            mock.patch(
                "stormlog.jax.cli.export_diagnose_bundle_to_wandb"
            ) as mock_wandb,
        ):
            cli.cmd_diagnose(args)
            mock_wandb.assert_called_once()

        # Test wandb error
        with (
            mock.patch("stormlog.jax.cli.WANDB_AVAILABLE", True),
            mock.patch("stormlog.jax.cli.run_diagnose", return_value=(tmp_path, 0)),
            mock.patch(
                "stormlog.jax.cli.export_diagnose_bundle_to_wandb",
                side_effect=Exception("error"),
            ),
            mock.patch("stormlog.jax.cli._warn_wandb_export_failure") as mock_warn,
        ):
            cli.cmd_diagnose(args)
            mock_warn.assert_called_once()


def test_cmd_track_jax_not_available(capsys: Any) -> None:
    args = argparse.Namespace()
    with mock.patch("stormlog.jax.cli.JAX_AVAILABLE", False):
        assert cli.cmd_track(args) == 1


def test_cmd_track_wandb_none(capsys: Any) -> None:
    args = argparse.Namespace()
    with (
        mock.patch("stormlog.jax.cli.JAX_AVAILABLE", True),
        mock.patch("stormlog.jax.cli._resolve_wandb_config", return_value=None),
    ):
        assert cli.cmd_track(args) == 1


def test_cmd_track_keyboard_interrupt(capsys: Any, tmp_path: Any) -> None:
    args = argparse.Namespace(
        interval=1.0,
        threshold=100,
        device=0,
        profile=False,
        output=str(tmp_path / "out.json"),
    )
    config = mock.Mock()
    config.enabled = False
    with (
        mock.patch("stormlog.jax.cli.JAX_AVAILABLE", True),
        mock.patch("stormlog.jax.cli._resolve_wandb_config", return_value=config),
        mock.patch("stormlog.jax.cli.MemoryTracker.start_tracking"),
        mock.patch("time.sleep", side_effect=KeyboardInterrupt),
    ):

        assert cli.cmd_track(args) == 0
        assert (tmp_path / "out.json").exists()


def test_cmd_track_stats_and_wandb(capsys: Any, tmp_path: Any) -> None:
    args = argparse.Namespace(
        interval=1.0,
        threshold=100,
        device=0,
        profile=False,
        output=str(tmp_path / "out2.json"),
    )
    config = mock.Mock()
    config.enabled = True

    # Custom side effect for sleep to run loop once
    sleep_mock = mock.Mock(side_effect=[None, KeyboardInterrupt])

    with (
        mock.patch("stormlog.jax.cli.JAX_AVAILABLE", True),
        mock.patch("stormlog.jax.cli._resolve_wandb_config", return_value=config),
        mock.patch("stormlog.jax.cli.MemoryTracker.start_tracking"),
        mock.patch("time.sleep", sleep_mock),
        mock.patch("stormlog.jax.cli.WANDB_AVAILABLE", True),
        mock.patch("stormlog.jax.cli.export_tracking_run_to_wandb") as mock_wandb,
    ):

        assert cli.cmd_track(args) == 0
        mock_wandb.assert_called_once()


def test_cmd_monitor_jax_not_available(capsys: Any) -> None:
    args = argparse.Namespace()
    with mock.patch("stormlog.jax.cli.JAX_AVAILABLE", False):
        assert cli.cmd_monitor(args) == 1


def test_cmd_monitor_keyboard_interrupt(capsys: Any, tmp_path: Any) -> None:
    args = argparse.Namespace(
        interval=1.0,
        duration=None,
        threshold=100,
        device=0,
        output=str(tmp_path / "mon.json"),
    )
    with (
        mock.patch("stormlog.jax.cli.JAX_AVAILABLE", True),
        mock.patch("stormlog.jax.cli.MemoryTracker.start_tracking"),
        mock.patch(
            "stormlog.jax.cli.MemoryTracker.get_current_memory", return_value=50.0
        ),
        mock.patch("time.sleep", side_effect=KeyboardInterrupt),
    ):

        assert cli.cmd_monitor(args) == 0
        assert (tmp_path / "mon.json").exists()


def test_main_no_args() -> None:
    with mock.patch("sys.argv", ["jaxmemprof"]):
        assert cli.main() == 0


def test_main_unknown_command() -> None:
    with mock.patch("sys.argv", ["jaxmemprof", "unknown"]):
        with pytest.raises(SystemExit):
            cli.main()


def test_main_info_command() -> None:
    with mock.patch("sys.argv", ["jaxmemprof", "info"]):
        # Mock get_system_info to avoid env variations
        with (
            mock.patch(
                "stormlog.jax.cli.get_system_info",
                return_value={
                    "platform": "test",
                    "python_version": "3.12",
                    "cpu_count": 4,
                    "total_memory_gb": 16.0,
                    "available_memory_gb": 8.0,
                    "backend": {"runtime_backend": "gpu", "device_count": 1},
                },
            ),
            mock.patch(
                "stormlog.jax.utils.get_device_info",
                return_value={
                    "kind": "gpu",
                    "memory_stats": {
                        "bytes_in_use": 1024,
                        "peak_bytes_in_use": 2048,
                        "bytes_limit": 4096,
                    },
                },
            ),
        ):
            assert cli.main() == 0


def test_main_analyze_command(tmp_path: Any) -> None:
    with mock.patch("sys.argv", ["jaxmemprof", "analyze", "--input", "test.json"]):
        with mock.patch("stormlog.jax.cli.cmd_analyze", return_value=0) as mock_analyze:
            assert cli.main() == 0
            mock_analyze.assert_called_once()


def test_main_routing_monitor() -> None:
    with (
        mock.patch("sys.argv", ["jaxmemprof", "monitor"]),
        mock.patch("stormlog.jax.cli.cmd_monitor", return_value=0) as mock_cmd,
    ):
        assert cli.main() == 0
        mock_cmd.assert_called_once()


def test_main_routing_track() -> None:
    with (
        mock.patch("sys.argv", ["jaxmemprof", "track", "--output", "out.json"]),
        mock.patch("stormlog.jax.cli.cmd_track", return_value=0) as mock_cmd,
    ):
        assert cli.main() == 0
        mock_cmd.assert_called_once()


def test_main_routing_diagnose() -> None:
    with (
        mock.patch("sys.argv", ["jaxmemprof", "diagnose"]),
        mock.patch("stormlog.jax.cli.cmd_diagnose", return_value=0) as mock_cmd,
    ):
        assert cli.main() == 0
        mock_cmd.assert_called_once()


def test_normalize_telemetry_events() -> None:
    events = cli._normalize_telemetry_events([{"timestamp": 1.0, "memory_mb": 10}], 100)
    assert len(events) == 1
    assert "timestamp_ns" in events[0]


def test_resolve_wandb_config_import_error() -> None:
    args = argparse.Namespace()
    config = mock.Mock()
    config.enabled = True
    with (
        mock.patch("stormlog.jax.cli.wandb_config_from_namespace", return_value=config),
        mock.patch(
            "stormlog.jax.cli.ensure_wandb_available",
            side_effect=ImportError("No wandb"),
        ),
    ):
        assert cli._resolve_wandb_config(args) is None


def test_monitor_dropped_samples(capsys: Any, tmp_path: Any) -> None:
    args = argparse.Namespace(
        interval=1.0,
        duration=None,
        threshold=100,
        device=0,
        output=str(tmp_path / "mon2.json"),
        max_history=10000,
    )
    with (
        mock.patch("stormlog.jax.cli.JAX_AVAILABLE", True),
        mock.patch("stormlog.jax.cli.MemoryTracker") as mock_tracker_cls,
        mock.patch("time.sleep", side_effect=KeyboardInterrupt),
    ):

        mock_tracker = mock_tracker_cls.return_value
        mock_tracker.get_current_memory.return_value = 50.0

        results = mock.Mock()
        results.peak_memory_bytes = 10000
        results.average_memory_bytes = 5000
        results.duration = 10
        results.memory_usage = [5000]
        results.history_dropped_samples = 5
        results.history_window_limit = 10000
        results.history_retained_samples = 5
        results.alert_count = 1
        results.timestamps = [1.0]
        mock_tracker.stop_tracking.return_value = results

        assert cli.cmd_monitor(args) == 0
        assert (tmp_path / "mon2.json").exists()


def test_track_oom_telemetry_branches(capsys: Any, tmp_path: Any) -> None:
    args = argparse.Namespace(
        interval=1.0,
        threshold=100,
        device=0,
        profile=False,
        output=str(tmp_path / "out3.json"),
        telemetry_sink_dir=str(tmp_path / "sink"),
        oom_flight_recorder=True,
        oom_dump_dir="oom_dumps",
        oom_buffer_size=None,
        oom_max_dumps=5,
        oom_max_total_mb=256,
        max_history=10000,
        job_id=None,
        rank=None,
        local_rank=None,
        world_size=None,
        telemetry_flush_seconds=2.0,
        telemetry_rollover_mb=64,
        telemetry_retention_files=8,
        telemetry_retention_total_mb=512,
    )
    config = mock.Mock()
    config.enabled = False
    with (
        mock.patch("stormlog.jax.cli.JAX_AVAILABLE", True),
        mock.patch("stormlog.jax.cli._resolve_wandb_config", return_value=config),
        mock.patch("stormlog.jax.cli.MemoryTracker") as mock_tracker_cls,
        mock.patch("time.sleep", side_effect=KeyboardInterrupt),
    ):

        mock_tracker = mock_tracker_cls.return_value
        mock_tracker.oom_buffer_size = 10000
        mock_tracker.last_oom_dump_path = None
        mock_tracker.get_session_summary.return_value = None

        results = mock.Mock()
        results.peak_memory_bytes = 10000
        results.average_memory_bytes = 5000
        results.duration = 10
        results.memory_usage = [5000]
        results.history_dropped_samples = 5
        results.history_window_limit = 10000
        results.history_retained_samples = 5
        results.history_retained_events = 0
        results.history_dropped_events = 2
        results.history_retained_alerts = 0
        results.history_dropped_alerts = 0
        results.alert_count = 1
        results.timestamps = [1.0]
        results.telemetry_events = []
        results.device_memory_profile_path = None
        mock_tracker.stop_tracking.return_value = results

        mock_tracker.get_statistics.return_value = {
            "current_memory_mb": 50,
            "collector_health_status": "unhealthy",
            "collector_next_retry_epoch_s": 0,
            "collector_last_error": "timeout",
            "history_dropped_samples": 5,
            "history_dropped_events": 2,
        }

        assert cli.cmd_track(args) == 0


def test_cli_wandb_fallback() -> None:
    # Test lines 24-34 for wandb fallback when missing
    with mock.patch("stormlog.jax.cli.WANDB_AVAILABLE", False):
        parser = argparse.ArgumentParser()
        cli.add_wandb_arguments(parser)
        config = cli.wandb_config_from_namespace(argparse.Namespace())
        assert config.enabled is False
