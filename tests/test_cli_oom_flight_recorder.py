"""CLI tests for OOM flight recorder track options."""

from __future__ import annotations

import sys
from argparse import Namespace
from contextlib import nullcontext
from typing import Any

import pytest

import stormlog.cli as gpumemprof_cli


def test_main_parses_oom_track_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_cmd_track(args: object) -> None:
        captured["args"] = args

    monkeypatch.setattr(gpumemprof_cli, "cmd_track", _fake_cmd_track)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpumemprof",
            "track",
            "--oom-flight-recorder",
            "--oom-dump-dir",
            "my_oom_dir",
            "--oom-buffer-size",
            "512",
            "--oom-max-dumps",
            "7",
            "--oom-max-total-mb",
            "2048",
            "--job-id",
            "train-42",
            "--rank",
            "2",
            "--local-rank",
            "0",
            "--world-size",
            "8",
        ],
    )

    gpumemprof_cli.main()

    args = captured["args"]
    assert args.oom_flight_recorder is True
    assert args.oom_dump_dir == "my_oom_dir"
    assert args.oom_buffer_size == 512
    assert args.oom_max_dumps == 7
    assert args.oom_max_total_mb == 2048
    assert args.job_id == "train-42"
    assert args.rank == 2
    assert args.local_rank == 0
    assert args.world_size == 8


def test_cmd_track_passes_oom_config_to_memorytracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, Any] = {}

    class _FakeTracker:
        def __init__(self, **kwargs: object) -> None:
            created.update(kwargs)
            self.max_events = 10_000
            self.oom_buffer_size = kwargs.get("oom_buffer_size") or self.max_events
            self.last_oom_dump_path = None

        def set_threshold(self, name: str, value: float) -> None:
            created[f"threshold_{name}"] = value

        def add_alert_callback(self, callback: object) -> None:
            created["alert_callback_registered"] = callback is not None

        def start_tracking(self) -> None:
            created["started"] = True

        def stop_tracking(self) -> None:
            created["stopped"] = True

        def get_statistics(self) -> dict[str, object]:
            return {
                "current_memory_allocated": 0,
                "peak_memory": 0,
                "memory_utilization_percent": 0,
                "total_events": 0,
            }

        def export_events(self, output: str, fmt: str) -> None:
            created["export"] = (output, fmt)

        def capture_oom(
            self, context: str = "runtime", metadata: object = None
        ) -> object:
            created["capture_context"] = context
            created["capture_metadata"] = metadata
            return nullcontext()

    monkeypatch.setattr(gpumemprof_cli, "MemoryTracker", _FakeTracker)
    monkeypatch.setattr(gpumemprof_cli, "MemoryWatchdog", lambda tracker: None)
    monkeypatch.setattr(
        gpumemprof_cli, "get_system_info", lambda: {"detected_backend": "cuda"}
    )

    def _interrupt(_: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(gpumemprof_cli.time, "sleep", _interrupt)

    args = Namespace(
        device=0,
        duration=None,
        interval=0.25,
        output=None,
        format="json",
        watchdog=False,
        warning_threshold=70.0,
        critical_threshold=90.0,
        oom_flight_recorder=True,
        oom_dump_dir="oom_test_dir",
        oom_buffer_size=1024,
        oom_max_dumps=9,
        oom_max_total_mb=512,
        job_id="train-42",
        rank=2,
        local_rank=0,
        world_size=8,
    )

    gpumemprof_cli.cmd_track(args)

    assert created["device"] == 0
    assert created["sampling_interval"] == 0.25
    assert created["enable_oom_flight_recorder"] is True
    assert created["oom_dump_dir"] == "oom_test_dir"
    assert created["oom_buffer_size"] == 1024
    assert created["oom_max_dumps"] == 9
    assert created["oom_max_total_mb"] == 512
    assert created["job_id"] == "train-42"
    assert created["rank"] == 2
    assert created["local_rank"] == 0
    assert created["world_size"] == 8
    assert created["capture_context"] == "stormlog.track"
    assert created["capture_metadata"]["command"] == "track"
    assert created["capture_metadata"]["runtime_backend"] == "cuda"


def test_cmd_track_reports_collector_health_in_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _FakeTracker:
        def __init__(self, **kwargs: object) -> None:
            self.last_oom_dump_path = None
            self._retry_at = gpumemprof_cli.time.time() + 5.0
            _ = kwargs

        def set_threshold(self, name: str, value: float) -> None:
            _ = (name, value)

        def add_alert_callback(self, callback: object) -> None:
            _ = callback

        def start_tracking(self) -> None:
            return None

        def stop_tracking(self) -> None:
            return None

        def get_statistics(self) -> dict[str, object]:
            return {
                "current_memory_allocated": None,
                "peak_memory": 0,
                "memory_utilization_percent": None,
                "total_events": 1,
                "collector_health_status": "unhealthy",
                "collector_last_error": "collector unavailable",
                "collector_next_retry_epoch_s": self._retry_at,
            }

        def capture_oom(
            self, context: str = "runtime", metadata: object = None
        ) -> object:
            _ = (context, metadata)
            return nullcontext()

    monkeypatch.setattr(gpumemprof_cli, "MemoryTracker", _FakeTracker)
    monkeypatch.setattr(gpumemprof_cli, "MemoryWatchdog", lambda tracker: None)
    monkeypatch.setattr(
        gpumemprof_cli, "get_system_info", lambda: {"detected_backend": "cuda"}
    )

    def _interrupt(_: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(gpumemprof_cli.time, "sleep", _interrupt)

    gpumemprof_cli.cmd_track(
        Namespace(
            device=0,
            duration=None,
            interval=0.25,
            output=None,
            format="json",
            watchdog=False,
            warning_threshold=70.0,
            critical_threshold=90.0,
            oom_flight_recorder=False,
            oom_dump_dir="oom_test_dir",
            oom_buffer_size=None,
            oom_max_dumps=9,
            oom_max_total_mb=512,
            job_id=None,
            rank=None,
            local_rank=None,
            world_size=None,
        )
    )

    output = capsys.readouterr().out
    assert "Health: unhealthy" in output
    assert "Collector health: unhealthy" in output
    assert "Last collector error: collector unavailable" in output


def test_cmd_monitor_handles_unavailable_current_memory_for_mps(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    current_time = {"value": 0.0}

    class _FakeTracker:
        def __init__(self, **kwargs: object) -> None:
            self.started = False
            self.stopped = False
            _ = kwargs

        def start_tracking(self) -> None:
            self.started = True

        def stop_tracking(self) -> None:
            self.stopped = True

        def get_statistics(self) -> dict[str, object]:
            return {
                "current_memory_allocated": None,
                "peak_memory": 0,
            }

        def get_events(self) -> list[object]:
            return []

        def export_events(self, output: str, fmt: str) -> None:
            _ = (output, fmt)

    monkeypatch.setattr(
        gpumemprof_cli, "get_system_info", lambda: {"detected_backend": "mps"}
    )
    monkeypatch.setattr(
        gpumemprof_cli,
        "_import_runtime_symbols",
        lambda module, names, command: (_FakeTracker,),
    )
    monkeypatch.setattr(gpumemprof_cli.time, "time", lambda: current_time["value"])

    def _sleep(_: float) -> None:
        current_time["value"] += 1.1

    monkeypatch.setattr(gpumemprof_cli.time, "sleep", _sleep)

    gpumemprof_cli.cmd_monitor(
        Namespace(
            device=None,
            duration=1.0,
            interval=0.25,
            output=None,
            format="json",
        )
    )

    output = capsys.readouterr().out
    assert "Current Memory: -" in output
