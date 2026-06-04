"""JAX CLI tests for OOM flight recorder track options."""

from __future__ import annotations

import sys
from argparse import Namespace
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

from stormlog.session import create_session_summary
from tests.jax_test_helpers import jax_fixture


@jax_fixture
def jax_cli(monkeypatch: pytest.MonkeyPatch) -> Any:
    import importlib

    fake_jax = SimpleNamespace(
        default_backend=lambda: "cpu",
        local_devices=lambda: [],
        profiler=SimpleNamespace(save_device_memory_profile=lambda path: None),
    )
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    return importlib.import_module("stormlog.jax.cli")


def test_jax_main_parses_oom_track_flags(
    monkeypatch: pytest.MonkeyPatch,
    jax_cli: Any,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_cmd_track(args: object) -> int:
        captured["args"] = args
        return 0

    monkeypatch.setattr(jax_cli, "cmd_track", _fake_cmd_track)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "jaxmemprof",
            "track",
            "--output",
            "track.json",
            "--oom-flight-recorder",
            "--oom-dump-dir",
            "jax_oom_dir",
            "--oom-buffer-size",
            "512",
            "--oom-max-dumps",
            "7",
            "--oom-max-total-mb",
            "2048",
        ],
    )

    assert jax_cli.main() == 0

    args = captured["args"]
    assert args.oom_flight_recorder is True
    assert args.oom_dump_dir == "jax_oom_dir"
    assert args.oom_buffer_size == 512
    assert args.oom_max_dumps == 7
    assert args.oom_max_total_mb == 2048


def test_jax_cmd_track_passes_oom_config_and_wraps_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    jax_cli: Any,
) -> None:
    created: dict[str, Any] = {}

    class _FakeTracker:
        oom_buffer_size = 1024
        last_oom_dump_path = None

        def __init__(self, **kwargs: object) -> None:
            created.update(kwargs)

        def add_alert_callback(self, callback: object) -> None:
            created["alert_callback_registered"] = callback is not None

        def start_tracking(self) -> None:
            created["started"] = True

        def stop_tracking(self) -> object:
            created["stopped"] = True
            return SimpleNamespace(
                peak_memory_bytes=0,
                average_memory_bytes=0,
                duration=0.0,
                memory_usage=[],
                timestamps=[],
                alert_count=0,
                telemetry_events=[],
                device_memory_profile_path=None,
            )

        def get_statistics(self) -> dict[str, object]:
            return {"total_events": 0, "peak_memory_mb": 0.0}

        def capture_oom(
            self,
            context: str = "runtime",
            metadata: object = None,
        ) -> object:
            created["capture_context"] = context
            created["capture_metadata"] = metadata
            return nullcontext()

        def get_session_summary(self) -> object:
            return None

    monkeypatch.setattr(jax_cli, "JAX_AVAILABLE", True)
    monkeypatch.setattr(jax_cli, "MemoryTracker", _FakeTracker, raising=False)
    monkeypatch.setattr(
        jax_cli,
        "wandb_config_from_namespace",
        lambda args: Namespace(enabled=False),
    )
    monkeypatch.setattr(
        jax_cli.time,
        "sleep",
        lambda _: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    assert (
        jax_cli.cmd_track(
            Namespace(
                interval=0.25,
                threshold=4000,
                device=0,
                profile=False,
                job_id="train-42",
                rank=2,
                local_rank=0,
                world_size=8,
                output=str(tmp_path / "track.json"),
                telemetry_sink_dir="telemetry_sink",
                telemetry_flush_seconds=3.5,
                telemetry_rollover_mb=32,
                telemetry_retention_files=4,
                telemetry_retention_total_mb=128,
                max_history=10_000,
                oom_flight_recorder=True,
                oom_dump_dir="jax_oom_dir",
                oom_buffer_size=1024,
                oom_max_dumps=9,
                oom_max_total_mb=512,
            )
        )
        == 0
    )

    assert created["sampling_interval"] == 0.25
    assert created["enable_oom_flight_recorder"] is True
    assert created["oom_dump_dir"] == "jax_oom_dir"
    assert created["oom_buffer_size"] == 1024
    assert created["oom_max_dumps"] == 9
    assert created["oom_max_total_mb"] == 512
    assert created["job_id"] == "train-42"
    assert created["rank"] == 2
    assert created["local_rank"] == 0
    assert created["world_size"] == 8
    assert created["capture_context"] == "jaxmemprof.track"
    assert created["capture_metadata"] == {
        "command": "track",
        "runtime_backend": "jax",
    }


def test_jax_cmd_track_exports_oom_bundle_to_wandb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    jax_cli: Any,
) -> None:
    exported: dict[str, Any] = {}
    wandb_config = Namespace(enabled=True)
    session_summary = create_session_summary(
        source="stormlog.jax.tracker",
        session_id="session-12345678",
        job_id="train-42",
    )

    class _FakeTracker:
        oom_buffer_size = 10_000
        last_oom_dump_path = "jax_oom_bundle"

        def __init__(self, **kwargs: object) -> None:
            _ = kwargs

        def add_alert_callback(self, callback: object) -> None:
            _ = callback

        def start_tracking(self) -> None:
            return None

        def stop_tracking(self) -> object:
            return SimpleNamespace(
                peak_memory_bytes=1024,
                average_memory_bytes=512,
                duration=1.0,
                memory_usage=[],
                timestamps=[],
                alert_count=0,
                telemetry_events=[{"timestamp": 1.0, "type": "sample", "memory_mb": 0}],
                device_memory_profile_path=None,
            )

        def get_statistics(self) -> dict[str, object]:
            return {"total_events": 1, "peak_memory_mb": 0.001}

        def capture_oom(
            self,
            context: str = "runtime",
            metadata: object = None,
        ) -> object:
            _ = (context, metadata)
            return nullcontext()

        def get_session_summary(self) -> object:
            return session_summary

    monkeypatch.setattr(jax_cli, "JAX_AVAILABLE", True)
    monkeypatch.setattr(jax_cli, "MemoryTracker", _FakeTracker, raising=False)
    monkeypatch.setattr(jax_cli, "WANDB_AVAILABLE", True)
    monkeypatch.setattr(
        jax_cli,
        "wandb_config_from_namespace",
        lambda args: wandb_config,
    )
    monkeypatch.setattr(
        jax_cli,
        "ensure_wandb_available",
        lambda config: exported.setdefault("ensured", config),
        raising=False,
    )
    monkeypatch.setattr(
        jax_cli,
        "export_tracking_run_to_wandb",
        lambda config, **kwargs: exported.update(config=config, kwargs=kwargs),
        raising=False,
    )
    monkeypatch.setattr(
        jax_cli.time,
        "sleep",
        lambda _: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    assert (
        jax_cli.cmd_track(
            Namespace(
                interval=0.25,
                threshold=4000,
                device=0,
                profile=False,
                job_id="train-42",
                rank=2,
                local_rank=0,
                world_size=8,
                output=str(tmp_path / "track.json"),
                telemetry_sink_dir="telemetry_sink",
                telemetry_flush_seconds=3.5,
                telemetry_rollover_mb=32,
                telemetry_retention_files=4,
                telemetry_retention_total_mb=128,
                max_history=10_000,
                oom_flight_recorder=False,
                oom_dump_dir="jax_oom_dir",
                oom_buffer_size=None,
                oom_max_dumps=9,
                oom_max_total_mb=512,
            )
        )
        == 0
    )

    assert exported["ensured"] is wandb_config
    assert exported["config"] is wandb_config
    assert exported["kwargs"]["command_name"] == "jaxmemprof-track"
    assert exported["kwargs"]["session_summary"] == session_summary
    assert exported["kwargs"]["events"] == [
        {"timestamp": 1.0, "type": "sample", "memory_mb": 0}
    ]
    assert exported["kwargs"]["output_path"] == str(tmp_path / "track.json")
    assert exported["kwargs"]["telemetry_sink_dir"] == "telemetry_sink"
    assert exported["kwargs"]["oom_dump_path"] == "jax_oom_bundle"
