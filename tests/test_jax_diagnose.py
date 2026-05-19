"""Tests for JAX diagnostic bundle builder."""

import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

pytest.importorskip("jax")

from stormlog.jax.diagnose import (
    build_diagnostic_summary,
    collect_environment,
    run_diagnose,
    run_timeline_capture,
)
from tests.jax_test_helpers import jax_mark


@jax_mark
def test_collect_environment() -> None:
    """Verify collect_environment returns expected structure."""
    env = collect_environment(device_index=0)
    assert "system" in env
    assert "backend" in env
    assert "device" in env
    assert "fragmentation" in env
    assert "JAX does not expose fragmentation" in env["fragmentation"]["note"]


@jax_mark
def test_run_timeline_capture_zero_duration() -> None:
    """Verify run_timeline_capture with zero duration returns empty."""
    timeline = run_timeline_capture(0, 0.0, 0.5)
    assert timeline["timestamps"] == []
    assert timeline["allocated"] == []
    assert timeline["reserved"] == []


@jax_mark
def test_run_timeline_capture_positive_duration() -> None:
    """Verify run_timeline_capture collects some samples."""
    # Use a small interval to ensure at least one sample
    timeline = run_timeline_capture(0, 0.2, 0.05)
    assert len(timeline["timestamps"]) > 0
    assert len(timeline["allocated"]) > 0
    assert len(timeline["reserved"]) > 0
    # For JAX, allocated == reserved
    assert timeline["allocated"] == timeline["reserved"]


@jax_mark
@mock.patch("stormlog.jax.diagnose.get_device_info")
@mock.patch("stormlog.jax.diagnose.get_backend_info")
def test_build_diagnostic_summary(mock_backend: Any, mock_device: Any) -> None:
    """Verify build_diagnostic_summary returns a valid payload."""
    mock_backend.return_value = {"runtime_backend": "gpu"}
    mock_device.return_value = {
        "memory_stats": {
            "bytes_in_use": 1000,
            "peak_bytes_in_use": 2000,
            "bytes_limit": 10000,
        }
    }

    summary, risk_detected = build_diagnostic_summary(0)

    assert summary["backend"] == "gpu"
    assert summary["allocated_bytes"] == 1000
    assert summary["reserved_bytes"] == 1000
    assert summary["peak_bytes"] == 2000
    assert summary["total_bytes"] == 10000
    assert summary["allocator_gap_bytes"] == 0
    assert summary["utilization_ratio"] == 0.1
    assert summary["num_ooms"] == 0
    assert not risk_detected


@jax_mark
@mock.patch("stormlog.jax.diagnose.get_device_info")
@mock.patch("stormlog.jax.diagnose.get_backend_info")
def test_build_diagnostic_summary_high_utilization(
    mock_backend: Any, mock_device: Any
) -> None:
    """Verify high utilization triggers memory risk."""
    mock_backend.return_value = {"runtime_backend": "gpu"}
    mock_device.return_value = {
        "memory_stats": {
            "bytes_in_use": 9000,
            "peak_bytes_in_use": 9000,
            "bytes_limit": 10000,
        }
    }

    summary, risk_detected = build_diagnostic_summary(0)
    assert risk_detected is True
    assert summary["risk_flags"]["high_utilization"] is True


@jax_mark
def test_run_diagnose(tmp_path: Path) -> None:
    """Verify full diagnose orchestrator produces valid artifacts."""
    artifact_dir, exit_code = run_diagnose(
        output=str(tmp_path),
        device_index=0,
        duration=0.1,
        interval=0.05,
        command_line="test_cmd",
    )

    assert artifact_dir.exists()
    assert artifact_dir.is_dir()
    assert exit_code in (0, 2)

    # Check files
    env_file = artifact_dir / "environment.json"
    timeline_file = artifact_dir / "telemetry_timeline.json"
    summary_file = artifact_dir / "diagnostic_summary.json"
    manifest_file = artifact_dir / "manifest.json"

    assert env_file.exists()
    assert timeline_file.exists()
    assert summary_file.exists()
    assert manifest_file.exists()

    with open(manifest_file) as f:
        manifest = json.load(f)

    assert manifest["schema_version"] == 2
    assert "session_id" in manifest
    assert "test_cmd" in manifest["command_line"]
