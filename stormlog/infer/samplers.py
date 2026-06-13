"""Best-effort system telemetry samplers for inference profiling."""

from __future__ import annotations

import os
import subprocess
import time
from typing import Protocol

from .events import InferenceSystemSample


class SystemSampler(Protocol):
    """System telemetry sampler contract."""

    name: str

    def sample(self, *, session_id: str) -> InferenceSystemSample | None:
        """Return one telemetry sample, or None when unavailable."""


class NoopSampler:
    """Sampler used for remote endpoints or unsupported systems."""

    name = "noop"

    def sample(self, *, session_id: str) -> InferenceSystemSample | None:
        return None


class ProcessSampler:
    """Sample the current Stormlog process RSS."""

    name = "psutil"

    def __init__(self) -> None:
        import psutil

        self._process = psutil.Process(os.getpid())

    def sample(self, *, session_id: str) -> InferenceSystemSample | None:
        return InferenceSystemSample(
            session_id=session_id,
            timestamp_ns=time.time_ns(),
            sampler=self.name,
            process_rss_bytes=int(self._process.memory_info().rss),
        )


class NvidiaSmiSampler:
    """Sample NVIDIA device memory and utilization through nvidia-smi."""

    name = "nvidia-smi"

    def __init__(self, *, device_id: int = 0) -> None:
        self.device_id = device_id

    def sample(self, *, session_id: str) -> InferenceSystemSample | None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    (
                        "--query-gpu=index,name,memory.total,memory.used,"
                        "memory.free,utilization.gpu"
                    ),
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            values = [value.strip() for value in line.split(",")]
            if len(values) < 6:
                continue
            try:
                device_id = int(values[0])
                total_mb = int(values[2])
                used_mb = int(values[3])
                free_mb = int(values[4])
                gpu_utilization = float(values[5])
            except ValueError:
                continue
            if device_id != self.device_id:
                continue
            return InferenceSystemSample(
                session_id=session_id,
                timestamp_ns=time.time_ns(),
                sampler=self.name,
                device_id=device_id,
                device_name=values[1],
                device_total_bytes=total_mb * 1024 * 1024,
                device_used_bytes=used_mb * 1024 * 1024,
                device_free_bytes=free_mb * 1024 * 1024,
                gpu_utilization_percent=gpu_utilization,
            )
        return None


def build_system_sampler(name: str) -> SystemSampler:
    """Build a system sampler by name."""
    normalized = name.strip().lower()
    if normalized in {"none", "noop", "remote"}:
        return NoopSampler()
    if normalized in {"psutil", "process"}:
        return ProcessSampler()
    if normalized in {"nvidia-smi", "nvidia", "gpu"}:
        return NvidiaSmiSampler()
    if normalized != "auto":
        raise ValueError(
            "--system-sampler must be one of auto, none, psutil, nvidia-smi"
        )

    try:
        sample = NvidiaSmiSampler()
        if sample.sample(session_id="probe") is not None:
            return sample
    except Exception:
        pass
    try:
        return ProcessSampler()
    except Exception:
        return NoopSampler()
