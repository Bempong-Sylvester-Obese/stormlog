"""Backend-aware device memory collector abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import torch


@dataclass(frozen=True)
class DeviceMemorySample:
    """Normalized device-memory sample produced by a backend collector."""

    allocated_bytes: int
    reserved_bytes: int
    used_bytes: int
    free_bytes: Optional[int]
    total_bytes: Optional[int]
    active_bytes: Optional[int]
    inactive_bytes: Optional[int]
    device_id: int


@dataclass(frozen=True)
class DeviceMemorySampleResult:
    """Device-memory sample plus diagnostics about partial/core collection failures."""

    sample: Optional[DeviceMemorySample]
    partial_fields: tuple[str, ...] = ()
    errors: dict[str, str] = field(default_factory=dict)
    core_error: Optional[str] = None

    @property
    def is_partial(self) -> bool:
        return self.sample is not None and bool(self.partial_fields)

    @property
    def is_core_failure(self) -> bool:
        return self.sample is None


class DeviceMemoryCollector(ABC):
    """Backend-specific collector contract for device memory signals."""

    @abstractmethod
    def name(self) -> str:
        """Return runtime backend name (cuda, rocm, mps)."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether this collector can sample in the current runtime."""

    @abstractmethod
    def sample(self) -> DeviceMemorySample:
        """Collect a single normalized memory sample."""

    def sample_with_diagnostics(self) -> DeviceMemorySampleResult:
        """Collect a sample while preserving core-failure diagnostics."""
        try:
            return DeviceMemorySampleResult(sample=self.sample())
        except Exception as exc:
            return DeviceMemorySampleResult(
                sample=None,
                errors={"core_metrics": str(exc)},
                core_error=str(exc),
            )

    @abstractmethod
    def capabilities(self) -> Dict[str, Any]:
        """Describe backend capability signals for telemetry metadata."""


def _is_mps_available() -> bool:
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None:
        return False
    try:
        return bool(mps_backend.is_available())
    except Exception:
        return False


def _is_rocm_runtime() -> bool:
    hip_version = getattr(torch.version, "hip", None)
    return bool(torch.cuda.is_available() and hip_version)


def detect_torch_runtime_backend() -> str:
    """Return the active torch runtime backend in this environment."""
    if torch.cuda.is_available():
        return "rocm" if _is_rocm_runtime() else "cuda"
    if _is_mps_available():
        return "mps"
    return "cpu"


def _resolve_device(device: Union[str, int, torch.device, None]) -> torch.device:
    if device is None:
        backend = detect_torch_runtime_backend()
        if backend in {"cuda", "rocm"}:
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        if backend == "mps":
            return torch.device("mps")
        raise RuntimeError("No supported GPU backend is available")
    if isinstance(device, int):
        backend = detect_torch_runtime_backend()
        if backend not in {"cuda", "rocm"}:
            raise ValueError(
                "Integer device IDs are only supported for CUDA/ROCm backends"
            )
        return torch.device(f"cuda:{device}")
    if isinstance(device, str):
        return torch.device(device)
    return device


class CudaDeviceCollector(DeviceMemoryCollector):
    """Collector for NVIDIA CUDA runtime memory counters."""

    telemetry_collector = "stormlog.cuda_tracker"

    def __init__(self, device: Union[str, int, torch.device, None] = None) -> None:
        self.device = _resolve_device(device)
        if self.device.type != "cuda":
            raise ValueError("CUDA collector requires a CUDA device")

    def name(self) -> str:
        return "cuda"

    def is_available(self) -> bool:
        return bool(torch.cuda.is_available() and not _is_rocm_runtime())

    def sample(self) -> DeviceMemorySample:
        result = self.sample_with_diagnostics()
        if result.sample is None:
            raise RuntimeError(result.core_error or "CUDA sample collection failed")
        return result.sample

    def sample_with_diagnostics(self) -> DeviceMemorySampleResult:
        device_index = (
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        )
        try:
            allocated = int(torch.cuda.memory_allocated(self.device))
            reserved = int(torch.cuda.memory_reserved(self.device))
        except Exception as exc:
            return DeviceMemorySampleResult(
                sample=None,
                errors={"core_metrics": str(exc)},
                core_error=str(exc),
            )

        used = max(allocated, reserved)
        total: Optional[int] = None
        free: Optional[int] = None
        active: Optional[int] = None
        inactive: Optional[int] = None
        partial_fields: list[str] = []
        errors: dict[str, str] = {}

        try:
            total = int(torch.cuda.get_device_properties(self.device).total_memory)
            free = max(total - used, 0)
        except Exception as exc:
            message = str(exc)
            partial_fields.extend(["device_total_bytes", "device_free_bytes"])
            errors["device_total_bytes"] = message
            errors["device_free_bytes"] = message

        try:
            stats = torch.cuda.memory_stats(self.device)
            active = int(stats.get("active_bytes.all.current", 0))
            inactive = int(stats.get("inactive_split_bytes.all.current", 0))
        except Exception as exc:
            message = str(exc)
            partial_fields.extend(
                ["allocator_active_bytes", "allocator_inactive_bytes"]
            )
            errors["allocator_active_bytes"] = message
            errors["allocator_inactive_bytes"] = message

        return DeviceMemorySampleResult(
            sample=DeviceMemorySample(
                allocated_bytes=allocated,
                reserved_bytes=reserved,
                used_bytes=used,
                free_bytes=free,
                total_bytes=total,
                active_bytes=active,
                inactive_bytes=inactive,
                device_id=device_index,
            ),
            partial_fields=tuple(dict.fromkeys(partial_fields)),
            errors=errors,
        )

    def capabilities(self) -> Dict[str, Any]:
        return {
            "backend": self.name(),
            "supports_device_total": True,
            "supports_device_free": True,
            "sampling_source": "torch.cuda.memory_allocated/reserved",
            "telemetry_collector": self.telemetry_collector,
        }


class ROCmDeviceCollector(CudaDeviceCollector):
    """Collector for ROCm runtimes surfaced through torch.cuda APIs."""

    telemetry_collector = "stormlog.rocm_tracker"

    def name(self) -> str:
        return "rocm"

    def is_available(self) -> bool:
        return _is_rocm_runtime()

    def capabilities(self) -> Dict[str, Any]:
        capabilities = super().capabilities()
        capabilities.update(
            {
                "backend": self.name(),
                "sampling_source": "torch.cuda.memory_* (HIP runtime)",
                "telemetry_collector": self.telemetry_collector,
            }
        )
        return capabilities


class MPSDeviceCollector(DeviceMemoryCollector):
    """Collector for Apple Metal (MPS) runtime counters."""

    telemetry_collector = "stormlog.mps_tracker"

    def __init__(self, device: Union[str, int, torch.device, None] = None) -> None:
        resolved = _resolve_device(device)
        if resolved.type != "mps":
            raise ValueError("MPS collector requires an MPS device")
        self.device = resolved

    def name(self) -> str:
        return "mps"

    def is_available(self) -> bool:
        return _is_mps_available()

    def sample(self) -> DeviceMemorySample:
        result = self.sample_with_diagnostics()
        if result.sample is None:
            raise RuntimeError(result.core_error or "MPS sample collection failed")
        return result.sample

    def sample_with_diagnostics(self) -> DeviceMemorySampleResult:
        import torch.mps as torch_mps

        try:
            allocated = int(torch_mps.current_allocated_memory())
            reserved = int(torch_mps.driver_allocated_memory())
        except Exception as exc:
            return DeviceMemorySampleResult(
                sample=None,
                errors={"core_metrics": str(exc)},
                core_error=str(exc),
            )

        used = max(allocated, reserved)

        total: Optional[int] = None
        partial_fields: list[str] = []
        errors: dict[str, str] = {}
        if hasattr(torch_mps, "recommended_max_memory"):
            try:
                # MPS does not expose a strict physical-total API here; this is the
                # best runtime approximation currently available from torch.
                raw_total = int(torch_mps.recommended_max_memory())
                total = raw_total if raw_total > 0 else None
            except Exception as exc:
                message = str(exc)
                partial_fields.extend(["device_total_bytes", "device_free_bytes"])
                errors["device_total_bytes"] = message
                errors["device_free_bytes"] = message
        free = max(total - used, 0) if total is not None else None

        return DeviceMemorySampleResult(
            sample=DeviceMemorySample(
                allocated_bytes=allocated,
                reserved_bytes=reserved,
                used_bytes=used,
                free_bytes=free,
                total_bytes=total,
                active_bytes=None,
                inactive_bytes=None,
                device_id=0,
            ),
            partial_fields=tuple(dict.fromkeys(partial_fields)),
            errors=errors,
        )

    def capabilities(self) -> Dict[str, Any]:
        import torch.mps as torch_mps

        supports_total = hasattr(torch_mps, "recommended_max_memory")
        return {
            "backend": self.name(),
            "supports_device_total": supports_total,
            "supports_device_free": supports_total,
            "sampling_source": "torch.mps.current_allocated_memory/driver_allocated_memory",
            "telemetry_collector": self.telemetry_collector,
        }


def build_device_memory_collector(
    device: Union[str, int, torch.device, None] = None,
) -> DeviceMemoryCollector:
    """Build a backend collector for CUDA/ROCm/MPS runtime environments."""
    resolved = _resolve_device(device)
    if resolved.type == "cuda":
        if _is_rocm_runtime():
            return ROCmDeviceCollector(resolved)
        return CudaDeviceCollector(resolved)
    if resolved.type == "mps":
        return MPSDeviceCollector(resolved)
    raise ValueError("Only CUDA/ROCm and MPS devices are supported for tracking")


__all__ = [
    "DeviceMemoryCollector",
    "DeviceMemorySample",
    "DeviceMemorySampleResult",
    "CudaDeviceCollector",
    "ROCmDeviceCollector",
    "MPSDeviceCollector",
    "build_device_memory_collector",
    "detect_torch_runtime_backend",
]
