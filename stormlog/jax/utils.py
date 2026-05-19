"""Utility functions for JAX memory profiling.

This module provides helper functions for JAX device discovery,
memory formatting, system information, and environment validation.
"""

from __future__ import annotations

import logging
import os
import platform
from typing import Any, Dict, List, Optional, Union

from .jax_env import configure_jax_logging

configure_jax_logging()

try:
    import jax

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

logger = logging.getLogger(__name__)

_JAX_INSTALL_GUIDANCE = (
    "JAX is required for this feature. Install with " "`pip install 'stormlog[jax]'`."
)


def jax_is_available() -> bool:
    """Return True when JAX is importable."""
    return JAX_AVAILABLE


def detect_jax_backend() -> str:
    """Return the active JAX backend name.

    Returns one of ``'gpu'``, ``'tpu'``, ``'cpu'``, or ``'cpu'``
    if JAX is not installed.
    """
    if not JAX_AVAILABLE:
        return "cpu"
    try:
        return str(jax.default_backend())
    except Exception as exc:
        logger.debug("JAX backend detection failed: %s", exc)
        return "cpu"


def get_device_info(device_index: int = 0) -> Dict[str, Any]:
    """Return device kind, platform, and live memory statistics.

    Args:
        device_index: Index into ``jax.local_devices()`` (default 0).

    Returns:
        Dictionary with keys ``kind``, ``platform``, ``device_id``,
        ``process_index``, ``memory_stats`` (raw dict from
        ``device.memory_stats()``), and ``client``.
    """
    if not JAX_AVAILABLE:
        return {
            "kind": "cpu",
            "platform": "cpu",
            "device_id": 0,
            "process_index": 0,
            "memory_stats": {},
            "client": None,
            "error": "JAX not available",
        }

    try:
        devices = jax.local_devices()
        if device_index >= len(devices):
            return {
                "kind": "unknown",
                "platform": detect_jax_backend(),
                "device_id": device_index,
                "process_index": 0,
                "memory_stats": {},
                "client": None,
                "error": f"Device index {device_index} out of range "
                f"(found {len(devices)} devices)",
            }

        device = devices[device_index]
        memory_stats: Dict[str, Any] = {}
        try:
            raw_stats = device.memory_stats()
            if raw_stats is not None:
                memory_stats = dict(raw_stats)
        except Exception as exc:
            logger.debug(
                "Could not read memory_stats for device %d: %s", device_index, exc
            )

        return {
            "kind": str(getattr(device, "device_kind", "unknown")),
            "platform": str(device.platform),
            "device_id": getattr(device, "id", device_index),
            "process_index": getattr(device, "process_index", 0),
            "memory_stats": memory_stats,
            "client": str(getattr(device, "client", None)),
        }
    except Exception as exc:
        logger.debug("get_device_info failed: %s", exc)
        return {
            "kind": "unknown",
            "platform": detect_jax_backend(),
            "device_id": device_index,
            "process_index": 0,
            "memory_stats": {},
            "client": None,
            "error": str(exc),
        }


def get_backend_info() -> Dict[str, Any]:
    """Return backend diagnostics for JAX.

    Returns a dictionary with the JAX runtime backend classification
    and platform details.
    """
    info: Dict[str, Any] = {
        "runtime_backend": detect_jax_backend(),
        "jax_available": JAX_AVAILABLE,
        "device_count": 0,
        "devices": [],
    }

    if not JAX_AVAILABLE:
        return info

    try:
        devices = jax.local_devices()
        info["device_count"] = len(devices)
        info["devices"] = [
            {
                "id": getattr(d, "id", i),
                "kind": str(getattr(d, "device_kind", "unknown")),
                "platform": str(d.platform),
            }
            for i, d in enumerate(devices)
        ]
    except Exception as exc:
        logger.debug("Could not enumerate JAX devices: %s", exc)

    return info


def get_system_info() -> Dict[str, Any]:
    """Return full system and JAX environment report.

    Includes JAX version, device list, platform, Python version,
    CPU count, and system memory statistics.
    """
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "jax_version": "Not installed",
        "jax_available": JAX_AVAILABLE,
        "cpu_count": os.cpu_count(),
        "total_memory_gb": 0.0,
        "available_memory_gb": 0.0,
    }

    if JAX_AVAILABLE:
        info["jax_version"] = str(jax.__version__)

    # System memory
    if PSUTIL_AVAILABLE and psutil is not None:
        try:
            memory = psutil.virtual_memory()
            info["total_memory_gb"] = memory.total / (1024**3)
            info["available_memory_gb"] = memory.available / (1024**3)
            info["memory_percent_used"] = memory.percent
        except Exception as exc:
            logger.debug("psutil memory query failed: %s", exc)

    # Backend and device info
    info["backend"] = get_backend_info()
    info["device_info"] = get_device_info()

    return info


def format_memory(bytes_value: Optional[Union[int, float]]) -> str:
    """Format memory size in human-readable format.

    Delegates to :func:`stormlog.utils.format_bytes` when available,
    otherwise provides a standalone implementation.
    """
    if bytes_value is None:
        return "N/A"

    try:
        from stormlog.utils import format_bytes

        return format_bytes(int(bytes_value))
    except (ImportError, Exception):
        pass

    value = float(bytes_value)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def validate_jax_environment() -> Dict[str, Any]:
    """Validate JAX environment for memory profiling.

    Returns a dictionary with validation results and a list of any
    issues found.
    """
    issues: List[str] = []
    validation: Dict[str, Any] = {
        "jax_available": JAX_AVAILABLE,
        "gpu_available": False,
        "tpu_available": False,
        "version_compatible": False,
        "issues": issues,
    }

    if not JAX_AVAILABLE:
        issues.append("JAX not installed")
        return validation

    # Check JAX version
    try:
        version = jax.__version__
        parts = version.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        # Require >= 0.4.20
        if major > 0 or (major == 0 and minor >= 4):
            validation["version_compatible"] = True
        else:
            issues.append(
                f"JAX {version} may not be fully compatible " "(recommend 0.4.20+)"
            )
    except Exception as exc:
        logger.debug("JAX version check failed: %s", exc)
        issues.append("Could not determine JAX version")

    # Check device availability
    try:
        backend = detect_jax_backend()
        devices = jax.local_devices()

        if backend == "gpu":
            validation["gpu_available"] = True
        elif backend == "tpu":
            validation["tpu_available"] = True
        elif backend == "cpu":
            if len(devices) > 0:
                # CPU-only is valid but note it
                issues.append(
                    "Only CPU devices found — GPU/TPU memory profiling "
                    "will fall back to psutil"
                )
            else:
                issues.append("No JAX devices found")
    except Exception as exc:
        issues.append(f"Error checking device availability: {exc}")

    return validation
