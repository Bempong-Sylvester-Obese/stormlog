"""Smoke tests for Stormlog's real JAX runtime path."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.jax


def test_stormlog_jax_real_runtime_smoke() -> None:
    jax = pytest.importorskip("jax")

    import stormlog.jax as stormlog_jax

    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
        pytest.skip("JAX CPU device is unavailable")

    profiler = stormlog_jax.JAXMemoryProfiler()

    with profiler.profile_context("real_jax_cpu_op"):
        result = jax.device_put(np.arange(3), cpu_devices[0]) + 1
        result.block_until_ready()

    results = profiler.get_results()

    assert result.tolist() == [1, 2, 3]
    assert results.function_profiles["real_jax_cpu_op"]["calls"] == 1
    if results.device_memory_available:
        assert results.peak_memory_bytes >= 0
    else:
        assert results.peak_memory_bytes == 0
        assert results.device_memory_unavailable_reason
