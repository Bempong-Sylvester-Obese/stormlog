"""Tests for JAX context profiler wrapper."""

from typing import Any
from unittest import mock

import pytest

pytest.importorskip("jax")
import numpy as np

from stormlog.jax.context_profiler import (
    JAXProfiler,
    ProfiledFunction,
)
from stormlog.jax.profiler import JAXMemoryProfiler
from tests.jax_test_helpers import jax_fixture, jax_mark


@jax_fixture
def mock_device() -> mock.Mock:
    device = mock.Mock()
    device.memory_stats.return_value = {
        "bytes_in_use": 1024,
        "peak_bytes_in_use": 2048,
    }
    return device


@jax_mark
def test_profiled_function_wrapper(mock_device: mock.Mock) -> None:
    with mock.patch(
        "stormlog.jax.profiler.jax.local_devices", return_value=[mock_device]
    ):
        with mock.patch("stormlog.jax.profiler.jax.numpy.zeros") as mock_zeros:
            mock_zeros.return_value.block_until_ready.return_value = None

            def my_func(x: Any) -> Any:
                return x * 2

            profiler = JAXMemoryProfiler(device_index=0)
            wrapped = ProfiledFunction(my_func, profiler=profiler, name="test_wrap")

            res = wrapped(5)
            assert res == 10

            results = profiler.get_results()
            assert "test_wrap" in results.function_profiles
            assert results.function_profiles["test_wrap"]["calls"] == 1


@jax_mark
def test_jax_profiler_training(mock_device: mock.Mock) -> None:
    with mock.patch(
        "stormlog.jax.profiler.jax.local_devices", return_value=[mock_device]
    ):
        with mock.patch("stormlog.jax.profiler.jax.numpy.zeros") as mock_zeros:
            mock_zeros.return_value.block_until_ready.return_value = None

            jp = JAXProfiler(device_index=0)

            def train_step(batch: Any) -> None:
                pass

            dataset = [[1, 2], [3, 4]]
            jp.profile_training(train_step, dataset, epochs=1)

            res = jp.get_results()
            # Should have contexts for training, epoch_0, step_0, step_1
            assert "training" in res.function_profiles
            assert "epoch_0" in res.function_profiles
            assert "step_0" in res.function_profiles
            assert "step_1" in res.function_profiles


@jax_mark
def test_jax_profiler_inference_iterable(mock_device: mock.Mock) -> None:
    with mock.patch(
        "stormlog.jax.profiler.jax.local_devices", return_value=[mock_device]
    ):
        with mock.patch("stormlog.jax.profiler.jax.numpy.zeros") as mock_zeros:
            mock_zeros.return_value.block_until_ready.return_value = None

            jp = JAXProfiler(device_index=0)

            def infer_step(batch: Any) -> None:
                pass

            dataset = [[1], [2], [3]]
            jp.profile_inference(infer_step, dataset)

            res = jp.get_results()
            assert "inference" in res.function_profiles
            assert "inference_batch_0" in res.function_profiles
            assert "inference_batch_2" in res.function_profiles


@jax_mark
def test_jax_profiler_inference_array(mock_device: mock.Mock) -> None:
    with mock.patch(
        "stormlog.jax.profiler.jax.local_devices", return_value=[mock_device]
    ):
        with mock.patch("stormlog.jax.profiler.jax.numpy.zeros") as mock_zeros:
            mock_zeros.return_value.block_until_ready.return_value = None

            jp = JAXProfiler(device_index=0)

            def infer_step(batch: Any) -> None:
                pass

            data = np.ones((10, 5))
            jp.profile_inference(infer_step, data, batch_size=4)

            res = jp.get_results()
            assert "inference" in res.function_profiles
            assert "inference_batch_0" in res.function_profiles
            assert "inference_batch_1" in res.function_profiles
            assert "inference_batch_2" in res.function_profiles  # 3 batches (4, 4, 2)
