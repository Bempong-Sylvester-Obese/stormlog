"""Tests for JAX context profiler wrapper."""

from typing import Any
from unittest import mock

import numpy as np
import pytest

from stormlog.jax.context_profiler import (
    JAXProfiler,
    ProfiledFunction,
)
from stormlog.jax.profiler import JAXMemoryProfiler
from tests.jax_test_helpers import (  # noqa: F401
    fake_jax_runtime,
    jax_fixture,
    jax_mark,
)

pytestmark = pytest.mark.usefixtures("fake_jax_runtime")


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
def test_jax_profiler_training_streams_single_epoch_generator(
    mock_device: mock.Mock,
) -> None:
    with mock.patch(
        "stormlog.jax.profiler.jax.local_devices", return_value=[mock_device]
    ):
        with mock.patch("stormlog.jax.profiler.jax.numpy.zeros") as mock_zeros:
            mock_zeros.return_value.block_until_ready.return_value = None
            jp = JAXProfiler(device_index=0)
            events: list[str] = []

            def dataset() -> Any:
                for batch in [1, 2, 3]:
                    events.append(f"yield_{batch}")
                    yield batch

            def train_step(batch: int) -> None:
                events.append(f"step_{batch}")

            jp.profile_training(train_step, dataset(), epochs=1)

            assert events == [
                "yield_1",
                "step_1",
                "yield_2",
                "step_2",
                "yield_3",
                "step_3",
            ]


@jax_mark
def test_jax_profiler_training_replays_multi_epoch_generator(
    mock_device: mock.Mock,
) -> None:
    with mock.patch(
        "stormlog.jax.profiler.jax.local_devices", return_value=[mock_device]
    ):
        with mock.patch("stormlog.jax.profiler.jax.numpy.zeros") as mock_zeros:
            mock_zeros.return_value.block_until_ready.return_value = None
            jp = JAXProfiler(device_index=0)
            events: list[str] = []

            def dataset() -> Any:
                for batch in [1, 2]:
                    events.append(f"yield_{batch}")
                    yield batch

            def train_step(batch: int) -> None:
                events.append(f"step_{batch}")

            jp.profile_training(train_step, dataset(), epochs=2)

            assert events == [
                "yield_1",
                "yield_2",
                "step_1",
                "step_2",
                "step_1",
                "step_2",
            ]


@jax_mark
def test_jax_profiler_training_caps_generator_replay_snapshot(
    mock_device: mock.Mock,
) -> None:
    with mock.patch(
        "stormlog.jax.profiler.jax.local_devices", return_value=[mock_device]
    ):
        with mock.patch("stormlog.jax.profiler.jax.numpy.zeros") as mock_zeros:
            mock_zeros.return_value.block_until_ready.return_value = None
            jp = JAXProfiler(device_index=0)
            yielded: list[int] = []
            stepped: list[int] = []

            def dataset() -> Any:
                batch = 0
                while True:
                    yielded.append(batch)
                    yield batch
                    batch += 1

            def train_step(batch: int) -> None:
                stepped.append(batch)

            jp.profile_training(train_step, dataset(), epochs=2, steps_per_epoch=2)

            assert yielded == [0, 1]
            assert stepped == [0, 1, 0, 1]


@jax_mark
def test_jax_profiler_training_raises_for_empty_later_epoch(
    mock_device: mock.Mock,
) -> None:
    class DrainingIterable:
        def __init__(self) -> None:
            self._batches = iter([1, 2])

        def __iter__(self) -> Any:
            return self._batches

    with mock.patch(
        "stormlog.jax.profiler.jax.local_devices", return_value=[mock_device]
    ):
        with mock.patch("stormlog.jax.profiler.jax.numpy.zeros") as mock_zeros:
            mock_zeros.return_value.block_until_ready.return_value = None
            jp = JAXProfiler(device_index=0)
            stepped: list[int] = []

            def train_step(batch: int) -> None:
                stepped.append(batch)

            with pytest.raises(ValueError, match="epoch 0"):
                jp.profile_training(train_step, DrainingIterable(), epochs=2)

            assert stepped == [1, 2]


@jax_mark
def test_jax_profiler_training_iterates_callable_iterable_dataset(
    mock_device: mock.Mock,
) -> None:
    class CallableIterable:
        def __iter__(self) -> Any:
            return iter([1, 2])

        def __call__(self, value: int) -> list[int]:
            raise AssertionError("profile_training should iterate this object directly")

    with mock.patch(
        "stormlog.jax.profiler.jax.local_devices", return_value=[mock_device]
    ):
        with mock.patch("stormlog.jax.profiler.jax.numpy.zeros") as mock_zeros:
            mock_zeros.return_value.block_until_ready.return_value = None
            jp = JAXProfiler(device_index=0)
            stepped: list[int] = []

            def train_step(batch: int) -> None:
                stepped.append(batch)

            jp.profile_training(train_step, CallableIterable(), epochs=1)

            assert stepped == [1, 2]


@jax_mark
def test_jax_profiler_training_uses_dataset_factory_per_epoch(
    mock_device: mock.Mock,
) -> None:
    with mock.patch(
        "stormlog.jax.profiler.jax.local_devices", return_value=[mock_device]
    ):
        with mock.patch("stormlog.jax.profiler.jax.numpy.zeros") as mock_zeros:
            mock_zeros.return_value.block_until_ready.return_value = None
            jp = JAXProfiler(device_index=0)
            factory_calls: list[int] = []
            stepped: list[int] = []

            def dataset_factory() -> list[int]:
                factory_calls.append(len(factory_calls))
                return [1, 2]

            def train_step(batch: int) -> None:
                stepped.append(batch)

            jp.profile_training(train_step, dataset_factory, epochs=3)

            assert factory_calls == [0, 1, 2]
            assert stepped == [1, 2, 1, 2, 1, 2]


@jax_mark
def test_jax_profiler_training_rejects_non_iterable_factory_result(
    mock_device: mock.Mock,
) -> None:
    with mock.patch(
        "stormlog.jax.profiler.jax.local_devices", return_value=[mock_device]
    ):
        with mock.patch("stormlog.jax.profiler.jax.numpy.zeros") as mock_zeros:
            mock_zeros.return_value.block_until_ready.return_value = None
            jp = JAXProfiler(device_index=0)

            def train_step(batch: int) -> None:
                pass

            def dataset_factory() -> int:
                return 1

            with pytest.raises(TypeError, match="dataset factory result"):
                jp.profile_training(train_step, dataset_factory, epochs=1)


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
