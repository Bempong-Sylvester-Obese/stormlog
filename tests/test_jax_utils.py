"""Tests for JAX utils module."""

from unittest import mock

import pytest

from stormlog.jax.utils import (
    detect_jax_backend,
    format_memory,
    get_backend_info,
    get_device_info,
    get_system_info,
    jax_is_available,
    validate_jax_environment,
)
from tests.jax_test_helpers import fake_jax_runtime  # noqa: F401

pytestmark = pytest.mark.usefixtures("fake_jax_runtime")


def test_jax_is_available() -> None:
    assert jax_is_available() is True


def test_detect_jax_backend() -> None:
    with mock.patch("stormlog.jax.utils.jax.default_backend", return_value="gpu"):
        assert detect_jax_backend() == "gpu"


def test_detect_jax_backend_exception() -> None:
    with mock.patch(
        "stormlog.jax.utils.jax.default_backend", side_effect=Exception("error")
    ):
        assert detect_jax_backend() == "cpu"


def test_detect_jax_backend_normalizes_metal() -> None:
    with mock.patch("stormlog.jax.utils.jax.default_backend", return_value="METAL"):
        assert detect_jax_backend() == "metal"


def test_get_device_info_valid() -> None:
    device_mock = mock.Mock()
    device_mock.device_kind = "gpu"
    device_mock.platform = "gpu"
    device_mock.id = 0
    device_mock.process_index = 0
    device_mock.client = "client"
    device_mock.memory_stats.return_value = {"bytes_in_use": 100}

    with mock.patch(
        "stormlog.jax.utils._cached_local_devices", return_value=[device_mock]
    ):
        info = get_device_info(0)
        assert info["kind"] == "gpu"
        assert info["memory_stats"] == {"bytes_in_use": 100}


@pytest.mark.parametrize("selector", ["0", "gpu"])
def test_get_device_info_accepts_cli_selectors(selector: str) -> None:
    device_mock = mock.Mock()
    device_mock.device_kind = "gpu"
    device_mock.platform = "gpu"
    device_mock.id = 0
    device_mock.process_index = 0
    device_mock.client = "client"
    device_mock.memory_stats.return_value = {"bytes_in_use": 100}

    with mock.patch(
        "stormlog.jax.utils.resolve_jax_device", return_value=(device_mock, 0)
    ) as mock_resolve:
        info = get_device_info(selector)

    mock_resolve.assert_called_once_with(selector)
    assert info["kind"] == "gpu"
    assert info["memory_stats_available"] is True


def test_get_device_info_out_of_range() -> None:
    with mock.patch("stormlog.jax.utils._cached_local_devices", return_value=[]):
        info = get_device_info(0)
        assert info["kind"] == "unknown"


def test_get_device_info_exception() -> None:
    with mock.patch(
        "stormlog.jax.utils._cached_local_devices", side_effect=Exception("error")
    ):
        info = get_device_info(0)
        assert info["kind"] == "unknown"


def test_get_backend_info() -> None:
    device_mock = mock.Mock()
    device_mock.id = 0
    device_mock.device_kind = "gpu"
    device_mock.platform = "gpu"
    with mock.patch(
        "stormlog.jax.utils._cached_local_devices", return_value=[device_mock]
    ):
        info = get_backend_info()
        assert info["device_count"] == 1
        assert info["devices"][0]["kind"] == "gpu"


def test_get_backend_info_exception() -> None:
    with mock.patch(
        "stormlog.jax.utils._cached_local_devices", side_effect=Exception("error")
    ):
        info = get_backend_info()
        assert info["device_count"] == 0


def test_get_system_info() -> None:
    with mock.patch("stormlog.jax.utils.jax.__version__", "0.4.20"):
        info = get_system_info()
        assert info["jax_available"] is True
        assert "total_memory_gb" in info
        assert "backend" in info


def test_format_memory() -> None:
    assert format_memory(None) == "N/A"
    assert "MB" in format_memory(1024 * 1024 * 5)

    # Test the fallback by mocking stormlog.utils.format_bytes import failure
    # We can just rely on the fact that format_memory does this internally if it fails
    # But it succeeds in the test env.


def test_validate_jax_environment_gpu() -> None:
    with (
        mock.patch("stormlog.jax.utils.jax.__version__", "0.4.20"),
        mock.patch("stormlog.jax.utils.detect_jax_backend", return_value="gpu"),
        mock.patch(
            "stormlog.jax.utils._cached_local_devices", return_value=[mock.Mock()]
        ),
    ):
        val = validate_jax_environment()
        assert val["gpu_available"] is True
        assert val["version_compatible"] is True


def test_validate_jax_environment_cpu() -> None:
    with (
        mock.patch("stormlog.jax.utils.jax.__version__", "0.4.20"),
        mock.patch("stormlog.jax.utils.detect_jax_backend", return_value="cpu"),
        mock.patch(
            "stormlog.jax.utils._cached_local_devices", return_value=[mock.Mock()]
        ),
    ):
        val = validate_jax_environment()
        assert val["gpu_available"] is False
        assert len(val["issues"]) > 0


def test_validate_jax_environment_exception() -> None:
    with (
        mock.patch("stormlog.jax.utils.jax.__version__", "invalid"),
        mock.patch(
            "stormlog.jax.utils.detect_jax_backend", side_effect=Exception("error")
        ),
    ):
        val = validate_jax_environment()
        assert val["version_compatible"] is False
        assert len(val["issues"]) >= 2


def test_validate_jax_environment_tpu() -> None:
    with (
        mock.patch("stormlog.jax.utils.jax.__version__", "0.4.20"),
        mock.patch("stormlog.jax.utils.detect_jax_backend", return_value="tpu"),
        mock.patch(
            "stormlog.jax.utils._cached_local_devices", return_value=[mock.Mock()]
        ),
    ):
        val = validate_jax_environment()
        assert val["tpu_available"] is True
        assert val["version_compatible"] is True


def test_jax_not_available() -> None:
    with mock.patch("stormlog.jax.utils.JAX_AVAILABLE", False):
        assert detect_jax_backend() == "cpu"
        assert get_device_info()["kind"] == "cpu"
        assert get_backend_info()["device_count"] == 0
        assert get_system_info()["jax_available"] is False
        assert validate_jax_environment()["jax_available"] is False
