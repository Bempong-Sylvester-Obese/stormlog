"""Tests for JAX __init__.py module."""

from unittest import mock

import pytest

import stormlog.jax as jax_pkg


def test_dir() -> None:
    assert "MemoryAnalyzer" in dir(jax_pkg)
    assert "JAXMemoryProfiler" in dir(jax_pkg)
    assert "__version__" in dir(jax_pkg)


def test_getattr_valid() -> None:
    # Calling getattr on jax_pkg for an unimported symbol triggers _resolve_symbol
    # It should succeed and return the symbol
    sym = getattr(jax_pkg, "get_device_info")
    assert callable(sym)


def test_getattr_invalid() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(jax_pkg, "DoesNotExist")


def test_is_jax_missing() -> None:
    # We can test the helper directly
    from stormlog.jax import _is_jax_missing

    class DummyExc(Exception):
        pass

    exc = ModuleNotFoundError("No module named 'jax'", name="jax")
    assert _is_jax_missing(exc) is True

    exc2 = ModuleNotFoundError("No module named 'numpy'", name="numpy")
    assert _is_jax_missing(exc2) is False

    exc3 = DummyExc("test")
    exc3.__cause__ = exc
    assert _is_jax_missing(exc3) is True


def test_resolve_symbol_jax_missing() -> None:
    from stormlog.jax import _resolve_symbol

    with mock.patch(
        "importlib.import_module",
        side_effect=ModuleNotFoundError("No module named 'jax'", name="jax"),
    ):
        with pytest.raises(ImportError, match="JAX is required for this feature"):
            # Need to pick a symbol we haven't resolved yet in the same Python process
            # or mock _SYMBOL_TO_MODULE
            _resolve_symbol("MemoryWatchdog")
