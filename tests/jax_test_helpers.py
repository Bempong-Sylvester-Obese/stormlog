"""Typed pytest helpers for JAX tests."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, Callable, TypeVar, cast

import pytest

F = TypeVar("F", bound=Callable[..., Any])

jax_mark: Callable[[F], F] = cast(Callable[[F], F], pytest.mark.jax)
jax_fixture: Callable[[F], F] = cast(Callable[[F], F], pytest.fixture)


class _ReadyValue:
    def block_until_ready(self) -> "_ReadyValue":
        return self


@jax_fixture
def fake_jax_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject the small JAX surface needed by mocked unit tests."""
    fake_numpy = ModuleType("jax.numpy")
    fake_numpy.zeros = lambda *args, **kwargs: _ReadyValue()  # type: ignore[attr-defined]
    fake_numpy.asarray = lambda value: value  # type: ignore[attr-defined]

    fake_profiler = ModuleType("jax.profiler")
    fake_profiler.save_device_memory_profile = lambda path: None  # type: ignore[attr-defined]

    fake_jax = ModuleType("jax")
    fake_jax.__dict__["__version__"] = "0.4.20"
    fake_jax.numpy = fake_numpy  # type: ignore[attr-defined]
    fake_jax.profiler = fake_profiler  # type: ignore[attr-defined]
    fake_jax.local_devices = lambda: []  # type: ignore[attr-defined]
    fake_jax.default_backend = lambda: "cpu"  # type: ignore[attr-defined]
    fake_jax.clear_caches = lambda: None  # type: ignore[attr-defined]
    fake_jax.live_arrays = lambda: []  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "jax.profiler", fake_profiler)

    for module_name in (
        "stormlog.jax.profiler",
        "stormlog.jax.tracker",
        "stormlog.jax.utils",
        "stormlog.jax.context_profiler",
        "stormlog.jax.visualizer",
        "stormlog.jax.cli",
    ):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        if hasattr(module, "jax"):
            monkeypatch.setattr(module, "jax", fake_jax)
        if hasattr(module, "_jax"):
            monkeypatch.setattr(module, "_jax", fake_jax)
        if hasattr(module, "JAX_AVAILABLE"):
            monkeypatch.setattr(module, "JAX_AVAILABLE", True)
