"""JAX support for Stormlog."""

from __future__ import annotations

import importlib
from typing import Any

from stormlog import __version__

__author__ = "Stormlog Team"
__email__ = "prince.agyei.tuffour@gmail.com"

_JAX_INSTALL_GUIDANCE = (
    "JAX is required for this feature. Install with " "`pip install 'stormlog[jax]'`."
)

_SYMBOL_TO_MODULE = {
    "JAXMemoryProfiler": (".profiler", "JAXMemoryProfiler"),
    "JAXMemoryTracker": (".tracker", "JAXMemoryTracker"),
    "get_device_info": (".utils", "get_device_info"),
    "get_system_info": (".utils", "get_system_info"),
}


def _is_jax_missing(exc: BaseException) -> bool:
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, ModuleNotFoundError) and current.name == "jax":
            return True
        next_exc = current.__cause__
        if next_exc is None and not current.__suppress_context__:
            next_exc = current.__context__
        current = next_exc
    return False


def _resolve_symbol(name: str) -> Any:
    module_name, symbol_name = _SYMBOL_TO_MODULE[name]
    try:
        module = importlib.import_module(module_name, __name__)
    except Exception as exc:
        if _is_jax_missing(exc):
            raise ImportError(_JAX_INSTALL_GUIDANCE) from exc
        raise

    value = getattr(module, symbol_name)
    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    if name in _SYMBOL_TO_MODULE:
        return _resolve_symbol(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)


__all__ = [
    "__version__",
    "JAXMemoryProfiler",
    "JAXMemoryTracker",
    "get_device_info",
    "get_system_info",
]
