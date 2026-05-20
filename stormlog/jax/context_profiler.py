"""JAX Context Profiling.

Provides module-level convenience functions, a high-level ``JAXProfiler``
class, and a ``ProfiledFunction`` wrapper analogous to the
``ProfiledModule``/``ProfiledLayer`` classes in the PyTorch and TensorFlow
context profilers.

Because JAX follows a functional paradigm (no ``nn.Module`` hierarchy),
``ProfiledFunction`` wraps arbitrary callables rather than layer objects.
"""

from __future__ import annotations

import functools
import logging
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar, Union, cast

from .jax_env import configure_jax_logging
from .profiler import (
    MemoryProfiler,
    ProfileResult,
)
from .profiler import clear_global_profiler as _clear_profiler
from .profiler import clear_profiles as _clear_profiles
from .profiler import get_global_profiler as _get_profiler
from .profiler import get_profile_summaries as _get_summaries
from .profiler import set_global_profiler as _set_profiler

try:
    import jax  # noqa: F401

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None

configure_jax_logging()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level global profiler state
# ---------------------------------------------------------------------------

_global_profiler: Optional[MemoryProfiler] = None
_profiler_lock = threading.Lock()
F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def get_global_profiler() -> MemoryProfiler:
    """Get or create the global :class:`MemoryProfiler` instance.

    Delegates to the singleton managed in :mod:`stormlog.jax.profiler`.

    Returns:
        The global :class:`MemoryProfiler`.
    """
    return _get_profiler()


def set_global_profiler(profiler: MemoryProfiler) -> None:
    """Replace the global :class:`MemoryProfiler` instance.

    Args:
        profiler: New profiler instance to install as the global singleton.
    """
    _set_profiler(profiler)


def clear_global_profiler() -> None:
    """Reset and discard the global profiler.

    After this call, :func:`get_global_profiler` will create a fresh
    instance on the next invocation.
    """
    _clear_profiler()


def clear_profiles() -> None:
    """Reset profiling data without discarding the global profiler."""
    _clear_profiles()


def get_profile_summaries(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return aggregated profiling summaries from the global profiler.

    Args:
        limit: Maximum number of summaries to return.

    Returns:
        A list of dicts, one per profiled function/context block,
        sorted by peak memory (descending).
    """
    return _get_summaries(limit)


# ---------------------------------------------------------------------------
# Decorator & context-manager helpers
# ---------------------------------------------------------------------------


def profile_function(
    func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    profiler: Optional[MemoryProfiler] = None,
) -> Union[Callable[[F], F], F]:
    """Decorator to profile a function's JAX device-memory usage.

    Can be used bare (``@profile_function``) or with keyword arguments
    (``@profile_function(name="custom")``).

    Args:
        func: Function to profile (when used as ``@profile_function``).
        name: Custom name for the profiled function.  Defaults to
            ``func.__name__``.
        profiler: Explicit :class:`MemoryProfiler` to use.  Falls back to
            the global profiler when *None*.

    Returns:
        Decorated callable (or decorator factory when called with keyword
        arguments).
    """

    def decorator(f: F) -> F:
        profiled_name = name or getattr(f, "__name__", "unknown_function")

        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            prof = profiler or get_global_profiler()
            with prof.profile_context(profiled_name or "profiled_func"):
                return f(*args, **kwargs)

        # Attach profiling metadata for introspection.
        setattr(wrapper, "_is_profiled", True)
        setattr(wrapper, "_profile_name", profiled_name)

        return cast(F, wrapper)

    if func is None:
        # Called as ``@profile_function(name=...)``
        return decorator
    # Called as ``@profile_function``
    return decorator(func)


@contextmanager
def profile_context(
    name: str = "context",
    profiler: Optional[MemoryProfiler] = None,
) -> Iterator[MemoryProfiler]:
    """Context manager for profiling a block of code.

    Args:
        name: Label for the profiled block.
        profiler: Explicit profiler.  Falls back to the global profiler.

    Yields:
        The :class:`MemoryProfiler` being used.

    Example::

        with profile_context("matmul") as prof:
            result = jax.numpy.dot(a, b)
    """
    prof = profiler or get_global_profiler()
    with prof.profile_context(name):
        yield prof


# ---------------------------------------------------------------------------
# ProfiledFunction – wraps arbitrary callables
# ---------------------------------------------------------------------------


class ProfiledFunction:
    """Wrapper that automatically profiles every call to a function.

    JAX does not use an ``nn.Module`` class hierarchy, so instead of
    wrapping a layer/module this class wraps any callable (pure functions,
    closures, ``jax.jit``-compiled functions, etc.).

    Args:
        func: The callable to profile.
        profiler: Explicit :class:`MemoryProfiler`.  Falls back to the
            global profiler when *None*.
        name: Label used in profiling output.  Defaults to the callable's
            ``__name__`` or ``__class__.__name__``.

    Example::

        profiled_forward = ProfiledFunction(forward_fn, name="forward")
        output = profiled_forward(params, batch)
    """

    def __init__(
        self,
        func: Callable[..., Any],
        profiler: Optional[MemoryProfiler] = None,
        name: Optional[str] = None,
    ) -> None:
        self.func = func
        self.profiler = profiler or get_global_profiler()
        self.name = name or getattr(func, "__name__", func.__class__.__name__)

        # Preserve introspection attributes from the wrapped callable.
        functools.update_wrapper(self, func)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the wrapped callable under memory profiling."""
        with self.profiler.profile_context(self.name or "profiled_func"):
            return self.func(*args, **kwargs)

    def __repr__(self) -> str:
        return f"ProfiledFunction({self.name!r})"


# ---------------------------------------------------------------------------
# JAXProfiler – high-level profiling interface
# ---------------------------------------------------------------------------


class JAXProfiler:
    """High-level JAX profiling interface.

    Provides convenience methods for profiling training loops and inference
    passes, analogous to :class:`stormlog.context_profiler.MemoryProfiler`
    (PyTorch) and :class:`stormlog.tensorflow.context_profiler.TensorFlowProfiler`.

    Args:
        device_index: Index of the JAX device to monitor (default ``0``).

    Example::

        jp = JAXProfiler()
        jp.profile_training(train_step, dataset, epochs=3)
        result = jp.get_results()
    """

    def __init__(self, device_index: int = 0) -> None:
        self.profiler = MemoryProfiler(device_index=device_index)
        set_global_profiler(self.profiler)

    # -- Training profiling ------------------------------------------------

    def profile_training(
        self,
        train_step_fn: Callable[..., Any],
        dataset: Any,
        epochs: int = 1,
        steps_per_epoch: Optional[int] = None,
    ) -> None:
        """Profile a JAX training loop.

        The caller supplies a *train_step_fn* that is invoked once per
        batch.  ``train_step_fn`` should accept a single batch as its
        first positional argument (additional arguments can be closed over
        or passed through the function itself).

        Args:
            train_step_fn: A callable ``(batch) -> Any`` that executes a
                single training step.
            dataset: An iterable of batches.  Each epoch iterates over the
                full dataset (or up to *steps_per_epoch* batches).
            epochs: Number of epochs to profile.
            steps_per_epoch: Optional cap on the number of steps per epoch.
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for JAXProfiler.profile_training")

        with self.profiler.profile_context("training"):
            for epoch in range(epochs):
                with self.profiler.profile_context(f"epoch_{epoch}"):
                    step_count = 0

                    for batch in dataset:
                        if (
                            steps_per_epoch is not None
                            and step_count >= steps_per_epoch
                        ):
                            break

                        with self.profiler.profile_context(f"step_{step_count}"):
                            train_step_fn(batch)

                        step_count += 1

    # -- Inference profiling -----------------------------------------------

    def profile_inference(
        self,
        inference_fn: Callable[..., Any],
        data: Any,
        batch_size: int = 32,
    ) -> None:
        """Profile a JAX inference pass.

        If *data* is an iterable of batches it is consumed directly;
        otherwise it is treated as a single array-like and sliced into
        batches of *batch_size*.

        Args:
            inference_fn: A callable ``(batch) -> Any`` that runs
                inference on a single batch.
            data: Input data – either an iterable of batches or a single
                array-like with a leading batch dimension.
            batch_size: Batch size used when *data* must be sliced.
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for JAXProfiler.profile_inference")

        with self.profiler.profile_context("inference"):
            # Try iterating first (e.g. tf.data.Dataset, list of batches).
            if hasattr(data, "__iter__") and not hasattr(data, "shape"):
                for i, batch in enumerate(data):
                    with self.profiler.profile_context(f"inference_batch_{i}"):
                        inference_fn(batch)
                return

            # Fall back to manual batching over an array-like.
            import jax.numpy as jnp

            if not hasattr(data, "shape"):
                data = jnp.asarray(data)

            num_samples = data.shape[0]
            num_batches = (num_samples + batch_size - 1) // batch_size

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                batch = data[start_idx:end_idx]

                with self.profiler.profile_context(f"inference_batch_{i}"):
                    inference_fn(batch)

    # -- Results / lifecycle -----------------------------------------------

    def get_results(self) -> ProfileResult:
        """Return the aggregated :class:`ProfileResult`."""
        return self.profiler.get_results()

    def reset(self) -> None:
        """Clear all captured profiling data."""
        self.profiler.reset()
