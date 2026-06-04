"""JAX environment configuration for Stormlog.

Suppresses verbose JAX/XLA logging and configures the JAX runtime
environment before any ``import jax`` call.  Every module in the
``stormlog.jax`` package should call :func:`configure_jax_logging`
at import time, **before** importing ``jax`` itself.
"""

from __future__ import annotations

import os

_CONFIGURED = False


def configure_jax_logging() -> None:
    """Suppress verbose JAX/XLA info-level logging.

    Idempotent — safe to call multiple times.  Sets environment
    variables that JAX and XLA inspect on first import:

    * ``JAX_LOG_COMPILES`` → ``"0"`` (suppress JIT compilation logs)
    * ``TF_CPP_MIN_LOG_LEVEL`` → ``"2"`` (suppress TF C++ backend noise
      when JAX falls back to the TF XLA bridge)
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    os.environ.setdefault("JAX_LOG_COMPILES", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    _CONFIGURED = True
