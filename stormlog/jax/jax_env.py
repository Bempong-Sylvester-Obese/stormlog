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
    * ``XLA_FLAGS`` → appends ``--xla_log_level=2`` (WARNING+)
    * ``TF_CPP_MIN_LOG_LEVEL`` → ``"2"`` (suppress TF C++ backend noise
      when JAX falls back to the TF XLA bridge)
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    os.environ.setdefault("JAX_LOG_COMPILES", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    # Append XLA log-level flag without clobbering user-defined flags.
    xla_flag = "--xla_log_level=2"
    existing = os.environ.get("XLA_FLAGS", "")
    if xla_flag not in existing:
        os.environ["XLA_FLAGS"] = f"{existing} {xla_flag}".strip()

    _CONFIGURED = True
