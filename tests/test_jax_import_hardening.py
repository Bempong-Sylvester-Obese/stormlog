"""Tests for JAX import hardening and lazy loading behavior."""

import subprocess
import sys
import textwrap

import pytest


def test_jax_imports_are_hardened_when_jax_is_missing() -> None:
    code = textwrap.dedent(
        """
        import builtins

        original_import = builtins.__import__

        def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "jax" or name.startswith("jax."):
                raise ModuleNotFoundError("No module named 'jax'", name="jax")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = blocked_import

        import stormlog.jax

        # Test that lazy loading doesn't crash on import
        assert stormlog.jax.__name__ == "stormlog.jax"

        memory_tracker = stormlog.jax.MemoryTracker
        memory_profiler = stormlog.jax.JAXMemoryProfiler
        assert callable(memory_tracker)
        assert callable(memory_profiler)

        for runtime_class in (memory_tracker, memory_profiler):
            try:
                runtime_class()
            except ImportError as exc:
                assert "stormlog[jax]" in str(exc)
            else:
                raise AssertionError(
                    f"Expected {runtime_class.__name__} construction to fail without jax"
                )

        print("ok")
        """
    )

    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"JAX import hardening subprocess timed out: {exc}")

    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout
