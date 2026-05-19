"""Tests for JAX import hardening and lazy loading behavior."""

import subprocess
import sys
import textwrap


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

        try:
            _ = stormlog.jax.JAXMemoryTracker
        except ImportError as exc:
            assert "stormlog[jax]" in str(exc)
        else:
            raise AssertionError("Expected JAXMemoryTracker symbol load to fail lazily without jax")

        try:
            _ = stormlog.jax.JAXMemoryProfiler
        except ImportError as exc:
            assert "stormlog[jax]" in str(exc)
        else:
            raise AssertionError("Expected JAXMemoryProfiler symbol load to fail lazily without jax")

        print("ok")
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout
