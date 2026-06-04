"""Typed pytest helpers for JAX tests."""

from __future__ import annotations

from typing import Any, Callable, TypeVar, cast

import pytest

F = TypeVar("F", bound=Callable[..., Any])

jax_mark: Callable[[F], F] = cast(Callable[[F], F], pytest.mark.jax)
jax_fixture: Callable[[F], F] = cast(Callable[[F], F], pytest.fixture)
