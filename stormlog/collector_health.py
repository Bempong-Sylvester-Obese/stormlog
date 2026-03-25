"""Shared collector-health state and retry helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

COLLECTOR_HEALTH_HEALTHY = "healthy"
COLLECTOR_HEALTH_DEGRADED = "degraded"
COLLECTOR_HEALTH_UNHEALTHY = "unhealthy"


def _normalize_partial_fields(fields: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(field for field in fields if field))


@dataclass(frozen=True)
class CollectorHealthState:
    """Current collector health as exposed through events and runtime stats."""

    status: str = COLLECTOR_HEALTH_HEALTHY
    telemetry_partial: bool = False
    partial_fields: tuple[str, ...] = ()
    last_error: str | None = None
    consecutive_failures: int = 0
    next_retry_epoch_s: float | None = None

    def __post_init__(self) -> None:
        if self.status not in {
            COLLECTOR_HEALTH_HEALTHY,
            COLLECTOR_HEALTH_DEGRADED,
            COLLECTOR_HEALTH_UNHEALTHY,
        }:
            raise ValueError(f"Unsupported collector health status: {self.status}")
        object.__setattr__(
            self,
            "partial_fields",
            _normalize_partial_fields(self.partial_fields),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "collector_health_status": self.status,
            "telemetry_partial": self.telemetry_partial,
            "collector_partial_fields": list(self.partial_fields),
            "collector_last_error": self.last_error,
            "collector_consecutive_failures": self.consecutive_failures,
            "collector_next_retry_epoch_s": self.next_retry_epoch_s,
        }


def collector_retry_delay_seconds(
    consecutive_failures: int,
    *,
    initial_delay_s: float,
    factor: float,
    max_delay_s: float,
) -> float:
    """Return bounded exponential backoff delay for collector retries."""
    if consecutive_failures <= 0:
        return 0.0
    if initial_delay_s <= 0:
        return 0.0
    power = max(consecutive_failures - 1, 0)
    delay = initial_delay_s * (factor**power)
    return min(max_delay_s, delay)


__all__ = [
    "COLLECTOR_HEALTH_HEALTHY",
    "COLLECTOR_HEALTH_DEGRADED",
    "COLLECTOR_HEALTH_UNHEALTHY",
    "CollectorHealthState",
    "collector_retry_delay_seconds",
]
