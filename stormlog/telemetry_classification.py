"""Shared classification helpers for canonical telemetry events."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from .issues import normalize_text_dimension

ALERT_EVENT_TYPES = frozenset({"warning", "critical", "error"})
COLLECTOR_TRANSITION_TYPES = frozenset({"collector_degraded", "collector_recovered"})
COLLECTOR_DEGRADED_STATUSES = frozenset({"degraded", "unhealthy"})


class TelemetryClassifiable(Protocol):
    """Telemetry fields required by shared classification helpers."""

    @property
    def event_type(self) -> str:
        """Return the telemetry event type."""

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return telemetry metadata."""


def is_alert_event(event: TelemetryClassifiable) -> bool:
    """Return whether a telemetry event should be counted as an alert."""

    if event.event_type in ALERT_EVENT_TYPES:
        return True
    severity = normalize_text_dimension(event.metadata.get("severity"))
    return severity in {"warning", "critical", "error"}


def event_severity(event: TelemetryClassifiable) -> str:
    """Return a normalized severity for alert and marker grouping."""

    metadata_severity = event.metadata.get("severity")
    if isinstance(metadata_severity, str) and metadata_severity.strip():
        return normalize_text_dimension(metadata_severity)
    if event.event_type in {"critical", "error"}:
        return "critical"
    if event.event_type == "warning":
        return "warning"
    return "info"


def is_oom_event(event: TelemetryClassifiable) -> bool:
    """Return whether an error telemetry event carries OOM marker metadata."""

    return event.event_type == "error" and any(
        key in event.metadata for key in ("oom_reason", "oom_dump_path")
    )


def is_collector_degradation_event(event: TelemetryClassifiable) -> bool:
    """Return whether an event indicates collector degradation."""

    health_status = normalize_text_dimension(
        event.metadata.get("collector_health_status")
    )
    return (
        event.event_type == "collector_degraded"
        or health_status in COLLECTOR_DEGRADED_STATUSES
    )


def event_backend(event: TelemetryClassifiable) -> str:
    """Return a normalized backend dimension for grouping."""

    return normalize_text_dimension(event.metadata.get("backend"))


__all__ = [
    "ALERT_EVENT_TYPES",
    "COLLECTOR_DEGRADED_STATUSES",
    "COLLECTOR_TRANSITION_TYPES",
    "event_backend",
    "event_severity",
    "is_alert_event",
    "is_collector_degradation_event",
    "is_oom_event",
]
