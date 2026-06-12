from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from stormlog.telemetry_classification import (
    event_backend,
    event_severity,
    is_alert_event,
    is_collector_degradation_event,
    is_oom_event,
)


@dataclass(frozen=True)
class _Event:
    event_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


def test_alert_classification_normalizes_metadata_severity() -> None:
    event = _Event(event_type="sample", metadata={"severity": " Warning "})

    assert is_alert_event(event) is True
    assert event_severity(event) == "warning"


def test_oom_and_collector_classification_use_shared_metadata_rules() -> None:
    oom_event = _Event(
        event_type="error",
        metadata={"oom_reason": "message_pattern:out of memory"},
    )
    collector_event = _Event(
        event_type="sample",
        metadata={"collector_health_status": " Unhealthy ", "backend": " CUDA "},
    )

    assert is_oom_event(oom_event) is True
    assert is_collector_degradation_event(collector_event) is True
    assert event_backend(collector_event) == "cuda"
