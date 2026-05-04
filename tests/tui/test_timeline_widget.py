from __future__ import annotations

import pytest

pytest.importorskip("textual")

from stormlog.timeline_markers import TimelineMarker
from stormlog.tui.widgets.timeline import DistributedTimelineCanvas


def _marker(
    *,
    start_ns: int,
    severity: str,
    label: str,
    kind: str = "alert",
    end_ns: int | None = None,
) -> TimelineMarker:
    return TimelineMarker(
        session_id="session-1",
        start_ns=start_ns,
        end_ns=end_ns,
        kind=kind,
        source="telemetry_event",
        severity=severity,
        label=label,
        metadata={},
    )


def test_marker_summary_prioritizes_severity_then_recency() -> None:
    canvas = DistributedTimelineCanvas()
    markers = [
        _marker(start_ns=100, severity="info", label="Tracking started"),
        _marker(
            start_ns=200,
            end_ns=300,
            severity="info",
            label="Phase: warmup",
            kind="phase",
        ),
        _marker(start_ns=400, severity="warning", label="Fragmentation warning"),
        _marker(start_ns=500, severity="info", label="Checkpoint saved"),
        _marker(start_ns=600, severity="critical", label="OOM detected"),
    ]

    summary = canvas._format_marker_summary(markers)

    assert summary == (
        "! OOM detected | ~ Fragmentation warning | i Checkpoint saved | +2 more"
    )


def test_marker_summary_keeps_recent_marker_within_severity() -> None:
    canvas = DistributedTimelineCanvas()
    markers = [
        _marker(start_ns=100, severity="warning", label="Older warning"),
        _marker(start_ns=300, severity="warning", label="Newer warning"),
        _marker(start_ns=200, severity="critical", label="Critical event"),
    ]

    summary = canvas._format_marker_summary(markers)

    assert summary == "! Critical event | ~ Newer warning | ~ Older warning"
