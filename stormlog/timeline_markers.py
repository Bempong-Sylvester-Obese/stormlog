"""Derived timeline marker helpers for telemetry sessions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from .phases import PhaseReplayIndex, format_phase_path
from .telemetry import LoadedTelemetrySession

MARKER_KIND_ALERT = "alert"
MARKER_KIND_COLLECTOR = "collector"
MARKER_KIND_LIFECYCLE = "lifecycle"
MARKER_KIND_OOM = "oom"
MARKER_KIND_PHASE = "phase"

MARKER_SEVERITY_INFO = "info"
MARKER_SEVERITY_WARNING = "warning"
MARKER_SEVERITY_CRITICAL = "critical"

MARKER_SOURCE_PHASE_REPLAY = "phase_replay"
MARKER_SOURCE_TELEMETRY_EVENT = "telemetry_event"

_POINT_EVENT_KINDS = {
    "start": (MARKER_KIND_LIFECYCLE, MARKER_SEVERITY_INFO, "Tracking started"),
    "stop": (MARKER_KIND_LIFECYCLE, MARKER_SEVERITY_INFO, "Tracking stopped"),
    "collector_degraded": (
        MARKER_KIND_COLLECTOR,
        MARKER_SEVERITY_WARNING,
        "Collector degraded",
    ),
    "collector_recovered": (
        MARKER_KIND_COLLECTOR,
        MARKER_SEVERITY_INFO,
        "Collector recovered",
    ),
    "warning": (MARKER_KIND_ALERT, MARKER_SEVERITY_WARNING, "Warning"),
    "critical": (MARKER_KIND_ALERT, MARKER_SEVERITY_CRITICAL, "Critical alert"),
    "error": (MARKER_KIND_ALERT, MARKER_SEVERITY_CRITICAL, "Error"),
}


@dataclass(frozen=True)
class TimelineMarker:
    """Normalized timeline landmark derived from telemetry or annotation sources."""

    session_id: str
    start_ns: int
    end_ns: int | None
    kind: str
    source: str
    severity: str
    label: str
    rank: int | None = None
    local_rank: int | None = None
    world_size: int | None = None
    event_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_interval(self) -> bool:
        """Return whether the marker spans a non-point interval."""
        return self.end_ns is not None and self.end_ns != self.start_ns


def derive_session_timeline_markers(
    session: LoadedTelemetrySession,
    *,
    include_phase_markers: bool = True,
) -> list[TimelineMarker]:
    """Derive normalized markers from one loaded telemetry session."""
    return derive_timeline_markers(
        session.events,
        include_phase_markers=include_phase_markers,
    )


def derive_timeline_markers(
    events: Sequence[Any],
    *,
    include_phase_markers: bool = True,
) -> list[TimelineMarker]:
    """Derive normalized timeline markers from telemetry events."""
    markers = [
        marker
        for event in events
        if (marker := _point_marker_from_event(event)) is not None
    ]
    if include_phase_markers:
        markers.extend(_phase_markers_from_events(events))
    return sorted(markers, key=_marker_sort_key)


def timeline_marker_to_dict(marker: TimelineMarker) -> dict[str, Any]:
    """Serialize a marker into a JSON-safe mapping."""
    return asdict(marker)


def _point_marker_from_event(event: Any) -> TimelineMarker | None:
    event_type = _event_field(event, "event_type")
    if not isinstance(event_type, str):
        return None
    event_type = event_type.strip()
    if event_type not in _POINT_EVENT_KINDS:
        return None

    session_id = _event_field(event, "session_id")
    timestamp_ns = _event_field(event, "timestamp_ns")
    if not isinstance(session_id, str) or not isinstance(timestamp_ns, int):
        return None

    metadata = _event_metadata(event)
    kind, severity, fallback_label = _POINT_EVENT_KINDS[event_type]
    if event_type == "error" and _has_oom_metadata(metadata):
        kind = MARKER_KIND_OOM
        fallback_label = _oom_label(metadata)

    context = _event_field(event, "context")
    label = (
        context.strip()
        if isinstance(context, str) and context.strip()
        else fallback_label
    )

    return TimelineMarker(
        session_id=session_id,
        start_ns=timestamp_ns,
        end_ns=None,
        kind=kind,
        source=MARKER_SOURCE_TELEMETRY_EVENT,
        severity=severity,
        label=label,
        rank=_optional_int_field(event, "rank"),
        local_rank=_optional_int_field(event, "local_rank"),
        world_size=_optional_int_field(event, "world_size"),
        event_type=event_type,
        metadata=metadata,
    )


def _phase_markers_from_events(events: Sequence[Any]) -> list[TimelineMarker]:
    spans = _all_phase_spans(events)
    rank_identity = _rank_identity_by_session(events)
    markers: list[TimelineMarker] = []
    for span in spans:
        identity = rank_identity.get((span.session_id, span.rank), {})
        markers.append(
            TimelineMarker(
                session_id=span.session_id,
                start_ns=span.start_ns,
                end_ns=span.end_ns,
                kind=MARKER_KIND_PHASE,
                source=MARKER_SOURCE_PHASE_REPLAY,
                severity=MARKER_SEVERITY_INFO,
                label=f"Phase: {format_phase_path(span.path)}",
                rank=span.rank,
                local_rank=identity.get("local_rank"),
                world_size=identity.get("world_size"),
                event_type="phase",
                metadata={
                    "phase_path": list(span.path),
                    "scope_id": span.scope_id,
                    "thread_id": span.thread_id,
                    "thread_name": span.thread_name,
                    "sequence": span.sequence,
                    "synthetic_end": span.synthetic_end,
                },
            )
        )
    return markers


def _all_phase_spans(events: Sequence[Any]) -> list[Any]:
    index = PhaseReplayIndex.from_events(events)
    session_ids = {
        session_id
        for event in events
        if isinstance((session_id := _event_field(event, "session_id")), str)
    }
    spans: list[Any] = []
    for session_id in session_ids:
        spans.extend(index.spans_for(session_id=session_id))
    return spans


def _rank_identity_by_session(
    events: Sequence[Any],
) -> dict[tuple[str, int], dict[str, int | None]]:
    identities: dict[tuple[str, int], dict[str, int | None]] = {}
    for event in events:
        session_id = _event_field(event, "session_id")
        rank = _optional_int_field(event, "rank")
        if not isinstance(session_id, str) or rank is None:
            continue
        identities.setdefault(
            (session_id, rank),
            {
                "local_rank": _optional_int_field(event, "local_rank"),
                "world_size": _optional_int_field(event, "world_size"),
            },
        )
    return identities


def _event_field(event: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(event, Mapping):
        return event.get(field_name, default)
    return getattr(event, field_name, default)


def _event_metadata(event: Any) -> dict[str, Any]:
    metadata = _event_field(event, "metadata", {})
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _optional_int_field(event: Any, field_name: str) -> int | None:
    value = _event_field(event, field_name)
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def _has_oom_metadata(metadata: Mapping[str, Any]) -> bool:
    return any(key in metadata for key in ("oom_reason", "oom_dump_path"))


def _oom_label(metadata: Mapping[str, Any]) -> str:
    reason = metadata.get("oom_reason")
    if isinstance(reason, str) and reason.strip():
        return f"OOM detected: {reason.strip()}"
    return "OOM detected"


def _marker_sort_key(marker: TimelineMarker) -> tuple[int, int, int, str, str]:
    rank = marker.rank if marker.rank is not None else -1
    end_ns = marker.end_ns if marker.end_ns is not None else marker.start_ns
    return (marker.start_ns, end_ns, rank, marker.kind, marker.label)


__all__ = [
    "MARKER_KIND_ALERT",
    "MARKER_KIND_COLLECTOR",
    "MARKER_KIND_LIFECYCLE",
    "MARKER_KIND_OOM",
    "MARKER_KIND_PHASE",
    "MARKER_SEVERITY_CRITICAL",
    "MARKER_SEVERITY_INFO",
    "MARKER_SEVERITY_WARNING",
    "MARKER_SOURCE_PHASE_REPLAY",
    "MARKER_SOURCE_TELEMETRY_EVENT",
    "TimelineMarker",
    "derive_session_timeline_markers",
    "derive_timeline_markers",
    "timeline_marker_to_dict",
]
