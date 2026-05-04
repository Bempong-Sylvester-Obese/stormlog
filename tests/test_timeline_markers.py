from __future__ import annotations

from typing import Any

from stormlog.telemetry import SCHEMA_VERSION_V3, telemetry_event_from_record
from stormlog.timeline_markers import (
    MARKER_KIND_ALERT,
    MARKER_KIND_COLLECTOR,
    MARKER_KIND_LIFECYCLE,
    MARKER_KIND_OOM,
    MARKER_KIND_PHASE,
    MARKER_SEVERITY_CRITICAL,
    MARKER_SEVERITY_INFO,
    MARKER_SEVERITY_WARNING,
    MARKER_SOURCE_PHASE_REPLAY,
    MARKER_SOURCE_TELEMETRY_EVENT,
    derive_timeline_markers,
    timeline_marker_to_dict,
)


def _event(
    *,
    event_type: str,
    timestamp_ns: int,
    session_id: str = "session-1",
    context: str | None = None,
    metadata: dict[str, Any] | None = None,
    rank: int = 0,
    local_rank: int = 0,
    world_size: int = 1,
) -> Any:
    return telemetry_event_from_record(
        {
            "schema_version": SCHEMA_VERSION_V3,
            "session_id": session_id,
            "timestamp_ns": timestamp_ns,
            "event_type": event_type,
            "collector": "stormlog.cuda_tracker",
            "sampling_interval_ms": 100,
            "pid": 1234,
            "host": "host-a",
            "job_id": "job-1",
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "device_id": 0,
            "allocator_allocated_bytes": 1024,
            "allocator_reserved_bytes": 2048,
            "allocator_active_bytes": 512,
            "allocator_inactive_bytes": 1536,
            "allocator_change_bytes": 0,
            "device_used_bytes": 2048,
            "device_free_bytes": 4096,
            "device_total_bytes": 6144,
            "context": context,
            "metadata": metadata or {},
        }
    )


def _phase_scope(
    *,
    action: str,
    scope_id: str,
    sequence: int,
    path: list[str],
) -> dict[str, Any]:
    return {
        "phase_scope": {
            "action": action,
            "name": path[-1],
            "path": path,
            "depth": len(path),
            "scope_id": scope_id,
            "parent_scope_id": None,
            "thread_id": 88,
            "thread_name": "MainThread",
            "sequence": sequence,
            "attributes": {"epoch": 3},
        }
    }


def test_derive_timeline_markers_promotes_existing_point_events() -> None:
    events = [
        _event(event_type="sample", timestamp_ns=90, context="ordinary sample"),
        _event(event_type="start", timestamp_ns=100, context="Memory tracking started"),
        _event(
            event_type="warning",
            timestamp_ns=200,
            context="High fragmentation: 40.0%",
            metadata={"fragmentation": 0.4},
        ),
        _event(
            event_type="collector_degraded",
            timestamp_ns=300,
            context="Collector degraded; telemetry is partial.",
            metadata={"collector_partial_fields": ["device_total_bytes"]},
        ),
        _event(
            event_type="error",
            timestamp_ns=400,
            context=None,
            metadata={"oom_reason": "torch.cuda.OutOfMemoryError"},
        ),
        _event(event_type="stop", timestamp_ns=500, context="Memory tracking stopped"),
    ]

    markers = derive_timeline_markers(events)

    assert [marker.kind for marker in markers] == [
        MARKER_KIND_LIFECYCLE,
        MARKER_KIND_ALERT,
        MARKER_KIND_COLLECTOR,
        MARKER_KIND_OOM,
        MARKER_KIND_LIFECYCLE,
    ]
    assert [marker.severity for marker in markers] == [
        MARKER_SEVERITY_INFO,
        MARKER_SEVERITY_WARNING,
        MARKER_SEVERITY_WARNING,
        MARKER_SEVERITY_CRITICAL,
        MARKER_SEVERITY_INFO,
    ]
    assert all(marker.source == MARKER_SOURCE_TELEMETRY_EVENT for marker in markers)
    assert markers[1].label == "High fragmentation: 40.0%"
    assert markers[2].metadata["collector_partial_fields"] == ["device_total_bytes"]
    assert markers[3].label == "OOM detected: torch.cuda.OutOfMemoryError"
    assert markers[3].event_type == "error"
    assert all(marker.end_ns is None for marker in markers)


def test_derive_timeline_markers_replays_phase_interval_markers() -> None:
    events = [
        _event(
            event_type="phase_enter",
            timestamp_ns=100,
            context="Phase entered: train / forward",
            metadata=_phase_scope(
                action="enter",
                scope_id="phase-1",
                sequence=1,
                path=["train", "forward"],
            ),
            rank=1,
            local_rank=0,
            world_size=4,
        ),
        _event(
            event_type="sample",
            timestamp_ns=150,
            context="Collected telemetry sample.",
            rank=1,
            local_rank=0,
            world_size=4,
        ),
        _event(
            event_type="phase_exit",
            timestamp_ns=250,
            context="Phase exited: train / forward",
            metadata=_phase_scope(
                action="exit",
                scope_id="phase-1",
                sequence=2,
                path=["train", "forward"],
            ),
            rank=1,
            local_rank=0,
            world_size=4,
        ),
    ]

    markers = derive_timeline_markers(events)

    assert len(markers) == 1
    marker = markers[0]
    assert marker.kind == MARKER_KIND_PHASE
    assert marker.source == MARKER_SOURCE_PHASE_REPLAY
    assert marker.severity == MARKER_SEVERITY_INFO
    assert marker.label == "Phase: train / forward"
    assert marker.start_ns == 100
    assert marker.end_ns == 250
    assert marker.is_interval is True
    assert marker.rank == 1
    assert marker.local_rank == 0
    assert marker.world_size == 4
    assert marker.metadata["phase_path"] == ["train", "forward"]
    assert marker.metadata["scope_id"] == "phase-1"
    assert marker.metadata["synthetic_end"] is False


def test_timeline_marker_to_dict_preserves_schema_fields() -> None:
    marker = derive_timeline_markers(
        [_event(event_type="critical", timestamp_ns=100, context="Critical alert")]
    )[0]

    payload = timeline_marker_to_dict(marker)

    assert payload["session_id"] == "session-1"
    assert payload["start_ns"] == 100
    assert payload["end_ns"] is None
    assert payload["kind"] == MARKER_KIND_ALERT
    assert payload["source"] == MARKER_SOURCE_TELEMETRY_EVENT
    assert payload["severity"] == MARKER_SEVERITY_CRITICAL
    assert payload["label"] == "Critical alert"
