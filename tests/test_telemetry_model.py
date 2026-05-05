"""Tests for backend-neutral telemetry record projection."""

from __future__ import annotations

import json
from pathlib import Path

from stormlog.telemetry import (
    SCHEMA_VERSION_V3,
    canonical_record_from_telemetry_event,
    canonical_records_from_telemetry_events,
    load_telemetry_sessions,
    telemetry_event_from_record,
)
from stormlog.telemetry_model import canonical_record_to_dict


def _v3_record() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION_V3,
        "session_id": "session-1",
        "timestamp_ns": 1_700_000_000_000_000_000,
        "event_type": "phase_enter",
        "collector": "stormlog.cuda_tracker",
        "sampling_interval_ms": 100,
        "pid": 1234,
        "host": "host-a",
        "job_id": "job-1",
        "rank": 1,
        "local_rank": 0,
        "world_size": 2,
        "device_id": 0,
        "allocator_allocated_bytes": 1024,
        "allocator_reserved_bytes": 2048,
        "allocator_active_bytes": 512,
        "allocator_inactive_bytes": 256,
        "allocator_change_bytes": 128,
        "device_used_bytes": 4096,
        "device_free_bytes": 8192,
        "device_total_bytes": 12288,
        "context": "Phase entered: train / forward",
        "metadata": {
            "backend": "cuda",
            "phase_scope": {
                "action": "enter",
                "name": "forward",
                "scope_id": "session-1:2",
                "parent_scope_id": "session-1:1",
                "sequence": 2,
            },
        },
    }


def test_canonical_record_projects_v3_into_backend_neutral_envelope() -> None:
    event = telemetry_event_from_record(_v3_record())

    canonical = canonical_record_from_telemetry_event(event)
    payload = canonical_record_to_dict(canonical)

    assert payload["schema_version"] == 1
    assert payload["session_id"] == "session-1"
    assert payload["source_kind"] == "cuda"
    assert payload["event_type"] == "phase_enter"
    assert payload["stage"] == "forward"
    assert payload["severity"] == "info"
    assert payload["body"] == "Phase entered: train / forward"
    assert payload["resource"]["collector"] == "stormlog.cuda_tracker"
    assert payload["resource"]["device_id"] == 0
    assert payload["attributes"]["memory.allocator.allocated_bytes"] == 1024
    assert payload["attributes"]["memory.device.total_bytes"] == 12288
    assert payload["correlation"]["phase.scope_id"] == "session-1:2"


def test_canonical_record_id_is_deterministic_for_same_event() -> None:
    event = telemetry_event_from_record(_v3_record())

    first = canonical_record_from_telemetry_event(event)
    second = canonical_record_from_telemetry_event(event)

    assert first.record_id == second.record_id


def test_canonical_record_accepts_legacy_mapping_through_existing_normalizer() -> None:
    canonical = canonical_record_from_telemetry_event(
        {
            "timestamp": 1_700_000_000.0,
            "event_type": "sample",
            "memory_allocated": 1024,
            "memory_reserved": 2048,
            "memory_change": 0,
            "metadata": {"backend": "cpu"},
        }
    )

    assert canonical.source_kind == "cpu"
    assert canonical.attributes["memory.allocator.allocated_bytes"] == 1024
    assert canonical.resource["collector"] == "stormlog.cpu_tracker"


def test_loaded_session_exposes_canonical_records_resources_and_correlations(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.json"
    path.write_text(json.dumps([_v3_record()]), encoding="utf-8")

    sessions = load_telemetry_sessions(path)

    assert len(sessions) == 1
    loaded = sessions[0]
    records = loaded.telemetry_records()
    resources = loaded.resources()
    correlations = loaded.correlations()

    assert len(records) == 1
    assert resources == [records[0].resource]
    assert correlations == [records[0].correlation]


def test_canonical_records_from_telemetry_events_preserves_order() -> None:
    first_record = _v3_record()
    second_record = dict(first_record)
    second_record["timestamp_ns"] = 1_700_000_000_000_000_100
    second_record["event_type"] = "phase_exit"

    events = [
        telemetry_event_from_record(first_record),
        telemetry_event_from_record(second_record),
    ]

    records = canonical_records_from_telemetry_events(events)

    assert [record.event_type for record in records] == ["phase_enter", "phase_exit"]
