from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stormlog.session import create_session_summary
from stormlog.telemetry import LoadedTelemetrySession, telemetry_event_from_record
from stormlog.telemetry_rollups import (
    DEFAULT_ROLLUP_WINDOW_NS,
    ROLLUP_FILENAME,
    RollupCoverage,
    build_telemetry_rollups,
    read_telemetry_rollups,
    telemetry_rollup_file_from_dict,
    telemetry_rollup_file_to_dict,
    write_telemetry_rollups,
)
from stormlog.telemetry_sink import TelemetrySinkManifest, TelemetrySinkSegment


def _event_record(
    *,
    session_id: str = "session-rollup",
    timestamp_ns: int,
    event_type: str = "sample",
    rank: int = 0,
    allocated: int = 100,
    reserved: int = 150,
    used: int = 180,
    metadata: dict[str, Any] | None = None,
    context: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "session_id": session_id,
        "timestamp_ns": timestamp_ns,
        "event_type": event_type,
        "collector": "stormlog.cuda_tracker",
        "sampling_interval_ms": 100,
        "pid": 123,
        "host": "host-a",
        "job_id": "job-a",
        "rank": rank,
        "local_rank": rank,
        "world_size": 2,
        "device_id": 0,
        "allocator_allocated_bytes": allocated,
        "allocator_reserved_bytes": reserved,
        "allocator_active_bytes": None,
        "allocator_inactive_bytes": None,
        "allocator_change_bytes": reserved - allocated,
        "device_used_bytes": used,
        "device_free_bytes": None,
        "device_total_bytes": 1000,
        "context": context or event_type,
        "metadata": metadata or {"backend": "cuda"},
    }


def _loaded_session(records: list[dict[str, Any]]) -> LoadedTelemetrySession:
    events = [telemetry_event_from_record(record) for record in records]
    summary = create_session_summary(
        source="stormlog.test",
        status="completed",
        session_id="session-rollup",
        started_at_ns=0,
        ended_at_ns=180 * 1_000_000_000,
        host="host-a",
        pid=123,
        job_id="job-a",
        world_size=2,
    )
    return LoadedTelemetrySession(
        summary=summary,
        events=events,
        sources_loaded=["segment-000001.jsonl"],
    )


def _manifest() -> TelemetrySinkManifest:
    return TelemetrySinkManifest(
        schema_version=2,
        format="stormlog.append_only_telemetry_sink",
        sessions=[],
        segments=[
            TelemetrySinkSegment(
                filename="segment-000001.jsonl",
                event_count=7,
                size_bytes=700,
                closed=True,
                session_id="session-rollup",
            )
        ],
    )


def test_build_telemetry_rollups_summarizes_sessions_ranks_and_windows() -> None:
    one_second = 1_000_000_000
    loaded = _loaded_session(
        [
            _event_record(timestamp_ns=1 * one_second, allocated=100, reserved=150),
            _event_record(timestamp_ns=20 * one_second, allocated=140, reserved=160),
            _event_record(
                timestamp_ns=61 * one_second,
                rank=1,
                allocated=220,
                reserved=240,
                used=390,
            ),
            _event_record(
                timestamp_ns=70 * one_second,
                event_type="warning",
                rank=1,
                metadata={"backend": "cuda", "severity": "warning"},
            ),
            _event_record(
                timestamp_ns=80 * one_second,
                event_type="collector_degraded",
                metadata={"backend": "cuda", "collector_health_status": "degraded"},
            ),
            _event_record(
                timestamp_ns=140 * one_second,
                event_type="collector_recovered",
                metadata={"backend": "cuda", "collector_health_status": "healthy"},
            ),
            _event_record(
                timestamp_ns=150 * one_second,
                event_type="error",
                metadata={
                    "backend": "cuda",
                    "oom_reason": "message_pattern:out of memory",
                    "oom_dump_path": "oom_dump",
                },
            ),
        ]
    )

    rollups = build_telemetry_rollups([loaded], _manifest())
    session = rollups.sessions[0]

    assert rollups.coverage.retained_segment_filenames == ["segment-000001.jsonl"]
    assert session.event_count == 7
    assert session.sample_count == 3
    assert session.counters.allocator_allocated_bytes.value == 220
    assert session.counters.allocator_allocated_bytes.timestamp_ns == 61 * one_second
    assert session.alerts.total_count == 2
    assert session.alerts.severity_counts == {"critical": 1, "warning": 1}
    assert session.alerts.event_type_counts == {"error": 1, "warning": 1}
    assert session.collector_health.transition_count == 2
    assert session.collector_health.degraded_time_ns == 60 * one_second
    assert session.collector_health.last_status == "healthy"
    assert session.oom.marker_count == 1
    assert session.oom.bundle_path_count == 1

    rank_one = session.ranks[1]
    assert rank_one.rank == 1
    assert rank_one.sample_count == 1
    assert rank_one.hidden_gap_first_bytes == 150
    assert rank_one.hidden_gap_latest_bytes == 150
    assert rank_one.hidden_gap_peak_bytes == 150

    assert [window.index for window in session.windows] == [0, 1, 2]
    assert session.windows[0].sample_count == 2
    assert session.windows[1].rank_count == 2
    assert session.windows[1].alert_count == 1
    assert session.windows[2].oom_count == 1


def test_rank_hidden_gap_rollup_tracks_delta_and_drift() -> None:
    one_second = 1_000_000_000
    loaded = _loaded_session(
        [
            _event_record(timestamp_ns=0, used=160, reserved=100),
            _event_record(timestamp_ns=10 * one_second, used=190, reserved=100),
            _event_record(timestamp_ns=20 * one_second, used=260, reserved=100),
        ]
    )

    rank = build_telemetry_rollups([loaded], None).sessions[0].ranks[0]

    assert rank.hidden_gap_first_bytes == 60
    assert rank.hidden_gap_latest_bytes == 160
    assert rank.hidden_gap_peak_bytes == 160
    assert rank.hidden_gap_delta_bytes == 100
    assert rank.hidden_gap_drift_bytes_per_second == 5.0


def test_rollup_sidecar_round_trips_and_validates_manifest_freshness(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    rollups = build_telemetry_rollups(
        [_loaded_session([_event_record(timestamp_ns=1)])],
        manifest,
        coverage=RollupCoverage(
            retained_segment_filenames=["segment-000001.jsonl"],
            retained_segment_count=1,
            retained_event_count=1,
            retained_bytes=700,
            pruned_segment_count=2,
            pruned_bytes=1400,
        ),
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "format": "stormlog.append_only_telemetry_sink",
                "segments": [
                    {
                        "filename": "segment-000001.jsonl",
                        "event_count": 1,
                        "size_bytes": 700,
                        "closed": True,
                        "session_id": "session-rollup",
                    }
                ],
                "sessions": [],
            }
        ),
        encoding="utf-8",
    )

    path = write_telemetry_rollups(tmp_path, rollups)
    loaded = read_telemetry_rollups(tmp_path)
    assert path.name == ROLLUP_FILENAME
    assert loaded is not None
    assert loaded.coverage.pruned_segment_count == 2
    assert loaded.window_duration_ns == DEFAULT_ROLLUP_WINDOW_NS

    payload = telemetry_rollup_file_to_dict(loaded)
    reparsed = telemetry_rollup_file_from_dict(payload)
    assert reparsed.coverage.retained_bytes == 700

    manifest_payload = json.loads((tmp_path / "manifest.json").read_text())
    manifest_payload["segments"][0]["event_count"] = 2
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest_payload),
        encoding="utf-8",
    )
    assert read_telemetry_rollups(tmp_path) is None


def test_read_telemetry_rollups_tolerates_missing_and_malformed(
    tmp_path: Path,
) -> None:
    assert read_telemetry_rollups(tmp_path) is None
    (tmp_path / ROLLUP_FILENAME).write_text("{bad", encoding="utf-8")
    assert read_telemetry_rollups(tmp_path) is None
