from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import stormlog.query as query_api
from stormlog.session import (
    SESSION_STATUS_COMPLETED,
    SESSION_STATUS_INCOMPLETE,
    SESSION_STATUS_INTERRUPTED,
    create_session_summary,
    session_summary_to_dict,
)
from stormlog.telemetry_sink import AppendOnlyTelemetrySink, TelemetrySinkConfig


def _event_record(
    *,
    session_id: str,
    timestamp_ns: int,
    event_type: str = "sample",
    rank: int = 0,
    allocated: int = 100,
    reserved: int = 150,
    used: int = 175,
    metadata: dict[str, Any] | None = None,
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
        "context": event_type,
        "metadata": metadata or {"backend": "cuda"},
    }


def _write_json_events(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


def _write_oom_bundle(
    root: Path,
    *,
    session_id: str,
    session_status: str | None = SESSION_STATUS_INTERRUPTED,
) -> Path:
    bundle = root / "oom_dump_20260512T000000Z_123_cuda_1"
    bundle.mkdir(parents=True)
    manifest = {
        "schema_version": 2,
        "bundle_name": bundle.name,
        "created_at_utc": "2026-05-12T00:00:00Z",
        "reason": "message_pattern:out of memory",
        "backend": "cuda",
        "event_count": 2,
        "session_id": session_id,
        "files": ["manifest.json", "metadata.json"],
    }
    if session_status is not None:
        manifest["session_status"] = session_status
    metadata = {
        "exception_type": "RuntimeError",
        "exception_module": "builtins",
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (bundle / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return bundle


def test_list_sessions_uses_sink_manifest_without_loading_events(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
        )
    )
    sink.append(_event_record(session_id="session-a", timestamp_ns=1))
    sink.close()

    def _fail_load(*args: Any, **kwargs: Any) -> list[Any]:
        raise AssertionError("list_sessions should not materialize sink events")

    monkeypatch.setattr(query_api, "load_telemetry_sessions", _fail_load)

    store = query_api.open([tmp_path])
    rows = store.list_sessions()

    assert len(rows) == 1
    assert rows[0].session_id == "session-a"
    assert rows[0].status == SESSION_STATUS_COMPLETED
    assert rows[0].source_kind == "sink"
    assert rows[0].event_count == 1


def test_query_events_filters_and_adds_provenance(tmp_path: Path) -> None:
    path = tmp_path / "track.json"
    _write_json_events(
        path,
        [
            _event_record(session_id="session-a", timestamp_ns=1, rank=0),
            _event_record(
                session_id="session-a",
                timestamp_ns=2,
                event_type="collector_degraded",
                rank=1,
                metadata={
                    "backend": "cuda",
                    "collector_health_status": "degraded",
                },
            ),
        ],
    )

    rows = query_api.open([path]).query_events(
        query_api.EventFilter(
            rank=1,
            event_type="collector_degraded",
            collector_health_status="degraded",
            backend="cuda",
        )
    )

    assert len(rows) == 1
    payload = rows[0].as_dict()
    assert payload["session_id"] == "session-a"
    assert payload["rank"] == 1
    assert payload["source_kind"] == "telemetry_json"
    assert payload["source_path"].endswith("track.json")
    assert payload["session_status"] == SESSION_STATUS_INCOMPLETE


def test_query_events_limit_applies_after_global_sort(tmp_path: Path) -> None:
    late_path = tmp_path / "late_track.json"
    early_path = tmp_path / "early_track.json"
    _write_json_events(
        late_path,
        [_event_record(session_id="session-late", timestamp_ns=200)],
    )
    _write_json_events(
        early_path,
        [_event_record(session_id="session-early", timestamp_ns=100)],
    )

    rows = query_api.open([late_path, early_path]).query_events(
        query_api.EventFilter(limit=1)
    )

    assert len(rows) == 1
    assert rows[0].event.session_id == "session-early"
    assert rows[0].event.timestamp_ns == 100


def test_list_oom_bundles_links_to_sessions(tmp_path: Path) -> None:
    session = create_session_summary(
        source="stormlog.test",
        status=SESSION_STATUS_INTERRUPTED,
        session_id="session-oom",
        started_at_ns=10,
        host="host-a",
        pid=123,
    )
    diagnose = tmp_path / "diag"
    diagnose.mkdir()
    (diagnose / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "created_iso": "2026-05-12T00:00:00Z",
                "command_line": "gpumemprof diagnose",
                "files": ["manifest.json"],
                "exit_code": 2,
                "risk_detected": True,
                "session_id": "session-oom",
                "session_status": SESSION_STATUS_INTERRUPTED,
                "session": session_summary_to_dict(session),
            }
        ),
        encoding="utf-8",
    )
    _write_oom_bundle(tmp_path, session_id="session-oom")

    store = query_api.open([tmp_path])
    session_rows = store.list_sessions(query_api.SessionFilter(has_oom_bundle=True))
    oom_rows = store.list_oom_bundles(query_api.OOMBundleFilter(backend="cuda"))

    assert [row.session_id for row in session_rows] == ["session-oom"]
    assert session_rows[0].oom_bundle_count == 1
    assert len(oom_rows) == 1
    assert oom_rows[0].session_id == "session-oom"
    assert oom_rows[0].session_status == SESSION_STATUS_INTERRUPTED
    assert oom_rows[0].exception_type == "RuntimeError"


def test_list_oom_bundles_uses_only_manifest_backed_session_status(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    session = create_session_summary(
        source="stormlog.test",
        status=SESSION_STATUS_COMPLETED,
        session_id="session-manifest-only",
        started_at_ns=10,
        host="host-a",
        pid=123,
    )
    diagnose = tmp_path / "diag"
    diagnose.mkdir()
    (diagnose / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "created_iso": "2026-05-12T00:00:00Z",
                "command_line": "gpumemprof diagnose",
                "files": ["manifest.json"],
                "exit_code": 0,
                "risk_detected": False,
                "session_id": "session-manifest-only",
                "session_status": SESSION_STATUS_COMPLETED,
                "session": session_summary_to_dict(session),
            }
        ),
        encoding="utf-8",
    )
    _write_oom_bundle(
        tmp_path,
        session_id="session-manifest-only",
        session_status=None,
    )
    _write_json_events(
        tmp_path / "track.json",
        [_event_record(session_id="flat-session", timestamp_ns=1)],
    )

    def _fail_load(*args: Any, **kwargs: Any) -> list[Any]:
        raise AssertionError("OOM listing should not materialize flat telemetry")

    monkeypatch.setattr(query_api, "load_telemetry_sessions", _fail_load)

    rows = query_api.open([tmp_path]).list_oom_bundles()

    assert len(rows) == 1
    assert rows[0].session_id == "session-manifest-only"
    assert rows[0].session_status == SESSION_STATUS_COMPLETED


def test_query_summaries_cover_sessions_peaks_alerts_and_gap_growth(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.json"
    _write_json_events(
        path,
        [
            _event_record(
                session_id="session-a",
                timestamp_ns=1,
                rank=0,
                allocated=100,
                reserved=150,
                used=200,
            ),
            _event_record(
                session_id="session-a",
                timestamp_ns=2,
                rank=0,
                allocated=180,
                reserved=220,
                used=330,
            ),
            _event_record(
                session_id="session-a",
                timestamp_ns=3,
                event_type="warning",
                rank=0,
            ),
            _event_record(
                session_id="session-a",
                timestamp_ns=4,
                event_type="collector_degraded",
                rank=1,
            ),
        ],
    )
    store = query_api.open([path])

    status_rows = store.summarize("session_count_by_status")
    peak_rows = store.summarize(
        "peak_allocator_reserved_bytes",
        group_by="session",
    )
    alert_rows = store.summarize("alert_count", group_by="session-rank")
    collector_rows = store.summarize(
        "collector_degradation_transitions",
        group_by="rank",
    )
    gap_rows = store.summarize("hidden_memory_gap_growth", group_by="session")

    assert status_rows[0].status == "incomplete"
    assert status_rows[0].value == 1
    assert peak_rows[0].value == 220
    assert alert_rows[0].value == 1
    assert collector_rows[0].rank == 1
    assert collector_rows[0].value == 1
    assert gap_rows[0].value == 60
    assert gap_rows[0].details["peak_gap_bytes"] == 110


def test_catalog_discovers_csv_telemetry(tmp_path: Path) -> None:
    path = tmp_path / "track.csv"
    record = _event_record(session_id="session-csv", timestamp_ns=1)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(
            {
                key: json.dumps(value) if key == "metadata" else value
                for key, value in record.items()
            }
        )

    rows = query_api.open([tmp_path]).list_sessions(
        query_api.SessionFilter(source_kind="telemetry_csv")
    )

    assert len(rows) == 1
    assert rows[0].session_id == "session-csv"
    assert rows[0].source_kind == "telemetry_csv"
