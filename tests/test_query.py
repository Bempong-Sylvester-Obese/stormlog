from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
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
    world_size: int = 2,
    allocated: int = 100,
    reserved: int = 150,
    used: int = 175,
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
        "world_size": world_size,
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


def _write_json_events(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


def _write_oom_bundle(
    root: Path,
    *,
    session_id: str,
    session_status: str | None = SESSION_STATUS_INTERRUPTED,
    created_at_utc: str = "2026-05-12T00:00:00Z",
) -> Path:
    bundle = root / "oom_dump_20260512T000000Z_123_cuda_1"
    bundle.mkdir(parents=True)
    manifest = {
        "schema_version": 2,
        "bundle_name": bundle.name,
        "created_at_utc": created_at_utc,
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


def _iso_from_ns(timestamp_ns: int) -> str:
    timestamp_s = timestamp_ns / 1_000_000_000
    return (
        datetime.fromtimestamp(timestamp_s, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _write_diagnose_bundle(
    root: Path,
    *,
    session_id: str,
    started_at_ns: int,
    ended_at_ns: int,
) -> Path:
    summary = create_session_summary(
        source="stormlog.test.diagnose",
        status=SESSION_STATUS_COMPLETED,
        session_id=session_id,
        started_at_ns=started_at_ns,
        ended_at_ns=ended_at_ns,
        host="host-a",
        pid=123,
        job_id="job-a",
        rank=0,
        local_rank=0,
        world_size=2,
    )
    diagnose = root / "diagnose_bundle"
    diagnose.mkdir()
    (diagnose / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "created_iso": _iso_from_ns(started_at_ns),
                "command_line": "gpumemprof diagnose",
                "files": ["manifest.json"],
                "exit_code": 0,
                "risk_detected": False,
                "session_id": session_id,
                "session_status": SESSION_STATUS_COMPLETED,
                "session": session_summary_to_dict(summary),
            }
        ),
        encoding="utf-8",
    )
    return diagnose


def _write_attachment_sidecar(
    root: Path,
    *,
    session_id: str,
    start_ns: int,
    end_ns: int | None = None,
) -> Path:
    trace_path = root / "profiler.trace"
    trace_path.write_text("trace", encoding="utf-8")
    sidecar = root / "stormlog_attachments.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "stormlog.attachments",
                "attachments": [
                    {
                        "attachment_id": "profiler-trace-1",
                        "title": "Profiler trace",
                        "kind": "profiler",
                        "path": trace_path.name,
                        "session_id": session_id,
                        "job_id": "job-a",
                        "rank": 0,
                        "start_ns": start_ns,
                        "end_ns": end_ns,
                        "created_at_utc": _iso_from_ns(start_ns),
                        "updated_at_utc": _iso_from_ns(start_ns),
                        "metadata": {"tool": "profiler"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return sidecar


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


def test_correlate_collects_same_session_evidence_across_artifacts(
    tmp_path: Path,
) -> None:
    base_ns = 1_800_000_000_000_000_000
    session_id = "session-correlate"
    sink_dir = tmp_path / "sink"
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=sink_dir,
            flush_every_events=1,
            flush_every_seconds=1.0,
        )
    )
    sink.start_session(
        create_session_summary(
            source="stormlog.test",
            status=SESSION_STATUS_COMPLETED,
            session_id=session_id,
            started_at_ns=base_ns,
            host="host-a",
            pid=123,
            job_id="job-a",
            rank=0,
            local_rank=0,
            world_size=2,
        )
    )
    sink.append(
        _event_record(
            session_id=session_id,
            timestamp_ns=base_ns,
            event_type="phase_enter",
            metadata={
                "backend": "cuda",
                "phase_scope": {
                    "action": "enter",
                    "name": "forward",
                    "path": ["train", "forward"],
                    "depth": 2,
                    "scope_id": "scope-1",
                    "parent_scope_id": None,
                    "thread_id": 1,
                    "thread_name": "MainThread",
                    "sequence": 1,
                },
            },
        )
    )
    sink.append(
        _event_record(
            session_id=session_id,
            timestamp_ns=base_ns + 10,
            event_type="warning",
            context="High fragmentation: 40.0%",
        )
    )
    sink.append(
        _event_record(
            session_id=session_id,
            timestamp_ns=base_ns + 20,
            event_type="phase_exit",
            metadata={
                "backend": "cuda",
                "phase_scope": {
                    "action": "exit",
                    "name": "forward",
                    "path": ["train", "forward"],
                    "depth": 2,
                    "scope_id": "scope-1",
                    "parent_scope_id": None,
                    "thread_id": 1,
                    "thread_name": "MainThread",
                    "sequence": 2,
                },
            },
        )
    )
    sink.close()
    _write_oom_bundle(
        tmp_path,
        session_id=session_id,
        session_status=SESSION_STATUS_COMPLETED,
        created_at_utc=_iso_from_ns(base_ns + 10),
    )
    _write_diagnose_bundle(
        tmp_path,
        session_id=session_id,
        started_at_ns=base_ns,
        ended_at_ns=base_ns + 30,
    )
    _write_attachment_sidecar(
        tmp_path,
        session_id=session_id,
        start_ns=base_ns,
        end_ns=base_ns + 30,
    )

    result = query_api.open([tmp_path]).correlate(
        query_api.CorrelationFilter(
            session_id=session_id,
            at_ns=base_ns + 10,
            window_ns=1_000,
        )
    )

    kinds = {row.kind for row in result.evidence}
    assert {
        "telemetry_event",
        "timeline_marker",
        "alert",
        "rollup_window",
        "oom_bundle",
        "diagnose_bundle",
        "attachment",
    }.issubset(kinds)
    assert all(row.confidence in {"high", "medium"} for row in result.evidence)
    attachment = next(row for row in result.evidence if row.kind == "attachment")
    assert attachment.source_path.endswith("profiler.trace")
    assert result.anchor["clock_domain"] == "unix_epoch_ns"
    assert result.anchor["clock_normalization"] == "producer_emitted_epoch_ns"


def test_correlate_distributed_scope_uses_job_id_across_ranks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "track.json"
    _write_json_events(
        path,
        [
            _event_record(session_id="session-r0", timestamp_ns=100, rank=0),
            _event_record(session_id="session-r1", timestamp_ns=105, rank=1),
        ],
    )

    result = query_api.open([path]).correlate(
        query_api.CorrelationFilter(
            job_id="job-a",
            scope="distributed",
            at_ns=100,
            window_ns=10,
        )
    )

    event_rows = [row for row in result.evidence if row.kind == "telemetry_event"]
    assert {row.rank for row in event_rows} == {0, 1}
    assert {row.confidence for row in event_rows} <= {"high", "medium"}
    assert any("same_job_distributed" in row.reasons for row in event_rows)


def test_correlate_allows_low_confidence_time_only_matches(tmp_path: Path) -> None:
    path = tmp_path / "track.json"
    _write_json_events(
        path,
        [_event_record(session_id="session-time-only", timestamp_ns=100)],
    )

    result = query_api.open([path]).correlate(
        query_api.CorrelationFilter(at_ns=100, window_ns=0)
    )

    assert result.evidence
    assert {row.confidence for row in result.evidence} == {"low"}
    assert all(
        "time_only_missing_shared_identifier" in row.reasons for row in result.evidence
    )


def test_attachment_sidecar_discovery_resolves_paths_and_reports_warnings(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    sidecar = _write_attachment_sidecar(
        tmp_path,
        session_id="session-attachment",
        start_ns=100,
    )
    bad_sidecar_dir = tmp_path / "bad"
    bad_sidecar_dir.mkdir()
    (bad_sidecar_dir / "stormlog_attachments.json").write_text(
        json.dumps({"schema_version": 1, "format": "wrong"}),
        encoding="utf-8",
    )

    def _fail_load(*args: Any, **kwargs: Any) -> list[Any]:
        raise AssertionError("attachment listing should not materialize telemetry")

    monkeypatch.setattr(query_api, "load_telemetry_sessions", _fail_load)

    store = query_api.open([tmp_path])
    rows = store.list_attachments(
        query_api.AttachmentFilter(session_id="session-attachment")
    )

    assert len(rows) == 1
    assert rows[0].sidecar_path == str(sidecar)
    assert rows[0].attachment_id == "profiler-trace-1"
    assert rows[0].updated_at_utc == rows[0].created_at_utc
    assert rows[0].path is not None
    assert rows[0].path.endswith("profiler.trace")
    assert any(
        "unrecognized attachment sidecar" in item.message
        for item in store.catalog.warnings
    )


def test_list_sessions_discovers_sink_manifest_file(tmp_path: Path) -> None:
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
        )
    )
    sink.append(_event_record(session_id="session-a", timestamp_ns=1))
    sink.close()

    rows = query_api.open([tmp_path / "manifest.json"]).list_sessions()

    assert len(rows) == 1
    assert rows[0].session_id == "session-a"
    assert rows[0].source_kind == "sink"


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


def test_query_summary_uses_fresh_sink_rollup_without_loading_events(
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
    sink.append(
        _event_record(
            session_id="session-rollup",
            timestamp_ns=1,
            allocated=100,
            reserved=150,
        )
    )
    sink.append(
        _event_record(
            session_id="session-rollup",
            timestamp_ns=2,
            allocated=200,
            reserved=275,
        )
    )
    sink.close()

    def _fail_load(*args: Any, **kwargs: Any) -> list[Any]:
        raise AssertionError("fresh rollup summary should not materialize events")

    monkeypatch.setattr(query_api, "load_telemetry_sessions", _fail_load)

    rows = query_api.open([tmp_path]).summarize(
        "peak_allocator_reserved_bytes",
        group_by="session",
    )

    assert len(rows) == 1
    assert rows[0].session_id == "session-rollup"
    assert rows[0].value == 275
    assert rows[0].details["timestamp_ns"] == 2


def test_query_summary_falls_back_when_sink_rollup_is_stale(
    tmp_path: Path,
) -> None:
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
        )
    )
    sink.append(
        _event_record(
            session_id="session-stale",
            timestamp_ns=1,
            reserved=150,
        )
    )
    sink.close()
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["segments"][0]["event_count"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    rows = query_api.open([tmp_path]).summarize(
        "peak_allocator_reserved_bytes",
        group_by="session",
    )

    assert len(rows) == 1
    assert rows[0].session_id == "session-stale"
    assert rows[0].value == 150


def test_query_summary_falls_back_when_sink_rollup_is_malformed(
    tmp_path: Path,
) -> None:
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
        )
    )
    sink.append(
        _event_record(
            session_id="session-malformed",
            timestamp_ns=1,
            reserved=160,
        )
    )
    sink.close()
    (tmp_path / "rollups.json").write_text("{bad", encoding="utf-8")

    rows = query_api.open([tmp_path]).summarize(
        "peak_allocator_reserved_bytes",
        group_by="session",
    )

    assert len(rows) == 1
    assert rows[0].session_id == "session-malformed"
    assert rows[0].value == 160


def test_query_hidden_gap_rollup_order_matches_raw_fallback(
    tmp_path: Path,
) -> None:
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
        )
    )
    for rank, used_values in ((10, (200, 260)), (2, (180, 230))):
        for offset, used in enumerate(used_values):
            sink.append(
                _event_record(
                    session_id="session-order",
                    timestamp_ns=rank * 10 + offset,
                    rank=rank,
                    world_size=16,
                    reserved=100,
                    used=used,
                )
            )
    sink.close()

    store = query_api.open([tmp_path])
    rollup_rows = store.summarize("hidden_memory_gap_growth", group_by="rank")
    (tmp_path / "rollups.json").write_text("{bad", encoding="utf-8")
    raw_rows = query_api.open([tmp_path]).summarize(
        "hidden_memory_gap_growth",
        group_by="rank",
    )

    assert [(row.session_id, row.rank, row.status) for row in rollup_rows] == [
        (row.session_id, row.rank, row.status) for row in raw_rows
    ]
    assert [row.value for row in rollup_rows] == [row.value for row in raw_rows]


def test_query_alert_count_rollup_matches_raw_fallback_for_metadata_severity(
    tmp_path: Path,
) -> None:
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
        )
    )
    sink.append(
        _event_record(
            session_id="session-alert",
            timestamp_ns=1,
            event_type="sample",
            metadata={"backend": "cuda", "severity": " Warning "},
        )
    )
    sink.close()

    rollup_rows = query_api.open([tmp_path]).summarize(
        "alert_count",
        group_by="session",
    )
    (tmp_path / "rollups.json").write_text("{bad", encoding="utf-8")
    raw_rows = query_api.open([tmp_path]).summarize(
        "alert_count",
        group_by="session",
    )

    assert [(row.session_id, row.value) for row in rollup_rows] == [
        (row.session_id, row.value) for row in raw_rows
    ]
    assert rollup_rows[0].value == 1


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


def test_csv_telemetry_preserves_large_integer_fields(tmp_path: Path) -> None:
    path = tmp_path / "track.csv"
    record = _event_record(
        session_id="session-csv",
        timestamp_ns=1_700_000_000_000_000_123,
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(
            {
                key: json.dumps(value) if key == "metadata" else value
                for key, value in record.items()
            }
        )

    rows = query_api.open([path]).query_events()

    assert len(rows) == 1
    assert rows[0].event.timestamp_ns == record["timestamp_ns"]
    assert rows[0].source_kind == "telemetry_csv"


def test_list_issues_groups_alerts_across_sessions(tmp_path: Path) -> None:
    path = tmp_path / "track.json"
    _write_json_events(
        path,
        [
            _event_record(
                session_id="session-alert-a",
                timestamp_ns=10,
                event_type="warning",
                context="High fragmentation: 40.0%",
            ),
            _event_record(
                session_id="session-alert-b",
                timestamp_ns=20,
                event_type="warning",
                context="High fragmentation: 51.5%",
            ),
        ],
    )

    rows = query_api.open([path]).list_issues(
        query_api.IssueFilter(kind="alert", session_id="session-alert-a")
    )

    assert len(rows) == 1
    issue = rows[0]
    assert issue.kind == "alert"
    assert issue.state == "open"
    assert issue.hit_count == 2
    assert issue.first_seen_ns == 10
    assert issue.last_seen_ns == 20
    assert issue.affected_sessions == ("session-alert-a", "session-alert-b")
    assert issue.fingerprint.dimensions["category"] == "high_fragmentation"
    assert issue.representative_evidence.event_type == "warning"


def test_list_issues_supports_state_overrides(tmp_path: Path) -> None:
    path = tmp_path / "track.json"
    _write_json_events(
        path,
        [
            _event_record(
                session_id="session-collector",
                timestamp_ns=10,
                event_type="collector_degraded",
                metadata={
                    "backend": "cuda",
                    "collector_health_status": "degraded",
                    "collector_partial_fields": ["device_free_bytes"],
                    "collector_last_error": "RuntimeError: failed at sample 42",
                },
            )
        ],
    )
    store = query_api.open([path])
    original = store.list_issues(query_api.IssueFilter(kind="collector_degradation"))[0]

    overridden = store.list_issues(
        query_api.IssueFilter(state="ignored"),
        state_overrides={original.fingerprint_id: "ignored"},
    )

    assert len(overridden) == 1
    assert overridden[0].state == "ignored"
    assert overridden[0].details["error_stem"] == "runtimeerror"


def test_list_issues_includes_oom_bundles_and_telemetry_ooms(tmp_path: Path) -> None:
    _write_oom_bundle(tmp_path, session_id="session-oom-bundle")
    path = tmp_path / "track.json"
    _write_json_events(
        path,
        [
            _event_record(
                session_id="session-oom-event",
                timestamp_ns=50,
                event_type="error",
                metadata={
                    "backend": "cuda",
                    "oom_reason": "message_pattern:out of memory",
                    "oom_dump_path": str(tmp_path / "oom"),
                },
            )
        ],
    )

    rows = query_api.open([tmp_path]).list_issues(query_api.IssueFilter(kind="oom"))

    assert len(rows) == 1
    assert rows[0].severity == "critical"
    assert rows[0].hit_count == 2
    assert rows[0].fingerprint.dimensions == {
        "backend": "cuda",
        "reason": "message_pattern:out of memory",
    }
    assert {"session-oom-bundle", "session-oom-event"} == {
        session for row in rows for session in row.affected_sessions
    }


def test_list_issues_includes_hidden_memory_anomalies(tmp_path: Path) -> None:
    path = tmp_path / "track.json"
    _write_json_events(
        path,
        [
            _event_record(
                session_id="session-gap",
                timestamp_ns=timestamp_ns,
                allocated=90,
                reserved=100,
                used=used,
            )
            for timestamp_ns, used in enumerate([160, 220, 280, 340, 400], start=1)
        ],
    )

    rows = query_api.open([path]).list_issues(
        query_api.IssueFilter(kind="hidden_memory_anomaly")
    )

    assert rows
    assert rows[0].kind == "hidden_memory_anomaly"
    assert rows[0].details["classification"] == "persistent_drift"
    assert rows[0].affected_sessions == ("session-gap",)
