from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from stormlog.query_cli import main as query_main
from stormlog.tui import run_app


def _event_record(
    *,
    session_id: str = "session-cli",
    timestamp_ns: int = 1,
    event_type: str = "sample",
    rank: int = 0,
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
        "allocator_allocated_bytes": 100,
        "allocator_reserved_bytes": 150,
        "allocator_active_bytes": None,
        "allocator_inactive_bytes": None,
        "allocator_change_bytes": 50,
        "device_used_bytes": 200,
        "device_free_bytes": None,
        "device_total_bytes": 1000,
        "context": event_type,
        "metadata": {"backend": "cuda"},
    }


def _write_json_events(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


def test_query_sessions_json_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "track.json"
    _write_json_events(path, [_event_record()])

    assert query_main(["sessions", str(path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["session_id"] == "session-cli"
    assert payload[0]["source_kind"] == "telemetry_json"


def test_query_events_table_and_limit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "track.json"
    _write_json_events(
        path,
        [
            _event_record(timestamp_ns=1, rank=0),
            _event_record(timestamp_ns=2, rank=1),
        ],
    )

    assert query_main(["events", str(path), "--rank", "1", "--limit", "1"]) == 0

    output = capsys.readouterr().out
    assert "Session Id" in output
    assert "session-cli" in output
    assert "  1  " in output or output.rstrip().endswith("  1")


def test_query_ooms_csv_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = tmp_path / "oom_dump_20260512T000000Z_123_cuda_1"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "bundle_name": bundle.name,
                "created_at_utc": "2026-05-12T00:00:00Z",
                "reason": "message_pattern:out of memory",
                "backend": "cuda",
                "event_count": 1,
                "session_id": "session-cli",
                "session_status": "interrupted",
                "files": ["manifest.json"],
            }
        ),
        encoding="utf-8",
    )

    assert query_main(["ooms", str(tmp_path), "--csv"]) == 0

    output = capsys.readouterr().out
    assert "bundle_path,created_at_utc,backend" in output
    assert "session-cli" in output


def test_query_summary_rejects_csv(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "track.json"
    _write_json_events(path, [_event_record()])

    assert (
        query_main(
            [
                "summary",
                str(path),
                "--metric",
                "session_count_by_status",
                "--csv",
            ]
        )
        == 2
    )

    assert "--csv is not supported" in capsys.readouterr().err


def test_stormlog_dispatcher_preserves_no_arg_tui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    fake_app = types.ModuleType("stormlog.tui.app")

    def _fake_run_app() -> None:
        calls.append("tui")

    fake_app.run_app = _fake_run_app  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "stormlog.tui.app", fake_app)

    run_app([])

    assert calls == ["tui"]


def test_stormlog_dispatcher_routes_query(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "track.json"
    _write_json_events(path, [_event_record()])

    with pytest.raises(SystemExit) as excinfo:
        run_app(["query", "sessions", str(path), "--json"])

    assert excinfo.value.code == 0
    assert json.loads(capsys.readouterr().out)[0]["session_id"] == "session-cli"
