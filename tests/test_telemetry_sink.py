from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from stormlog.telemetry_sink import AppendOnlyTelemetrySink, TelemetrySinkConfig


def _segment_records(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_append_only_sink_writes_jsonl_segment_and_manifest(tmp_path: Path) -> None:
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
        )
    )

    sink.append({"schema_version": 2, "event_type": "start", "seq": 1})
    sink.close()

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert len(manifest["segments"]) == 1
    assert manifest["segments"][0]["event_count"] == 1
    assert manifest["segments"][0]["closed"] is True

    records = _segment_records(tmp_path / manifest["segments"][0]["filename"])
    assert records == [{"event_type": "start", "schema_version": 2, "seq": 1}]


def test_append_only_sink_rolls_over_by_event_count(tmp_path: Path) -> None:
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
            rollover_max_events=2,
            retention_max_files=4,
        )
    )

    sink.append({"seq": 1})
    sink.append({"seq": 2})
    sink.append({"seq": 3})
    sink.close()

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    segments = manifest["segments"]
    assert [segment["event_count"] for segment in segments] == [2, 1]
    assert [segment["closed"] for segment in segments] == [True, True]

    first = _segment_records(tmp_path / segments[0]["filename"])
    second = _segment_records(tmp_path / segments[1]["filename"])
    assert [record["seq"] for record in first] == [1, 2]
    assert [record["seq"] for record in second] == [3]


def test_append_only_sink_prunes_oldest_closed_segments(tmp_path: Path) -> None:
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=1,
            flush_every_seconds=1.0,
            rollover_max_events=1,
            rollover_max_bytes=1024,
            retention_max_files=2,
            retention_max_total_bytes=1024 * 1024,
        )
    )

    sink.append({"seq": 1})
    sink.append({"seq": 2})
    sink.append({"seq": 3})
    sink.close()

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    filenames = [segment["filename"] for segment in manifest["segments"]]
    assert filenames == ["segment-000002.jsonl", "segment-000003.jsonl"]
    assert not (tmp_path / "segment-000001.jsonl").exists()


def test_append_only_sink_flushes_without_new_events(tmp_path: Path) -> None:
    sink = AppendOnlyTelemetrySink(
        TelemetrySinkConfig(
            root_dir=tmp_path,
            flush_every_events=10,
            flush_every_seconds=0.05,
        )
    )

    try:
        sink.append({"seq": 1})

        segment_path = tmp_path / "segment-000001.jsonl"
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if segment_path.exists() and _segment_records(segment_path) == [{"seq": 1}]:
                break
            time.sleep(0.01)
        else:
            pytest.fail("append-only sink did not flush buffered records in time")
    finally:
        sink.close()


def test_telemetry_sink_config_rejects_total_retention_below_rollover() -> None:
    with pytest.raises(
        ValueError, match="retention_max_total_bytes must be >= rollover_max_bytes"
    ):
        TelemetrySinkConfig(
            root_dir=Path("/tmp/telemetry"),
            rollover_max_bytes=2048,
            retention_max_total_bytes=1024,
        )
