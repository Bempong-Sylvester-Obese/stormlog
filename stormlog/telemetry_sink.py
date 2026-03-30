"""Append-only telemetry sink with rollover and retention bounds."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, TextIO

MANIFEST_FILENAME = "manifest.json"
SEGMENT_PREFIX = "segment-"
SEGMENT_SUFFIX = ".jsonl"
SINK_SCHEMA_VERSION = 1


@dataclass
class TelemetrySinkConfig:
    """Runtime policy for append-only telemetry persistence."""

    root_dir: Path
    flush_every_events: int = 50
    flush_every_seconds: float = 2.0
    rollover_max_bytes: int = 64 * 1024 * 1024
    rollover_max_events: int = 10000
    retention_max_files: int = 8
    retention_max_total_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        if self.flush_every_events <= 0:
            raise ValueError("flush_every_events must be >= 1")
        if self.flush_every_seconds <= 0:
            raise ValueError("flush_every_seconds must be > 0")
        if self.rollover_max_bytes <= 0:
            raise ValueError("rollover_max_bytes must be >= 1")
        if self.rollover_max_events <= 0:
            raise ValueError("rollover_max_events must be >= 1")
        if self.retention_max_files <= 0:
            raise ValueError("retention_max_files must be >= 1")
        if self.retention_max_total_bytes <= 0:
            raise ValueError("retention_max_total_bytes must be >= 1")
        if self.retention_max_total_bytes < self.rollover_max_bytes:
            raise ValueError("retention_max_total_bytes must be >= rollover_max_bytes")


@dataclass
class _SegmentState:
    filename: str
    event_count: int
    size_bytes: int
    closed: bool


class AppendOnlyTelemetrySink:
    """Write telemetry records to newline-delimited JSON segments."""

    def __init__(self, config: TelemetrySinkConfig) -> None:
        self.config = config
        self.root_dir = config.root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root_dir / MANIFEST_FILENAME
        self._segments: list[_SegmentState] = []
        self._next_segment_index = 1
        self._buffer: list[str] = []
        self._buffered_event_count = 0
        self._handle: TextIO | None = None
        self._lock = threading.Lock()
        self._last_flush_monotonic = time.monotonic()
        self._closed = False
        self._load_existing_state()

    def append(self, record: Mapping[str, Any]) -> None:
        with self._lock:
            self._closed = False
            self._buffer.append(json.dumps(dict(record), sort_keys=True) + "\n")
            self._buffered_event_count += 1
            self._flush_locked(force=False)

    def flush(self, force: bool = False) -> None:
        with self._lock:
            self._flush_locked(force=force)

    def close(self) -> None:
        with self._lock:
            self._flush_locked(force=True)
            if self._handle is not None:
                self._handle.close()
                self._handle = None
            current = self._current_segment()
            if current is not None and not current.closed:
                current.closed = True
                self._write_manifest_locked()
            self._closed = True

    def _flush_locked(self, force: bool) -> None:
        if not self._buffer:
            return

        now = time.monotonic()
        if not force:
            if self._buffered_event_count < self.config.flush_every_events and (
                now - self._last_flush_monotonic < self.config.flush_every_seconds
            ):
                return

        current = self._ensure_current_segment_locked()
        payload = "".join(self._buffer)
        payload_bytes = payload.encode("utf-8")
        handle = self._ensure_handle_locked(current)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())

        current.event_count += self._buffered_event_count
        current.size_bytes += len(payload_bytes)
        self._buffer.clear()
        self._buffered_event_count = 0
        self._last_flush_monotonic = now

        self._rollover_locked(current)
        self._prune_retention_locked()
        self._write_manifest_locked()

    def _rollover_locked(self, current: _SegmentState) -> None:
        if (
            current.event_count < self.config.rollover_max_events
            and current.size_bytes < self.config.rollover_max_bytes
        ):
            return
        current.closed = True
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _prune_retention_locked(self) -> None:
        while True:
            total_bytes = sum(segment.size_bytes for segment in self._segments)
            over_file_limit = len(self._segments) > self.config.retention_max_files
            over_size_limit = total_bytes > self.config.retention_max_total_bytes
            if not over_file_limit and not over_size_limit:
                return

            removable = next(
                (segment for segment in self._segments if segment.closed), None
            )
            if removable is None:
                return

            path = self.root_dir / removable.filename
            if path.exists():
                path.unlink()
            self._segments.remove(removable)

    def _current_segment(self) -> _SegmentState | None:
        if not self._segments:
            return None
        current = self._segments[-1]
        if current.closed:
            return None
        return current

    def _ensure_current_segment_locked(self) -> _SegmentState:
        current = self._current_segment()
        if current is not None:
            return current

        segment = _SegmentState(
            filename=f"{SEGMENT_PREFIX}{self._next_segment_index:06d}{SEGMENT_SUFFIX}",
            event_count=0,
            size_bytes=0,
            closed=False,
        )
        self._next_segment_index += 1
        self._segments.append(segment)
        return segment

    def _ensure_handle_locked(self, current: _SegmentState) -> TextIO:
        if self._handle is None:
            segment_path = self.root_dir / current.filename
            self._recover_segment_tail_locked(segment_path, current)
            self._handle = segment_path.open("a", encoding="utf-8")
        return self._handle

    def _load_existing_state(self) -> None:
        if self._manifest_path.exists():
            payload = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            self._segments = [
                _SegmentState(**dict(segment))
                for segment in payload.get("segments", [])
                if isinstance(segment, dict)
            ]
            self._next_segment_index = self._compute_next_segment_index()
            return

        discovered = sorted(self.root_dir.glob(f"{SEGMENT_PREFIX}*{SEGMENT_SUFFIX}"))
        if not discovered:
            return

        for index, path in enumerate(discovered):
            self._segments.append(
                _SegmentState(
                    filename=path.name,
                    event_count=self._count_records(path),
                    size_bytes=path.stat().st_size,
                    closed=index < len(discovered) - 1,
                )
            )
        self._next_segment_index = self._compute_next_segment_index()
        self._write_manifest_locked()

    def _compute_next_segment_index(self) -> int:
        max_index = 0
        for segment in self._segments:
            stem = Path(segment.filename).stem
            if not stem.startswith(SEGMENT_PREFIX):
                continue
            suffix = stem[len(SEGMENT_PREFIX) :]
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))
        return max_index + 1

    def _write_manifest_locked(self) -> None:
        payload = {
            "schema_version": SINK_SCHEMA_VERSION,
            "format": "stormlog.append_only_telemetry_sink",
            "segments": [asdict(segment) for segment in self._segments],
        }
        temp_path = self._manifest_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temp_path.replace(self._manifest_path)

    @staticmethod
    def _count_records(path: Path) -> int:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def _recover_segment_tail_locked(
        self,
        segment_path: Path,
        current: _SegmentState,
    ) -> None:
        if not segment_path.exists():
            current.event_count = 0
            current.size_bytes = 0
            return

        payload = segment_path.read_bytes()
        if payload and not payload.endswith(b"\n"):
            last_newline = payload.rfind(b"\n")
            payload = payload[: last_newline + 1] if last_newline >= 0 else b""
            segment_path.write_bytes(payload)

        current.size_bytes = len(payload)
        current.event_count = self._count_records(segment_path)


def resolve_telemetry_sink_segment_paths(path: str | Path) -> list[Path]:
    """Resolve append-only sink inputs to ordered JSONL segment paths."""
    resolved_path = Path(path)
    if resolved_path.is_file():
        if resolved_path.suffix == SEGMENT_SUFFIX:
            return [resolved_path]
        if resolved_path.name == MANIFEST_FILENAME:
            return _merge_segment_paths(
                _segment_paths_from_manifest(resolved_path),
                _discover_segment_paths(resolved_path.parent),
            )
        return []

    if not resolved_path.is_dir():
        return []

    manifest_path = resolved_path / MANIFEST_FILENAME
    manifest_segments = _segment_paths_from_manifest(manifest_path)
    if manifest_segments:
        return _merge_segment_paths(
            manifest_segments,
            _discover_segment_paths(resolved_path),
        )
    return _discover_segment_paths(resolved_path)


def _discover_segment_paths(root_dir: Path) -> list[Path]:
    return sorted(root_dir.glob(f"{SEGMENT_PREFIX}*{SEGMENT_SUFFIX}"))


def _merge_segment_paths(
    manifest_segments: list[Path],
    discovered_segments: list[Path],
) -> list[Path]:
    merged_by_name = {path.name: path for path in discovered_segments}
    for path in manifest_segments:
        merged_by_name[path.name] = path
    return [merged_by_name[name] for name in sorted(merged_by_name)]


def _segment_paths_from_manifest(manifest_path: Path) -> list[Path]:
    if not manifest_path.exists():
        return []

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    root_dir = manifest_path.parent
    segments = []
    for segment in payload.get("segments", []):
        if not isinstance(segment, dict):
            continue
        filename = segment.get("filename")
        if isinstance(filename, str):
            candidate = root_dir / filename
            if candidate.exists():
                segments.append(candidate)
    return segments


__all__ = [
    "AppendOnlyTelemetrySink",
    "MANIFEST_FILENAME",
    "SEGMENT_PREFIX",
    "SEGMENT_SUFFIX",
    "SINK_SCHEMA_VERSION",
    "TelemetrySinkConfig",
    "resolve_telemetry_sink_segment_paths",
]
