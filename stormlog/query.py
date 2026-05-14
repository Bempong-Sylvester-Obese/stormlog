"""Local query API for Stormlog artifact directories and telemetry files."""

from __future__ import annotations

import builtins
import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .session import (
    SESSION_STATUS_INTERRUPTED,
    SessionSummary,
    infer_session_summary_from_events,
    session_summary_from_dict,
    session_summary_to_dict,
    stable_legacy_session_id,
)
from .telemetry import (
    LoadedTelemetrySession,
    TelemetryEvent,
    load_telemetry_sessions,
    telemetry_event_from_record,
    telemetry_event_to_dict,
)
from .telemetry_sink import (
    MANIFEST_FILENAME,
    SEGMENT_SUFFIX,
    TelemetrySinkManifest,
    read_telemetry_sink_manifest,
    resolve_telemetry_sink_segment_paths,
)

SourceKind = Literal[
    "sink",
    "telemetry_json",
    "telemetry_jsonl",
    "telemetry_csv",
    "diagnose_bundle",
    "oom_bundle",
]
SummaryMetric = Literal[
    "session_count_by_status",
    "peak_allocator_allocated_bytes",
    "peak_allocator_reserved_bytes",
    "peak_device_used_bytes",
    "alert_count",
    "collector_degradation_transitions",
    "interrupted_sessions_with_oom_bundles",
    "hidden_memory_gap_growth",
]
SummaryGroupBy = Literal["session", "session-rank", "rank", "status"]

_ALERT_EVENT_TYPES = frozenset({"warning", "critical", "error"})
_COLLECTOR_TRANSITION_TYPES = frozenset({"collector_degraded", "collector_recovered"})
_TELEMETRY_FILE_NAME_PARTS = ("event", "events", "track", "telemetry")


@dataclass(frozen=True)
class CatalogWarning:
    """Non-fatal discovery or loading warning."""

    path: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "message": self.message}


@dataclass(frozen=True)
class CatalogSource:
    """One discovered source of queryable artifact data."""

    path: Path
    source_kind: SourceKind
    event_paths: tuple[Path, ...] = ()
    manifest_path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "source_kind": self.source_kind,
            "event_paths": [str(path) for path in self.event_paths],
            "manifest_path": (
                str(self.manifest_path) if self.manifest_path is not None else None
            ),
        }


@dataclass(frozen=True)
class CatalogOOMBundle:
    """Manifest-backed OOM dump bundle discovered during cataloging."""

    bundle_path: Path
    manifest_path: Path
    created_at_utc: str | None
    backend: str | None
    reason: str | None
    event_count: int | None
    session_id: str | None
    session_status: str | None
    exception_type: str | None = None
    exception_module: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "bundle_path": str(self.bundle_path),
            "manifest_path": str(self.manifest_path),
            "created_at_utc": self.created_at_utc,
            "backend": self.backend,
            "reason": self.reason,
            "event_count": self.event_count,
            "session_id": self.session_id,
            "session_status": self.session_status,
            "exception_type": self.exception_type,
            "exception_module": self.exception_module,
        }


@dataclass(frozen=True)
class SessionFilter:
    """Filters for catalog/session rows."""

    session_id: str | None = None
    status: str | None = None
    job_id: str | None = None
    rank: int | None = None
    world_size: int | None = None
    has_oom_bundle: bool | None = None
    source_kind: str | None = None


@dataclass(frozen=True)
class EventFilter:
    """Filters for canonical telemetry event rows."""

    session_id: str | None = None
    event_type: str | None = None
    rank: int | None = None
    collector: str | None = None
    status: str | None = None
    time_start_ns: int | None = None
    time_end_ns: int | None = None
    has_alert: bool | None = None
    collector_health_status: str | None = None
    backend: str | None = None
    limit: int | None = None


@dataclass(frozen=True)
class OOMBundleFilter:
    """Filters for OOM bundle rows."""

    session_id: str | None = None
    backend: str | None = None
    reason: str | None = None
    created_after: str | None = None
    created_before: str | None = None


@dataclass(frozen=True)
class SessionRow:
    """Query row describing one loaded or manifest-backed session."""

    session_id: str
    status: str
    started_at_ns: int
    ended_at_ns: int | None
    host: str
    pid: int
    job_id: str | None
    rank: int
    local_rank: int
    world_size: int
    source: str
    source_path: str
    source_kind: str
    source_count: int
    warning_count: int
    event_count: int | None
    oom_bundle_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": self.status,
            "started_at_ns": self.started_at_ns,
            "ended_at_ns": self.ended_at_ns,
            "host": self.host,
            "pid": self.pid,
            "job_id": self.job_id,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "source": self.source,
            "source_path": self.source_path,
            "source_kind": self.source_kind,
            "source_count": self.source_count,
            "warning_count": self.warning_count,
            "event_count": self.event_count,
            "oom_bundle_count": self.oom_bundle_count,
        }


@dataclass(frozen=True)
class EventRow:
    """Query row wrapping a canonical telemetry event and provenance."""

    event: TelemetryEvent
    source_path: str
    source_kind: str
    session_status: str

    def as_dict(self) -> dict[str, Any]:
        row = telemetry_event_to_dict(self.event)
        row["source_path"] = self.source_path
        row["source_kind"] = self.source_kind
        row["session_status"] = self.session_status
        return row


@dataclass(frozen=True)
class OOMBundleRow:
    """Query row for one OOM bundle manifest."""

    bundle_path: str
    created_at_utc: str | None
    backend: str | None
    reason: str | None
    event_count: int | None
    session_id: str | None
    session_status: str | None
    exception_type: str | None
    exception_module: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "bundle_path": self.bundle_path,
            "created_at_utc": self.created_at_utc,
            "backend": self.backend,
            "reason": self.reason,
            "event_count": self.event_count,
            "session_id": self.session_id,
            "session_status": self.session_status,
            "exception_type": self.exception_type,
            "exception_module": self.exception_module,
        }


@dataclass(frozen=True)
class SummaryRow:
    """Built-in summary result row."""

    metric: str
    group_by: str
    value: int | float | str | None
    session_id: str | None = None
    rank: int | None = None
    status: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        row = {
            "metric": self.metric,
            "group_by": self.group_by,
            "session_id": self.session_id,
            "rank": self.rank,
            "status": self.status,
            "value": self.value,
        }
        row.update(dict(self.details))
        return row


class ArtifactCatalog:
    """Manifest-first catalog of local Stormlog artifacts."""

    def __init__(self, paths: Sequence[str | Path]) -> None:
        self.paths = tuple(Path(path) for path in paths)
        self.sources: list[CatalogSource] = []
        self.oom_bundles: list[CatalogOOMBundle] = []
        self.warnings: list[CatalogWarning] = []
        self._source_keys: set[tuple[Path, SourceKind]] = set()
        self._oom_bundle_paths: set[Path] = set()
        self._covered_event_paths: set[Path] = set()
        self._discover()

    def as_dict(self) -> dict[str, Any]:
        return {
            "paths": [str(path) for path in self.paths],
            "sources": [source.as_dict() for source in self.sources],
            "oom_bundles": [bundle.as_dict() for bundle in self.oom_bundles],
            "warnings": [warning.as_dict() for warning in self.warnings],
        }

    def _discover(self) -> None:
        for path in self.paths:
            self._discover_path(path)

    def _discover_path(self, path: Path) -> None:
        if not path.exists():
            self._warn(path, "path does not exist")
            return
        if path.is_file():
            self._discover_file(path)
            return
        if not path.is_dir():
            self._warn(path, "path is neither a file nor a directory")
            return
        self._discover_directory(path)

    def _discover_directory(self, directory: Path) -> None:
        manifest_path = directory / MANIFEST_FILENAME
        manifest_payload = _read_json_object(manifest_path)
        if manifest_payload is not None:
            if _is_oom_manifest(manifest_payload):
                self._add_oom_bundle(directory, manifest_path, manifest_payload)
                return
            if _is_diagnose_manifest(manifest_payload):
                self._add_source(
                    CatalogSource(
                        path=directory,
                        source_kind="diagnose_bundle",
                        manifest_path=manifest_path,
                    )
                )
            if _is_sink_manifest(manifest_payload):
                segment_paths = tuple(resolve_telemetry_sink_segment_paths(directory))
                self._covered_event_paths.update(segment_paths)
                self._add_source(
                    CatalogSource(
                        path=directory,
                        source_kind="sink",
                        event_paths=segment_paths,
                        manifest_path=manifest_path,
                    )
                )

        for nested_manifest in sorted(directory.rglob(MANIFEST_FILENAME)):
            if nested_manifest == manifest_path:
                continue
            payload = _read_json_object(nested_manifest)
            if payload is None:
                continue
            parent = nested_manifest.parent
            if _is_oom_manifest(payload):
                self._add_oom_bundle(parent, nested_manifest, payload)
            elif _is_diagnose_manifest(payload):
                self._add_source(
                    CatalogSource(
                        path=parent,
                        source_kind="diagnose_bundle",
                        manifest_path=nested_manifest,
                    )
                )
            elif _is_sink_manifest(payload):
                segment_paths = tuple(resolve_telemetry_sink_segment_paths(parent))
                self._covered_event_paths.update(segment_paths)
                self._add_source(
                    CatalogSource(
                        path=parent,
                        source_kind="sink",
                        event_paths=segment_paths,
                        manifest_path=nested_manifest,
                    )
                )

        for candidate in self._discover_candidate_files(directory):
            if candidate in self._covered_event_paths:
                continue
            self._discover_file(candidate)

    def _discover_file(self, path: Path) -> None:
        if path.name == MANIFEST_FILENAME:
            payload = _read_json_object(path)
            if payload is None:
                self._warn(path, "manifest is not a JSON object")
                return
            if _is_oom_manifest(payload):
                self._add_oom_bundle(path.parent, path, payload)
                return
            if _is_diagnose_manifest(payload):
                self._add_source(
                    CatalogSource(
                        path=path.parent,
                        source_kind="diagnose_bundle",
                        manifest_path=path,
                    )
                )
                return
            if _is_sink_manifest(payload):
                segment_paths = tuple(resolve_telemetry_sink_segment_paths(path))
                self._covered_event_paths.update(segment_paths)
                self._add_source(
                    CatalogSource(
                        path=path.parent,
                        source_kind="sink",
                        event_paths=segment_paths,
                        manifest_path=path,
                    )
                )
                return
            self._warn(path, "unrecognized manifest shape")
            return

        suffix = path.suffix.lower()
        if suffix == SEGMENT_SUFFIX:
            self._add_source(
                CatalogSource(
                    path=path,
                    source_kind="telemetry_jsonl",
                    event_paths=(path,),
                )
            )
        elif suffix == ".json":
            self._add_source(
                CatalogSource(
                    path=path,
                    source_kind="telemetry_json",
                    event_paths=(path,),
                )
            )
        elif suffix == ".csv":
            self._add_source(
                CatalogSource(
                    path=path,
                    source_kind="telemetry_csv",
                    event_paths=(path,),
                )
            )

    def _discover_candidate_files(self, directory: Path) -> list[Path]:
        candidates: set[Path] = set()
        for suffix in ("*.json", "*.jsonl", "*.csv"):
            for path in directory.rglob(suffix):
                if not path.is_file() or path.name == MANIFEST_FILENAME:
                    continue
                if self._is_inside_discovered_bundle(path):
                    continue
                if path.suffix.lower() == SEGMENT_SUFFIX:
                    candidates.add(path)
                    continue
                lowered = path.name.lower()
                if any(part in lowered for part in _TELEMETRY_FILE_NAME_PARTS):
                    candidates.add(path)
        return sorted(candidates)

    def _is_inside_discovered_bundle(self, path: Path) -> bool:
        return any(
            bundle_path in path.parents for bundle_path in self._oom_bundle_paths
        )

    def _add_source(self, source: CatalogSource) -> None:
        key = (source.path.resolve(), source.source_kind)
        if key in self._source_keys:
            return
        self._source_keys.add(key)
        self.sources.append(source)

    def _add_oom_bundle(
        self,
        bundle_path: Path,
        manifest_path: Path,
        payload: Mapping[str, Any],
    ) -> None:
        resolved = bundle_path.resolve()
        if resolved in self._oom_bundle_paths:
            return
        self._oom_bundle_paths.add(resolved)
        metadata = _read_json_object(bundle_path / "metadata.json") or {}
        self.oom_bundles.append(
            CatalogOOMBundle(
                bundle_path=bundle_path,
                manifest_path=manifest_path,
                created_at_utc=_string_or_none(payload.get("created_at_utc")),
                backend=_string_or_none(payload.get("backend")),
                reason=_string_or_none(payload.get("reason")),
                event_count=_int_or_none(payload.get("event_count")),
                session_id=_string_or_none(payload.get("session_id")),
                session_status=_string_or_none(payload.get("session_status")),
                exception_type=_string_or_none(metadata.get("exception_type")),
                exception_module=_string_or_none(metadata.get("exception_module")),
            )
        )

    def _warn(self, path: Path, message: str) -> None:
        self.warnings.append(CatalogWarning(path=str(path), message=message))


class QueryStore:
    """Reusable local query surface over Stormlog artifacts."""

    def __init__(self, catalog: ArtifactCatalog) -> None:
        self.catalog = catalog
        self._loaded_sessions_by_source: dict[
            tuple[Path, SourceKind], list[LoadedTelemetrySession]
        ] = {}

    def list_sessions(self, filters: SessionFilter | None = None) -> list[SessionRow]:
        """Return session rows from manifest metadata or loaded flat files."""
        filters = filters or SessionFilter()
        oom_counts = _count_oom_bundles_by_session(self.catalog.oom_bundles)
        rows: list[SessionRow] = []
        seen: set[tuple[str, str, str]] = set()

        for source in self.catalog.sources:
            for row in self._session_rows_for_source(source, oom_counts):
                key = (row.session_id, row.source_path, row.source_kind)
                if key in seen:
                    continue
                seen.add(key)
                if _session_matches(row, filters):
                    rows.append(row)

        rows.sort(key=lambda row: (row.started_at_ns, row.session_id), reverse=True)
        return rows

    def query_events(self, filters: EventFilter | None = None) -> list[EventRow]:
        """Return filtered canonical telemetry event rows."""
        filters = filters or EventFilter()
        rows: list[EventRow] = []

        for source in self.catalog.sources:
            if source.source_kind in {"diagnose_bundle", "oom_bundle"}:
                continue
            for loaded in self._load_sessions_for_source(source):
                summary = loaded.summary
                if filters.session_id is not None and (
                    summary.session_id != filters.session_id
                ):
                    continue
                if filters.status is not None and summary.status != filters.status:
                    continue
                for event in loaded.events:
                    row = EventRow(
                        event=event,
                        source_path=_event_source_path(loaded, source),
                        source_kind=source.source_kind,
                        session_status=summary.status,
                    )
                    if _event_matches(row, filters):
                        rows.append(row)

        rows.sort(key=lambda row: (row.event.timestamp_ns, row.event.session_id))
        if filters.limit is not None:
            return rows[: filters.limit]
        return rows

    def list_oom_bundles(
        self,
        filters: OOMBundleFilter | None = None,
    ) -> list[OOMBundleRow]:
        """Return filtered OOM bundle rows."""
        filters = filters or OOMBundleFilter()
        session_status_by_id = self._manifest_session_status_by_id()
        rows = [
            OOMBundleRow(
                bundle_path=str(bundle.bundle_path),
                created_at_utc=bundle.created_at_utc,
                backend=bundle.backend,
                reason=bundle.reason,
                event_count=bundle.event_count,
                session_id=bundle.session_id,
                session_status=(
                    bundle.session_status
                    or (
                        session_status_by_id.get(bundle.session_id)
                        if bundle.session_id is not None
                        else None
                    )
                ),
                exception_type=bundle.exception_type,
                exception_module=bundle.exception_module,
            )
            for bundle in self.catalog.oom_bundles
        ]
        rows = [row for row in rows if _oom_matches(row, filters)]
        rows.sort(key=lambda row: (row.created_at_utc or "", row.bundle_path))
        return rows

    def _manifest_session_status_by_id(self) -> dict[str, str]:
        statuses: dict[str, str] = {}
        for source in self.catalog.sources:
            if source.source_kind == "sink":
                manifest = read_telemetry_sink_manifest(source.path)
                if manifest is None:
                    continue
                for summary in manifest.sessions:
                    statuses.setdefault(summary.session_id, summary.status)
            elif source.source_kind == "diagnose_bundle":
                diagnose_summary = _diagnose_session_summary(source.manifest_path)
                if diagnose_summary is not None:
                    statuses.setdefault(
                        diagnose_summary.session_id,
                        diagnose_summary.status,
                    )
        return statuses

    def summarize(
        self,
        metric: SummaryMetric,
        *,
        group_by: SummaryGroupBy | None = None,
    ) -> list[SummaryRow]:
        """Run one built-in summary query."""
        if metric == "session_count_by_status":
            return self._summarize_session_count_by_status()
        if metric == "interrupted_sessions_with_oom_bundles":
            return self._summarize_interrupted_sessions_with_oom_bundles()

        resolved_group_by: SummaryGroupBy = group_by or "session"
        events = self.query_events(EventFilter())
        if metric in {
            "peak_allocator_allocated_bytes",
            "peak_allocator_reserved_bytes",
            "peak_device_used_bytes",
        }:
            field_name = {
                "peak_allocator_allocated_bytes": "allocator_allocated_bytes",
                "peak_allocator_reserved_bytes": "allocator_reserved_bytes",
                "peak_device_used_bytes": "device_used_bytes",
            }[metric]
            return _summarize_peak(events, metric, field_name, resolved_group_by)
        if metric == "alert_count":
            alert_events = [row for row in events if _is_alert_event(row.event)]
            return _summarize_count(alert_events, metric, resolved_group_by)
        if metric == "collector_degradation_transitions":
            transition_events = [
                row
                for row in events
                if row.event.event_type in _COLLECTOR_TRANSITION_TYPES
            ]
            return _summarize_count(transition_events, metric, resolved_group_by)
        if metric == "hidden_memory_gap_growth":
            return _summarize_hidden_memory_gap_growth(events, resolved_group_by)
        raise ValueError(f"unsupported summary metric: {metric}")

    def _session_rows_for_source(
        self,
        source: CatalogSource,
        oom_counts: Mapping[str, int],
    ) -> list[SessionRow]:
        if source.source_kind == "sink":
            manifest = read_telemetry_sink_manifest(source.path)
            if manifest is not None and manifest.sessions:
                return [
                    _session_row_from_manifest(
                        summary=summary,
                        manifest=manifest,
                        source=source,
                        oom_count=oom_counts.get(summary.session_id, 0),
                    )
                    for summary in manifest.sessions
                ]

        if source.source_kind == "diagnose_bundle":
            summary = _diagnose_session_summary(source.manifest_path)
            if summary is None:
                return []
            return [
                _session_row_from_summary(
                    summary=summary,
                    source=source,
                    source_count=1,
                    warning_count=0,
                    event_count=None,
                    oom_count=oom_counts.get(summary.session_id, 0),
                )
            ]

        rows: list[SessionRow] = []
        for loaded in self._load_sessions_for_source(source):
            rows.append(
                _session_row_from_summary(
                    summary=loaded.summary,
                    source=source,
                    source_count=len(loaded.sources_loaded) or 1,
                    warning_count=len(loaded.warnings),
                    event_count=len(loaded.events),
                    oom_count=oom_counts.get(loaded.summary.session_id, 0),
                )
            )
        return rows

    def _load_sessions_for_source(
        self,
        source: CatalogSource,
    ) -> list[LoadedTelemetrySession]:
        key = (source.path.resolve(), source.source_kind)
        cached = self._loaded_sessions_by_source.get(key)
        if cached is not None:
            return cached

        try:
            if source.source_kind == "telemetry_csv":
                loaded = _load_csv_sessions(source.path)
            elif source.source_kind in {"sink", "telemetry_json", "telemetry_jsonl"}:
                loaded = load_telemetry_sessions(source.path, permissive_legacy=True)
            else:
                loaded = []
        except Exception as exc:
            self.catalog.warnings.append(
                CatalogWarning(path=str(source.path), message=f"load failed: {exc}")
            )
            loaded = []

        self._loaded_sessions_by_source[key] = loaded
        return loaded

    def _summarize_session_count_by_status(self) -> list[SummaryRow]:
        counts: dict[str, int] = defaultdict(int)
        for row in self.list_sessions(SessionFilter()):
            counts[row.status] += 1
        return [
            SummaryRow(
                metric="session_count_by_status",
                group_by="status",
                status=status,
                value=count,
            )
            for status, count in sorted(counts.items())
        ]

    def _summarize_interrupted_sessions_with_oom_bundles(self) -> list[SummaryRow]:
        oom_counts = _count_oom_bundles_by_session(self.catalog.oom_bundles)
        rows: list[SummaryRow] = []
        for session in self.list_sessions(
            SessionFilter(
                status=SESSION_STATUS_INTERRUPTED,
                has_oom_bundle=True,
            )
        ):
            rows.append(
                SummaryRow(
                    metric="interrupted_sessions_with_oom_bundles",
                    group_by="session",
                    session_id=session.session_id,
                    status=session.status,
                    value=oom_counts.get(session.session_id, 0),
                )
            )
        return rows


def open(paths: Sequence[str | Path]) -> QueryStore:
    """Open one or more local artifact paths for in-process querying."""
    return QueryStore(ArtifactCatalog(paths))


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _is_sink_manifest(payload: Mapping[str, Any]) -> bool:
    fmt = payload.get("format")
    return fmt == "stormlog.append_only_telemetry_sink" or "segments" in payload


def _is_oom_manifest(payload: Mapping[str, Any]) -> bool:
    return (
        "bundle_name" in payload
        and "reason" in payload
        and "backend" in payload
        and "event_count" in payload
    )


def _is_diagnose_manifest(payload: Mapping[str, Any]) -> bool:
    return (
        "command_line" in payload
        and "files" in payload
        and "risk_detected" in payload
        and "session_id" in payload
    )


def _diagnose_session_summary(manifest_path: Path | None) -> SessionSummary | None:
    if manifest_path is None:
        return None
    payload = _read_json_object(manifest_path)
    if payload is None:
        return None
    session_payload = payload.get("session")
    if isinstance(session_payload, Mapping):
        try:
            return session_summary_from_dict(session_payload)
        except Exception:
            pass
    session_id = _string_or_none(payload.get("session_id"))
    if session_id is None:
        return None
    try:
        return session_summary_from_dict(
            {
                "session_id": session_id,
                "status": payload.get("session_status", "incomplete"),
                "started_at_ns": 0,
                "ended_at_ns": None,
                "host": "unknown",
                "pid": -1,
                "job_id": None,
                "rank": 0,
                "local_rank": 0,
                "world_size": 1,
                "source": "stormlog.diagnose",
            }
        )
    except Exception:
        return None


def _session_row_from_manifest(
    *,
    summary: SessionSummary,
    manifest: TelemetrySinkManifest,
    source: CatalogSource,
    oom_count: int,
) -> SessionRow:
    session_segments = [
        segment
        for segment in manifest.segments
        if segment.session_id == summary.session_id
    ]
    event_count = sum(segment.event_count for segment in session_segments)
    return _session_row_from_summary(
        summary=summary,
        source=source,
        source_count=len(session_segments),
        warning_count=0,
        event_count=event_count,
        oom_count=oom_count,
    )


def _session_row_from_summary(
    *,
    summary: SessionSummary,
    source: CatalogSource,
    source_count: int,
    warning_count: int,
    event_count: int | None,
    oom_count: int,
) -> SessionRow:
    payload = session_summary_to_dict(summary)
    return SessionRow(
        session_id=str(payload["session_id"]),
        status=str(payload["status"]),
        started_at_ns=int(payload["started_at_ns"]),
        ended_at_ns=(
            int(payload["ended_at_ns"])
            if payload.get("ended_at_ns") is not None
            else None
        ),
        host=str(payload["host"]),
        pid=int(payload["pid"]),
        job_id=_string_or_none(payload.get("job_id")),
        rank=int(payload["rank"]),
        local_rank=int(payload["local_rank"]),
        world_size=int(payload["world_size"]),
        source=str(payload["source"]),
        source_path=str(source.path),
        source_kind=source.source_kind,
        source_count=source_count,
        warning_count=warning_count,
        event_count=event_count,
        oom_bundle_count=oom_count,
    )


def _load_csv_sessions(path: Path) -> list[LoadedTelemetrySession]:
    warnings: list[str] = []
    events: list[TelemetryEvent] = []
    default_session_id = stable_legacy_session_id(str(path.resolve()), "csv")

    with builtins.open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, start=2):
            try:
                event = telemetry_event_from_record(
                    _normalize_csv_record(row),
                    permissive_legacy=True,
                    default_collector="legacy.csv",
                    default_sampling_interval_ms=0,
                    default_session_id=default_session_id,
                )
                events.append(event)
            except Exception as exc:
                warnings.append(f"CSV parse error {path}:{line_number}: {exc}")

    grouped: dict[str, list[TelemetryEvent]] = defaultdict(list)
    for event in events:
        grouped[event.session_id].append(event)

    sessions: list[LoadedTelemetrySession] = []
    for session_id, session_events in grouped.items():
        session_events.sort(key=lambda event: event.timestamp_ns)
        summary = infer_session_summary_from_events(
            session_id=session_id,
            events=session_events,
            source=f"artifact:{path.name or path.resolve()}",
        )
        sessions.append(
            LoadedTelemetrySession(
                summary=summary,
                events=session_events,
                sources_loaded=[str(path.resolve())],
                warnings=warnings,
            )
        )
    return sessions


def _normalize_csv_record(row: Mapping[str, str]) -> dict[str, Any]:
    int_fields = {
        "schema_version",
        "timestamp_ns",
        "sampling_interval_ms",
        "pid",
        "rank",
        "local_rank",
        "world_size",
        "device_id",
        "allocator_allocated_bytes",
        "allocator_reserved_bytes",
        "allocator_active_bytes",
        "allocator_inactive_bytes",
        "allocator_change_bytes",
        "device_used_bytes",
        "device_free_bytes",
        "device_total_bytes",
        "memory_allocated",
        "memory_reserved",
        "memory_change",
        "total_memory",
    }
    float_fields = {"timestamp"}
    normalized: dict[str, Any] = {}
    for key, raw_value in row.items():
        value: Any = raw_value.strip() if isinstance(raw_value, str) else raw_value
        if value == "":
            normalized[key] = None
        elif key == "metadata" and isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = {}
            normalized[key] = parsed if isinstance(parsed, dict) else {}
        elif key in int_fields:
            normalized[key] = int(float(value))
        elif key in float_fields:
            normalized[key] = float(value)
        else:
            normalized[key] = value
    return normalized


def _count_oom_bundles_by_session(
    bundles: Iterable[CatalogOOMBundle],
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for bundle in bundles:
        if bundle.session_id is not None:
            counts[bundle.session_id] += 1
    return counts


def _session_matches(row: SessionRow, filters: SessionFilter) -> bool:
    if filters.session_id is not None and row.session_id != filters.session_id:
        return False
    if filters.status is not None and row.status != filters.status:
        return False
    if filters.job_id is not None and row.job_id != filters.job_id:
        return False
    if filters.rank is not None and row.rank != filters.rank:
        return False
    if filters.world_size is not None and row.world_size != filters.world_size:
        return False
    if filters.has_oom_bundle is not None and (
        (row.oom_bundle_count > 0) != filters.has_oom_bundle
    ):
        return False
    if filters.source_kind is not None and row.source_kind != filters.source_kind:
        return False
    return True


def _event_matches(row: EventRow, filters: EventFilter) -> bool:
    event = row.event
    if filters.event_type is not None and event.event_type != filters.event_type:
        return False
    if filters.rank is not None and event.rank != filters.rank:
        return False
    if filters.collector is not None and event.collector != filters.collector:
        return False
    if filters.time_start_ns is not None and event.timestamp_ns < filters.time_start_ns:
        return False
    if filters.time_end_ns is not None and event.timestamp_ns > filters.time_end_ns:
        return False
    if filters.has_alert is not None and (_is_alert_event(event) != filters.has_alert):
        return False
    metadata = event.metadata
    if filters.collector_health_status is not None and (
        metadata.get("collector_health_status") != filters.collector_health_status
    ):
        return False
    if filters.backend is not None and metadata.get("backend") != filters.backend:
        return False
    return True


def _oom_matches(row: OOMBundleRow, filters: OOMBundleFilter) -> bool:
    if filters.session_id is not None and row.session_id != filters.session_id:
        return False
    if filters.backend is not None and row.backend != filters.backend:
        return False
    if filters.reason is not None and row.reason != filters.reason:
        return False
    created = _parse_datetime(row.created_at_utc)
    after = _parse_datetime(filters.created_after)
    before = _parse_datetime(filters.created_before)
    if after is not None and (created is None or created < after):
        return False
    if before is not None and (created is None or created > before):
        return False
    return True


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_alert_event(event: TelemetryEvent) -> bool:
    if event.event_type in _ALERT_EVENT_TYPES:
        return True
    severity = event.metadata.get("severity")
    return severity in {"warning", "critical", "error"}


def _event_source_path(loaded: LoadedTelemetrySession, source: CatalogSource) -> str:
    if loaded.sources_loaded:
        return loaded.sources_loaded[0]
    return str(source.path)


def _summary_group_key(
    row: EventRow,
    group_by: SummaryGroupBy,
) -> tuple[str | None, int | None, str | None]:
    if group_by == "rank":
        return None, row.event.rank, None
    if group_by == "session-rank":
        return row.event.session_id, row.event.rank, None
    if group_by == "status":
        return None, None, row.session_status
    return row.event.session_id, None, None


def _summarize_peak(
    rows: Sequence[EventRow],
    metric: str,
    field_name: str,
    group_by: SummaryGroupBy,
) -> list[SummaryRow]:
    best: dict[tuple[str | None, int | None, str | None], tuple[int, EventRow]] = {}
    for row in rows:
        value = int(getattr(row.event, field_name))
        key = _summary_group_key(row, group_by)
        existing = best.get(key)
        if existing is None or value > existing[0]:
            best[key] = (value, row)

    output: list[SummaryRow] = []
    for key, (value, event_row) in sorted(best.items(), key=lambda item: str(item[0])):
        session_id, rank, status = key
        output.append(
            SummaryRow(
                metric=metric,
                group_by=group_by,
                session_id=session_id,
                rank=rank,
                status=status,
                value=value,
                details={"timestamp_ns": event_row.event.timestamp_ns},
            )
        )
    return output


def _summarize_count(
    rows: Sequence[EventRow],
    metric: str,
    group_by: SummaryGroupBy,
) -> list[SummaryRow]:
    counts: dict[tuple[str | None, int | None, str | None], int] = defaultdict(int)
    for row in rows:
        counts[_summary_group_key(row, group_by)] += 1
    output: list[SummaryRow] = []
    for session_id, rank, status in sorted(counts, key=str):
        output.append(
            SummaryRow(
                metric=metric,
                group_by=group_by,
                session_id=session_id,
                rank=rank,
                status=status,
                value=counts[(session_id, rank, status)],
            )
        )
    return output


def _summarize_hidden_memory_gap_growth(
    rows: Sequence[EventRow],
    group_by: SummaryGroupBy,
) -> list[SummaryRow]:
    grouped: dict[tuple[str | None, int | None, str | None], list[EventRow]] = (
        defaultdict(list)
    )
    for row in rows:
        if row.event.event_type != "sample":
            continue
        grouped[_summary_group_key(row, group_by)].append(row)

    output: list[SummaryRow] = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        group_rows.sort(key=lambda row: row.event.timestamp_ns)
        gaps = [
            row.event.device_used_bytes - row.event.allocator_reserved_bytes
            for row in group_rows
        ]
        if not gaps:
            continue
        session_id, rank, status = key
        output.append(
            SummaryRow(
                metric="hidden_memory_gap_growth",
                group_by=group_by,
                session_id=session_id,
                rank=rank,
                status=status,
                value=gaps[-1] - gaps[0],
                details={
                    "first_gap_bytes": gaps[0],
                    "latest_gap_bytes": gaps[-1],
                    "peak_gap_bytes": max(gaps),
                    "sample_count": len(gaps),
                },
            )
        )
    return output


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, (int, float, str)) or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "ArtifactCatalog",
    "CatalogOOMBundle",
    "CatalogSource",
    "CatalogWarning",
    "EventFilter",
    "EventRow",
    "OOMBundleFilter",
    "OOMBundleRow",
    "QueryStore",
    "SessionFilter",
    "SessionRow",
    "SummaryRow",
    "open",
]
