"""Run envelope and attachment catalog helpers for local queries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from .correlation import ExternalAttachment
from .session import SessionSummary
from .telemetry_rollups import ROLLUP_FILENAME
from .telemetry_sink import TelemetrySinkManifest

AttachmentStorage = Literal["reference", "copy"]

RUN_ENVELOPE_FILENAME = "stormlog_run.json"
RUN_ENVELOPE_FORMAT = "stormlog.run_envelope"
RUN_ENVELOPE_SCHEMA_VERSION = 1


class SessionRowLike(Protocol):
    """Structural session row contract needed by run projection."""

    @property
    def session_id(self) -> str: ...

    @property
    def started_at_ns(self) -> int: ...

    @property
    def ended_at_ns(self) -> int | None: ...

    @property
    def job_id(self) -> str | None: ...

    @property
    def rank(self) -> int: ...

    @property
    def local_rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    @property
    def source_path(self) -> str: ...

    @property
    def source_kind(self) -> str: ...

    @property
    def event_count(self) -> int | None: ...


class CatalogSourceLike(Protocol):
    """Structural artifact source contract needed by run projection."""

    @property
    def path(self) -> Path: ...

    @property
    def source_kind(self) -> str: ...

    @property
    def manifest_path(self) -> Path | None: ...


class OOMBundleLike(Protocol):
    """Structural OOM bundle contract needed by run projection."""

    @property
    def bundle_path(self) -> Path: ...

    @property
    def created_at_utc(self) -> str | None: ...

    @property
    def session_id(self) -> str | None: ...

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of the bundle."""


@dataclass(frozen=True)
class CatalogRunSessionRef:
    """Session membership declared by a run envelope."""

    session_id: str
    job_id: str | None
    rank: int | None
    local_rank: int | None
    world_size: int | None
    role: str | None
    source_namespace: str | None
    source_ref: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "job_id": self.job_id,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "role": self.role,
            "source_namespace": self.source_namespace,
            "source_ref": self.source_ref,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CatalogRunAttachment:
    """Attachment declared by a run envelope."""

    title: str
    kind: str
    storage: AttachmentStorage
    attachment_id: str | None
    url: str | None
    path: str | None
    run_id: str | None
    session_id: str | None
    job_id: str | None
    rank: int | None
    local_rank: int | None
    world_size: int | None
    start_ns: int | None
    end_ns: int | None
    created_at_utc: str | None
    updated_at_utc: str | None
    source_namespace: str | None
    source_ref: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "kind": self.kind,
            "storage": self.storage,
            "attachment_id": self.attachment_id,
            "url": self.url,
            "path": self.path,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "job_id": self.job_id,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "source_namespace": self.source_namespace,
            "source_ref": self.source_ref,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CatalogRunEnvelope:
    """Manifest-backed top-level Stormlog run envelope."""

    run_id: str
    path: Path
    title: str | None
    description: str | None
    job_id: str | None
    started_at_ns: int | None
    ended_at_ns: int | None
    created_at_utc: str | None
    updated_at_utc: str | None
    source_namespace: str | None
    source_ref: str | None
    tags: tuple[str, ...]
    sessions: tuple[CatalogRunSessionRef, ...]
    attachments: tuple[CatalogRunAttachment, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "path": str(self.path),
            "title": self.title,
            "description": self.description,
            "job_id": self.job_id,
            "started_at_ns": self.started_at_ns,
            "ended_at_ns": self.ended_at_ns,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "source_namespace": self.source_namespace,
            "source_ref": self.source_ref,
            "tags": list(self.tags),
            "sessions": [session.as_dict() for session in self.sessions],
            "attachments": [attachment.as_dict() for attachment in self.attachments],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RunFilter:
    """Filters for top-level run envelope rows."""

    run_id: str | None = None
    session_id: str | None = None
    job_id: str | None = None
    rank: int | None = None
    source_namespace: str | None = None
    source_ref: str | None = None


@dataclass(frozen=True)
class RunAttachmentFilter:
    """Filters for run attachment catalog rows."""

    run_id: str | None = None
    session_id: str | None = None
    job_id: str | None = None
    rank: int | None = None
    kind: str | None = None
    source_namespace: str | None = None
    source_ref: str | None = None


@dataclass(frozen=True)
class RunRow:
    """Query row describing one explicit or synthesized run envelope."""

    run_id: str
    explicit: bool
    title: str | None
    description: str | None
    job_id: str | None
    started_at_ns: int | None
    ended_at_ns: int | None
    source_path: str
    source_kind: str
    source_namespace: str | None
    source_ref: str | None
    session_count: int
    attachment_count: int
    sessions: tuple[str, ...]
    ranks: tuple[int, ...]
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "explicit": self.explicit,
            "title": self.title,
            "description": self.description,
            "job_id": self.job_id,
            "started_at_ns": self.started_at_ns,
            "ended_at_ns": self.ended_at_ns,
            "source_path": self.source_path,
            "source_kind": self.source_kind,
            "source_namespace": self.source_namespace,
            "source_ref": self.source_ref,
            "session_count": self.session_count,
            "attachment_count": self.attachment_count,
            "sessions": list(self.sessions),
            "ranks": list(self.ranks),
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RunAttachmentRow:
    """Query row for local, distributed, or external run evidence."""

    run_id: str
    title: str
    kind: str
    storage: AttachmentStorage
    attachment_id: str | None
    url: str | None
    path: str | None
    session_id: str | None
    job_id: str | None
    rank: int | None
    local_rank: int | None
    world_size: int | None
    start_ns: int | None
    end_ns: int | None
    source_path: str
    source_kind: str
    source_namespace: str | None
    source_ref: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "title": self.title,
            "kind": self.kind,
            "storage": self.storage,
            "attachment_id": self.attachment_id,
            "url": self.url,
            "path": self.path,
            "session_id": self.session_id,
            "job_id": self.job_id,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "source_path": self.source_path,
            "source_kind": self.source_kind,
            "source_namespace": self.source_namespace,
            "source_ref": self.source_ref,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RunContext:
    """Internal normalized run context used to project rows."""

    run_id: str
    explicit: bool
    title: str | None
    description: str | None
    job_id: str | None
    started_at_ns: int | None
    ended_at_ns: int | None
    source_path: str
    source_kind: str
    source_namespace: str | None
    source_ref: str | None
    sessions: tuple[SessionRowLike, ...]
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_row(self, attachment_count: int) -> RunRow:
        ranks = sorted({session.rank for session in self.sessions})
        return RunRow(
            run_id=self.run_id,
            explicit=self.explicit,
            title=self.title,
            description=self.description,
            job_id=self.job_id,
            started_at_ns=self.started_at_ns,
            ended_at_ns=self.ended_at_ns,
            source_path=self.source_path,
            source_kind=self.source_kind,
            source_namespace=self.source_namespace,
            source_ref=self.source_ref,
            session_count=len(self.sessions),
            attachment_count=attachment_count,
            sessions=tuple(session.session_id for session in self.sessions),
            ranks=tuple(ranks),
            tags=self.tags,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class RunIdentityConflict:
    """Ambiguous identity mapping across multiple run contexts."""

    identity_kind: str
    identity_value: str
    run_ids: tuple[str, ...]

    @property
    def message(self) -> str:
        runs = ", ".join(self.run_ids)
        return f"ambiguous run {self.identity_kind} {self.identity_value!r}: {runs}"


@dataclass(frozen=True)
class RunIdentityIndex:
    """Resolved run identity indexes plus ambiguity diagnostics."""

    session_to_run: Mapping[str, str]
    job_to_run: Mapping[str, str]
    source_ref_to_run: Mapping[tuple[str, str], str]
    conflicts: tuple[RunIdentityConflict, ...]


def run_envelope_from_payload(
    payload: Mapping[str, Any],
    envelope_path: Path,
) -> CatalogRunEnvelope | None:
    """Parse a run envelope payload using the published v1 contract."""
    if not is_run_envelope(payload):
        return None
    run_id = _string_or_none(payload.get("run_id"))
    metadata = payload.get("metadata")
    if run_id is None or not isinstance(metadata, Mapping):
        return None

    sessions_payload = payload.get("sessions", ())
    attachments_payload = payload.get("attachments", ())
    if not _is_optional_array_field(sessions_payload):
        return None
    if not _is_optional_array_field(attachments_payload):
        return None

    sessions = _run_session_refs_from_payload(sessions_payload)
    if sessions is None:
        return None
    attachments = _run_attachments_from_payload(
        attachments_payload,
        envelope_path,
        run_id,
    )
    if attachments is None:
        return None
    tags_payload = payload.get("tags", ())
    if not _is_optional_array_field(tags_payload):
        return None
    tags = tuple(item for item in tags_payload if isinstance(item, str) and item)
    return CatalogRunEnvelope(
        run_id=run_id,
        path=envelope_path,
        title=_string_or_none(payload.get("title")),
        description=_string_or_none(payload.get("description")),
        job_id=_string_or_none(payload.get("job_id")),
        started_at_ns=_int_or_none(payload.get("started_at_ns")),
        ended_at_ns=_int_or_none(payload.get("ended_at_ns")),
        created_at_utc=_string_or_none(payload.get("created_at_utc")),
        updated_at_utc=_string_or_none(payload.get("updated_at_utc")),
        source_namespace=_string_or_none(payload.get("source_namespace")),
        source_ref=_string_or_none(payload.get("source_ref")),
        tags=tags,
        sessions=sessions,
        attachments=attachments,
        metadata=dict(metadata),
    )


def is_run_envelope(payload: Mapping[str, Any]) -> bool:
    """Return whether a payload advertises the run envelope v1 format."""
    return (
        payload.get("schema_version") == RUN_ENVELOPE_SCHEMA_VERSION
        and payload.get("format") == RUN_ENVELOPE_FORMAT
    )


def attachment_storage_or_none(value: Any) -> AttachmentStorage | None:
    """Parse an attachment storage value."""
    if value == "reference" or value == "copy":
        return cast(AttachmentStorage, value)
    return None


def attachment_storage_or_default(value: Any) -> AttachmentStorage:
    """Parse an attachment storage value, defaulting legacy sidecars."""
    return attachment_storage_or_none(value) or "reference"


def build_run_contexts(
    sessions: Sequence[SessionRowLike],
    envelopes: Sequence[CatalogRunEnvelope],
) -> dict[str, RunContext]:
    """Build explicit contexts plus implicit contexts for uncovered sessions."""
    contexts = _explicit_run_contexts(sessions, envelopes)
    covered_session_ids = {
        session.session_id
        for context in contexts.values()
        for session in context.sessions
    }
    uncovered_sessions = [
        session for session in sessions if session.session_id not in covered_session_ids
    ]
    contexts.update(
        _implicit_run_contexts(
            uncovered_sessions,
            existing_run_ids=set(contexts),
        )
    )
    return contexts


def build_identity_index(
    contexts: Mapping[str, RunContext],
) -> RunIdentityIndex:
    """Build unambiguous identity maps for attachment projection."""
    session_candidates: dict[str, set[str]] = defaultdict(set)
    job_candidates: dict[str, set[str]] = defaultdict(set)
    source_candidates: dict[tuple[str, str], set[str]] = defaultdict(set)

    for context in contexts.values():
        if context.job_id is not None:
            job_candidates[context.job_id].add(context.run_id)
        if context.source_namespace is not None and context.source_ref is not None:
            source_candidates[(context.source_namespace, context.source_ref)].add(
                context.run_id
            )
        for session in context.sessions:
            session_candidates[session.session_id].add(context.run_id)
            if session.job_id is not None:
                job_candidates[session.job_id].add(context.run_id)

    conflicts: list[RunIdentityConflict] = []
    session_to_run = _unique_identity_map("session_id", session_candidates, conflicts)
    job_to_run = _unique_identity_map("job_id", job_candidates, conflicts)
    source_ref_to_run = _unique_source_ref_map(source_candidates, conflicts)
    return RunIdentityIndex(
        session_to_run=session_to_run,
        job_to_run=job_to_run,
        source_ref_to_run=source_ref_to_run,
        conflicts=tuple(conflicts),
    )


def envelope_attachment_rows(
    envelopes: Sequence[CatalogRunEnvelope],
    contexts: Mapping[str, RunContext],
) -> list[RunAttachmentRow]:
    """Project explicit envelope attachments into run attachment rows."""
    rows: list[RunAttachmentRow] = []
    for envelope in envelopes:
        if envelope.run_id not in contexts:
            continue
        for attachment in envelope.attachments:
            run_id = attachment.run_id or envelope.run_id
            rows.append(
                RunAttachmentRow(
                    run_id=run_id,
                    title=attachment.title,
                    kind=attachment.kind,
                    storage=attachment.storage,
                    attachment_id=attachment.attachment_id,
                    url=attachment.url,
                    path=attachment.path,
                    session_id=attachment.session_id,
                    job_id=attachment.job_id or envelope.job_id,
                    rank=attachment.rank,
                    local_rank=attachment.local_rank,
                    world_size=attachment.world_size,
                    start_ns=attachment.start_ns,
                    end_ns=attachment.end_ns,
                    source_path=attachment.url or attachment.path or str(envelope.path),
                    source_kind="run_envelope_attachment",
                    source_namespace=attachment.source_namespace
                    or envelope.source_namespace,
                    source_ref=attachment.source_ref or envelope.source_ref,
                    metadata=attachment.metadata,
                )
            )
    return rows


def sidecar_attachment_rows(
    attachments: Sequence[ExternalAttachment],
    identity_index: RunIdentityIndex,
) -> list[RunAttachmentRow]:
    """Project external sidecar attachments into run attachment rows."""
    rows: list[RunAttachmentRow] = []
    for attachment in attachments:
        run_id = run_id_for_identity(
            run_id=attachment.run_id,
            session_id=attachment.session_id,
            job_id=attachment.job_id,
            source_namespace=attachment.source_namespace,
            source_ref=attachment.source_ref,
            identity_index=identity_index,
        )
        if run_id is None:
            continue
        rows.append(
            RunAttachmentRow(
                run_id=run_id,
                title=attachment.title,
                kind=attachment.kind,
                storage=attachment_storage_or_default(attachment.storage),
                attachment_id=attachment.attachment_id,
                url=attachment.url,
                path=attachment.path,
                session_id=attachment.session_id,
                job_id=attachment.job_id,
                rank=attachment.rank,
                local_rank=None,
                world_size=None,
                start_ns=attachment.start_ns,
                end_ns=attachment.end_ns,
                source_path=attachment.url
                or attachment.path
                or attachment.sidecar_path,
                source_kind="attachment_sidecar",
                source_namespace=attachment.source_namespace,
                source_ref=attachment.source_ref,
                metadata=attachment.metadata,
            )
        )
    return rows


def sink_attachment_rows(
    source: CatalogSourceLike,
    manifest: TelemetrySinkManifest,
    identity_index: RunIdentityIndex,
) -> list[RunAttachmentRow]:
    """Project sink directory and segment rows for all mapped sessions."""
    rows: list[RunAttachmentRow] = []
    for summary in manifest.sessions:
        run_id = run_id_for_identity(
            run_id=None,
            session_id=summary.session_id,
            job_id=summary.job_id,
            source_namespace=None,
            source_ref=None,
            identity_index=identity_index,
        )
        if run_id is None:
            continue
        rows.append(
            local_attachment_row(
                run_id=run_id,
                title="Telemetry sink",
                kind="telemetry_sink",
                path=str(source.path),
                session_id=summary.session_id,
                job_id=summary.job_id,
                rank=summary.rank,
                local_rank=summary.local_rank,
                world_size=summary.world_size,
                start_ns=summary.started_at_ns,
                end_ns=summary.ended_at_ns,
                source_kind="sink",
                metadata={
                    "manifest_path": (
                        str(source.manifest_path)
                        if source.manifest_path is not None
                        else None
                    ),
                },
            )
        )
    rows.extend(sink_segment_attachment_rows(source, manifest, identity_index))
    if (source.path / ROLLUP_FILENAME).exists():
        rows.extend(rollup_attachment_rows(source, manifest, identity_index))
    return rows


def sink_segment_attachment_rows(
    source: CatalogSourceLike,
    manifest: TelemetrySinkManifest,
    identity_index: RunIdentityIndex,
) -> list[RunAttachmentRow]:
    """Project sink segment files into run attachment rows."""
    rows: list[RunAttachmentRow] = []
    session_by_id = {summary.session_id: summary for summary in manifest.sessions}
    for segment in manifest.segments:
        if segment.session_id is None:
            continue
        summary = session_by_id.get(segment.session_id)
        run_id = run_id_for_identity(
            run_id=None,
            session_id=segment.session_id,
            job_id=summary.job_id if summary is not None else None,
            source_namespace=None,
            source_ref=None,
            identity_index=identity_index,
        )
        if run_id is None:
            continue
        rows.append(
            local_attachment_row(
                run_id=run_id,
                title=segment.filename,
                kind="telemetry_sink_segment",
                path=str(source.path / segment.filename),
                session_id=segment.session_id,
                job_id=summary.job_id if summary is not None else None,
                rank=summary.rank if summary is not None else None,
                local_rank=summary.local_rank if summary is not None else None,
                world_size=summary.world_size if summary is not None else None,
                start_ns=summary.started_at_ns if summary is not None else None,
                end_ns=summary.ended_at_ns if summary is not None else None,
                source_kind="sink_segment",
                metadata={
                    "event_count": segment.event_count,
                    "size_bytes": segment.size_bytes,
                    "closed": segment.closed,
                },
            )
        )
    return rows


def rollup_attachment_rows(
    source: CatalogSourceLike,
    manifest: TelemetrySinkManifest,
    identity_index: RunIdentityIndex,
) -> list[RunAttachmentRow]:
    """Project sink rollup sidecars into run attachment rows."""
    rows: list[RunAttachmentRow] = []
    rollup_path = source.path / ROLLUP_FILENAME
    for summary in manifest.sessions:
        run_id = run_id_for_identity(
            run_id=None,
            session_id=summary.session_id,
            job_id=summary.job_id,
            source_namespace=None,
            source_ref=None,
            identity_index=identity_index,
        )
        if run_id is None:
            continue
        rows.append(
            local_attachment_row(
                run_id=run_id,
                title="Telemetry rollups",
                kind="telemetry_rollup",
                path=str(rollup_path),
                session_id=summary.session_id,
                job_id=summary.job_id,
                rank=summary.rank,
                local_rank=summary.local_rank,
                world_size=summary.world_size,
                start_ns=summary.started_at_ns,
                end_ns=summary.ended_at_ns,
                source_kind="rollup",
                metadata={
                    "manifest_session_count": len(manifest.sessions),
                    "manifest_segment_count": len(manifest.segments),
                },
            )
        )
    return rows


def diagnose_attachment_rows(
    source: CatalogSourceLike,
    summary: SessionSummary | None,
    identity_index: RunIdentityIndex,
) -> list[RunAttachmentRow]:
    """Project a diagnose bundle into a run attachment row."""
    if summary is None:
        return []
    run_id = run_id_for_identity(
        run_id=None,
        session_id=summary.session_id,
        job_id=summary.job_id,
        source_namespace=None,
        source_ref=None,
        identity_index=identity_index,
    )
    if run_id is None:
        return []
    return [
        local_attachment_row(
            run_id=run_id,
            title="Diagnose bundle",
            kind="diagnose_bundle",
            path=str(source.path),
            session_id=summary.session_id,
            job_id=summary.job_id,
            rank=summary.rank,
            local_rank=summary.local_rank,
            world_size=summary.world_size,
            start_ns=summary.started_at_ns,
            end_ns=summary.ended_at_ns,
            source_kind="diagnose_bundle",
            metadata={
                "manifest_path": (
                    str(source.manifest_path)
                    if source.manifest_path is not None
                    else None
                ),
                "session_status": summary.status,
            },
        )
    ]


def flat_telemetry_attachment_rows(
    source: CatalogSourceLike,
    contexts: Mapping[str, RunContext],
) -> list[RunAttachmentRow]:
    """Project flat telemetry files into run attachment rows."""
    identity_index = build_identity_index(contexts)
    rows: list[RunAttachmentRow] = []
    for context in contexts.values():
        for session in context.sessions:
            if session.source_path != str(source.path):
                continue
            run_id = run_id_for_identity(
                run_id=None,
                session_id=session.session_id,
                job_id=session.job_id,
                source_namespace=None,
                source_ref=None,
                identity_index=identity_index,
            )
            if run_id is None:
                continue
            rows.append(
                local_attachment_row(
                    run_id=run_id,
                    title=Path(session.source_path).name or "Telemetry file",
                    kind="telemetry_file",
                    path=session.source_path,
                    session_id=session.session_id,
                    job_id=session.job_id,
                    rank=session.rank,
                    local_rank=session.local_rank,
                    world_size=session.world_size,
                    start_ns=session.started_at_ns,
                    end_ns=session.ended_at_ns,
                    source_kind=session.source_kind,
                    metadata={"event_count": session.event_count},
                )
            )
    return rows


def oom_attachment_rows(
    bundles: Sequence[OOMBundleLike],
    identity_index: RunIdentityIndex,
    session_by_id: Mapping[str, SessionRowLike],
) -> list[RunAttachmentRow]:
    """Project OOM bundles into run attachment rows with session metadata."""
    rows: list[RunAttachmentRow] = []
    for bundle in bundles:
        if bundle.session_id is None:
            continue
        session = session_by_id.get(bundle.session_id)
        run_id = run_id_for_identity(
            run_id=None,
            session_id=bundle.session_id,
            job_id=session.job_id if session is not None else None,
            source_namespace=None,
            source_ref=None,
            identity_index=identity_index,
        )
        if run_id is None:
            continue
        seen_ns = _datetime_to_ns(_parse_datetime(bundle.created_at_utc))
        rows.append(
            local_attachment_row(
                run_id=run_id,
                title="OOM bundle",
                kind="oom_bundle",
                path=str(bundle.bundle_path),
                session_id=bundle.session_id,
                job_id=session.job_id if session is not None else None,
                rank=session.rank if session is not None else None,
                local_rank=session.local_rank if session is not None else None,
                world_size=session.world_size if session is not None else None,
                start_ns=seen_ns,
                end_ns=seen_ns,
                source_kind="oom_bundle",
                metadata=bundle.as_dict(),
            )
        )
    return rows


def local_attachment_row(
    *,
    run_id: str,
    title: str,
    kind: str,
    path: str,
    session_id: str | None,
    job_id: str | None,
    rank: int | None,
    local_rank: int | None,
    world_size: int | None,
    start_ns: int | None,
    end_ns: int | None,
    source_kind: str,
    metadata: Mapping[str, Any],
) -> RunAttachmentRow:
    """Build a copied local artifact attachment row."""
    return RunAttachmentRow(
        run_id=run_id,
        title=title,
        kind=kind,
        storage="copy",
        attachment_id=None,
        url=None,
        path=path,
        session_id=session_id,
        job_id=job_id,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        start_ns=start_ns,
        end_ns=end_ns,
        source_path=path,
        source_kind=source_kind,
        source_namespace="stormlog",
        source_ref=None,
        metadata=metadata,
    )


def run_id_for_identity(
    *,
    run_id: str | None,
    session_id: str | None,
    job_id: str | None,
    source_namespace: str | None,
    source_ref: str | None,
    identity_index: RunIdentityIndex,
) -> str | None:
    """Resolve an attachment run id without using ambiguous identities."""
    if run_id is not None:
        return run_id
    if source_namespace is not None and source_ref is not None:
        resolved = identity_index.source_ref_to_run.get((source_namespace, source_ref))
        if resolved is not None:
            return resolved
    if session_id is not None:
        resolved = identity_index.session_to_run.get(session_id)
        if resolved is not None:
            return resolved
    if job_id is not None:
        return identity_index.job_to_run.get(job_id)
    return None


def run_matches(row: RunRow, filters: RunFilter) -> bool:
    """Return whether a run row satisfies filters."""
    if filters.run_id is not None and row.run_id != filters.run_id:
        return False
    if filters.session_id is not None and filters.session_id not in row.sessions:
        return False
    if filters.job_id is not None and row.job_id != filters.job_id:
        return False
    if filters.rank is not None and filters.rank not in row.ranks:
        return False
    if (
        filters.source_namespace is not None
        and row.source_namespace != filters.source_namespace
    ):
        return False
    if filters.source_ref is not None and row.source_ref != filters.source_ref:
        return False
    return True


def run_attachment_matches(
    row: RunAttachmentRow,
    filters: RunAttachmentFilter,
) -> bool:
    """Return whether an attachment row satisfies filters."""
    if filters.run_id is not None and row.run_id != filters.run_id:
        return False
    if filters.session_id is not None and row.session_id != filters.session_id:
        return False
    if filters.job_id is not None and row.job_id != filters.job_id:
        return False
    if filters.rank is not None and row.rank != filters.rank:
        return False
    if filters.kind is not None and row.kind != filters.kind:
        return False
    if (
        filters.source_namespace is not None
        and row.source_namespace != filters.source_namespace
    ):
        return False
    if filters.source_ref is not None and row.source_ref != filters.source_ref:
        return False
    return True


def _explicit_run_contexts(
    sessions: Sequence[SessionRowLike],
    envelopes: Sequence[CatalogRunEnvelope],
) -> dict[str, RunContext]:
    session_by_id = _first_session_by_id(sessions)
    contexts: dict[str, RunContext] = {}
    for envelope in envelopes:
        member_ids = {session.session_id for session in envelope.sessions}
        members = [
            session_by_id[session_id]
            for session_id in member_ids
            if session_id in session_by_id
        ]
        if not members and envelope.job_id is not None:
            members = [
                session for session in sessions if session.job_id == envelope.job_id
            ]
        members.sort(key=lambda session: (session.started_at_ns, session.session_id))
        contexts[envelope.run_id] = RunContext(
            run_id=envelope.run_id,
            explicit=True,
            title=envelope.title,
            description=envelope.description,
            job_id=envelope.job_id or _common_job_id(members),
            started_at_ns=(
                envelope.started_at_ns
                if envelope.started_at_ns is not None
                else _min_started_at_ns(members)
            ),
            ended_at_ns=(
                envelope.ended_at_ns
                if envelope.ended_at_ns is not None
                else _max_ended_at_ns(members)
            ),
            source_path=str(envelope.path),
            source_kind="run_envelope",
            source_namespace=envelope.source_namespace,
            source_ref=envelope.source_ref,
            sessions=tuple(members),
            tags=envelope.tags,
            metadata=envelope.metadata,
        )
    return contexts


def _implicit_run_contexts(
    sessions: Sequence[SessionRowLike],
    *,
    existing_run_ids: set[str],
) -> dict[str, RunContext]:
    grouped: dict[str, list[SessionRowLike]] = defaultdict(list)
    for session in sessions:
        if session.job_id is not None:
            grouped[f"job:{session.job_id}"].append(session)
        else:
            grouped[f"session:{session.session_id}"].append(session)

    contexts: dict[str, RunContext] = {}
    used_run_ids = set(existing_run_ids)
    for base_run_id, members in grouped.items():
        run_id = _unique_run_id(base_run_id, used_run_ids)
        used_run_ids.add(run_id)
        members.sort(key=lambda session: (session.started_at_ns, session.session_id))
        job_id = _common_job_id(members)
        contexts[run_id] = RunContext(
            run_id=run_id,
            explicit=False,
            title=(
                f"Distributed job {job_id}"
                if job_id is not None
                else f"Session {members[0].session_id}"
            ),
            description=None,
            job_id=job_id,
            started_at_ns=_min_started_at_ns(members),
            ended_at_ns=_max_ended_at_ns(members),
            source_path=members[0].source_path if members else "",
            source_kind="implicit_run",
            source_namespace=None,
            source_ref=None,
            sessions=tuple(members),
        )
    return contexts


def _run_session_ref_from_payload(
    payload: Mapping[str, Any],
) -> CatalogRunSessionRef | None:
    session_id = _string_or_none(payload.get("session_id"))
    metadata = payload.get("metadata")
    if session_id is None or not isinstance(metadata, Mapping):
        return None
    return CatalogRunSessionRef(
        session_id=session_id,
        job_id=_string_or_none(payload.get("job_id")),
        rank=_int_or_none(payload.get("rank")),
        local_rank=_int_or_none(payload.get("local_rank")),
        world_size=_int_or_none(payload.get("world_size")),
        role=_string_or_none(payload.get("role")),
        source_namespace=_string_or_none(payload.get("source_namespace")),
        source_ref=_string_or_none(payload.get("source_ref")),
        metadata=dict(metadata),
    )


def _run_session_refs_from_payload(
    payload: Sequence[Any],
) -> tuple[CatalogRunSessionRef, ...] | None:
    refs: list[CatalogRunSessionRef] = []
    for item in payload:
        if not isinstance(item, Mapping):
            return None
        ref = _run_session_ref_from_payload(item)
        if ref is None:
            return None
        refs.append(ref)
    return tuple(refs)


def _run_attachment_from_payload(
    payload: Mapping[str, Any],
    envelope_path: Path,
    envelope_run_id: str,
) -> CatalogRunAttachment | None:
    title = _string_or_none(payload.get("title"))
    kind = _string_or_none(payload.get("kind"))
    storage = attachment_storage_or_none(payload.get("storage"))
    if title is None or kind is None or storage is None:
        return None
    url = _string_or_none(payload.get("url"))
    raw_path = _string_or_none(payload.get("path"))
    if url is None and raw_path is None:
        return None
    resolved_path = _resolve_optional_path(raw_path, envelope_path)
    start_ns = _int_or_none(payload.get("start_ns"))
    end_ns = _int_or_none(payload.get("end_ns"))
    if start_ns is not None and end_ns is not None and end_ns < start_ns:
        return None
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    return CatalogRunAttachment(
        title=title,
        kind=kind,
        storage=storage,
        attachment_id=_string_or_none(payload.get("attachment_id")),
        url=url,
        path=resolved_path,
        run_id=_string_or_none(payload.get("run_id")) or envelope_run_id,
        session_id=_string_or_none(payload.get("session_id")),
        job_id=_string_or_none(payload.get("job_id")),
        rank=_int_or_none(payload.get("rank")),
        local_rank=_int_or_none(payload.get("local_rank")),
        world_size=_int_or_none(payload.get("world_size")),
        start_ns=start_ns,
        end_ns=end_ns,
        created_at_utc=_string_or_none(payload.get("created_at_utc")),
        updated_at_utc=_string_or_none(payload.get("updated_at_utc")),
        source_namespace=_string_or_none(payload.get("source_namespace")),
        source_ref=_string_or_none(payload.get("source_ref")),
        metadata=dict(metadata),
    )


def _run_attachments_from_payload(
    payload: Sequence[Any],
    envelope_path: Path,
    envelope_run_id: str,
) -> tuple[CatalogRunAttachment, ...] | None:
    attachments: list[CatalogRunAttachment] = []
    for item in payload:
        if not isinstance(item, Mapping):
            return None
        attachment = _run_attachment_from_payload(
            item,
            envelope_path,
            envelope_run_id,
        )
        if attachment is None:
            return None
        attachments.append(attachment)
    return tuple(attachments)


def _is_optional_array_field(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str)


def _unique_identity_map(
    identity_kind: str,
    candidates: Mapping[str, set[str]],
    conflicts: list[RunIdentityConflict],
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for identity_value, run_ids in candidates.items():
        if len(run_ids) == 1:
            resolved[identity_value] = next(iter(run_ids))
            continue
        conflicts.append(
            RunIdentityConflict(
                identity_kind=identity_kind,
                identity_value=identity_value,
                run_ids=tuple(sorted(run_ids)),
            )
        )
    return resolved


def _unique_source_ref_map(
    candidates: Mapping[tuple[str, str], set[str]],
    conflicts: list[RunIdentityConflict],
) -> dict[tuple[str, str], str]:
    resolved: dict[tuple[str, str], str] = {}
    for source_ref, run_ids in candidates.items():
        if len(run_ids) == 1:
            resolved[source_ref] = next(iter(run_ids))
            continue
        conflicts.append(
            RunIdentityConflict(
                identity_kind="source_ref",
                identity_value=f"{source_ref[0]}:{source_ref[1]}",
                run_ids=tuple(sorted(run_ids)),
            )
        )
    return resolved


def _unique_run_id(base_run_id: str, used_run_ids: set[str]) -> str:
    if base_run_id not in used_run_ids:
        return base_run_id
    candidate = f"implicit:{base_run_id}"
    if candidate not in used_run_ids:
        return candidate
    suffix = 2
    while f"{candidate}:{suffix}" in used_run_ids:
        suffix += 1
    return f"{candidate}:{suffix}"


def _first_session_by_id(
    sessions: Sequence[SessionRowLike],
) -> dict[str, SessionRowLike]:
    rows: dict[str, SessionRowLike] = {}
    for session in sessions:
        rows.setdefault(session.session_id, session)
    return rows


def _common_job_id(sessions: Sequence[SessionRowLike]) -> str | None:
    job_ids = {session.job_id for session in sessions if session.job_id is not None}
    return next(iter(job_ids)) if len(job_ids) == 1 else None


def _min_started_at_ns(sessions: Sequence[SessionRowLike]) -> int | None:
    if not sessions:
        return None
    return min(session.started_at_ns for session in sessions)


def _max_ended_at_ns(sessions: Sequence[SessionRowLike]) -> int | None:
    if not sessions or any(session.ended_at_ns is None for session in sessions):
        return None
    return max(cast(int, session.ended_at_ns) for session in sessions)


def _resolve_optional_path(raw_path: str | None, source_path: Path) -> str | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = source_path.parent / path
    return str(path.resolve())


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


def _datetime_to_ns(value: datetime | None) -> int | None:
    if value is None:
        return None
    return int(value.timestamp() * 1_000_000_000)


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
    "AttachmentStorage",
    "CatalogRunAttachment",
    "CatalogRunEnvelope",
    "CatalogRunSessionRef",
    "RUN_ENVELOPE_FILENAME",
    "RUN_ENVELOPE_FORMAT",
    "RUN_ENVELOPE_SCHEMA_VERSION",
    "RunAttachmentFilter",
    "RunAttachmentRow",
    "RunContext",
    "RunFilter",
    "RunIdentityConflict",
    "RunIdentityIndex",
    "RunRow",
    "attachment_storage_or_default",
    "attachment_storage_or_none",
    "build_identity_index",
    "build_run_contexts",
    "diagnose_attachment_rows",
    "envelope_attachment_rows",
    "flat_telemetry_attachment_rows",
    "is_run_envelope",
    "local_attachment_row",
    "oom_attachment_rows",
    "rollup_attachment_rows",
    "run_attachment_matches",
    "run_envelope_from_payload",
    "run_id_for_identity",
    "run_matches",
    "sidecar_attachment_rows",
    "sink_attachment_rows",
    "sink_segment_attachment_rows",
]
