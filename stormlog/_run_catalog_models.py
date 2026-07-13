"""Internal data contracts for run envelope and attachment catalog queries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

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


__all__ = [
    "AttachmentStorage",
    "CatalogRunAttachment",
    "CatalogRunEnvelope",
    "CatalogRunSessionRef",
    "CatalogSourceLike",
    "OOMBundleLike",
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
    "SessionRowLike",
]
