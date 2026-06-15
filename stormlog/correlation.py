"""Derived correlation rows for local Stormlog artifact investigations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

ATTACHMENTS_FILENAME = "stormlog_attachments.json"
ATTACHMENTS_FORMAT = "stormlog.attachments"
ATTACHMENTS_SCHEMA_VERSION = 1
CLOCK_DOMAIN_UNIX_EPOCH_NS = "unix_epoch_ns"
CLOCK_NORMALIZATION_PRODUCER_EPOCH_NS = "producer_emitted_epoch_ns"
DEFAULT_CORRELATION_WINDOW_NS = 60_000_000_000

CorrelationScope = Literal["session", "distributed"]


@dataclass(frozen=True)
class CorrelationFilter:
    """Filters and anchor selection for a correlation query."""

    session_id: str | None = None
    job_id: str | None = None
    rank: int | None = None
    scope: CorrelationScope = "session"
    at_ns: int | None = None
    record_id: str | None = None
    window_ns: int = DEFAULT_CORRELATION_WINDOW_NS
    kinds: tuple[str, ...] = ()
    limit: int | None = None

    def __post_init__(self) -> None:
        if self.scope not in {"session", "distributed"}:
            raise ValueError("scope must be 'session' or 'distributed'")
        if self.window_ns < 0:
            raise ValueError("window_ns must be >= 0")
        if self.limit is not None and self.limit < 0:
            raise ValueError("limit must be >= 0")


@dataclass(frozen=True)
class CorrelationEvidence:
    """One piece of evidence related to a correlation anchor."""

    evidence_id: str
    kind: str
    title: str
    session_id: str | None
    job_id: str | None
    rank: int | None
    world_size: int | None
    start_ns: int | None
    end_ns: int | None
    source_path: str
    source_kind: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    confidence: str = "low"
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize the evidence row to a JSON-safe dictionary."""
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "title": self.title,
            "session_id": self.session_id,
            "job_id": self.job_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "source_path": self.source_path,
            "source_kind": self.source_kind,
            "metadata": dict(self.metadata),
            "confidence": self.confidence,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class CorrelationResult:
    """Correlation query output anchored to a timestamp or telemetry record."""

    anchor: Mapping[str, Any]
    evidence: Sequence[CorrelationEvidence] = ()
    warnings: Sequence[str] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize the result to a JSON-safe dictionary."""
        return {
            "anchor": dict(self.anchor),
            "evidence": [row.as_dict() for row in self.evidence],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class AttachmentFilter:
    """Filters for external attachment sidecar rows."""

    session_id: str | None = None
    job_id: str | None = None
    rank: int | None = None
    kind: str | None = None


@dataclass(frozen=True)
class ExternalAttachment:
    """External evidence discovered from a local attachment sidecar."""

    title: str
    kind: str
    attachment_id: str | None
    url: str | None
    path: str | None
    session_id: str | None
    job_id: str | None
    rank: int | None
    start_ns: int | None
    end_ns: int | None
    created_at_utc: str | None
    updated_at_utc: str | None
    metadata: Mapping[str, Any]
    sidecar_path: str

    def as_dict(self) -> dict[str, Any]:
        """Serialize the attachment row to a JSON-safe dictionary."""
        return {
            "title": self.title,
            "kind": self.kind,
            "attachment_id": self.attachment_id,
            "url": self.url,
            "path": self.path,
            "session_id": self.session_id,
            "job_id": self.job_id,
            "rank": self.rank,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "metadata": dict(self.metadata),
            "sidecar_path": self.sidecar_path,
        }


__all__ = [
    "ATTACHMENTS_FILENAME",
    "ATTACHMENTS_FORMAT",
    "ATTACHMENTS_SCHEMA_VERSION",
    "CLOCK_DOMAIN_UNIX_EPOCH_NS",
    "CLOCK_NORMALIZATION_PRODUCER_EPOCH_NS",
    "DEFAULT_CORRELATION_WINDOW_NS",
    "AttachmentFilter",
    "CorrelationEvidence",
    "CorrelationFilter",
    "CorrelationResult",
    "CorrelationScope",
    "ExternalAttachment",
]
