"""Durable issue fingerprints and grouped issue row models."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

IssueKind = Literal[
    "oom",
    "collector_degradation",
    "alert",
    "hidden_memory_anomaly",
]
IssueState = Literal["open", "resolved", "ignored", "regressed"]

ISSUE_FINGERPRINT_SCHEMA_VERSION = 1
ISSUE_STATE_OPEN: IssueState = "open"

_ISSUE_KINDS = frozenset(
    {
        "oom",
        "collector_degradation",
        "alert",
        "hidden_memory_anomaly",
    }
)
_ISSUE_STATES = frozenset({"open", "resolved", "ignored", "regressed"})
_RE_NUMERIC_TOKEN = re.compile(r"(?<!\w)\d+(?:\.\d+)?%?(?!\w)")
_RE_HEX_TOKEN = re.compile(r"0x[0-9a-fA-F]+")
_RE_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class IssueFingerprint:
    """Stable grouping identity for recurring Stormlog issues."""

    kind: IssueKind
    dimensions: Mapping[str, Any]
    schema_version: int = ISSUE_FINGERPRINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.kind not in _ISSUE_KINDS:
            raise ValueError(f"unsupported issue kind: {self.kind}")
        if self.schema_version < 1:
            raise ValueError("schema_version must be >= 1")
        object.__setattr__(
            self,
            "dimensions",
            freeze_dimensions(normalize_dimensions(self.dimensions)),
        )

    @property
    def fingerprint_id(self) -> str:
        """Return a deterministic hash for this fingerprint."""

        payload = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "dimensions": json_safe_dimensions(self.dimensions),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return f"issue:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:24]}"

    def as_dict(self) -> dict[str, Any]:
        """Serialize the fingerprint into a JSON-safe mapping."""

        return {
            "schema_version": self.schema_version,
            "fingerprint_id": self.fingerprint_id,
            "kind": self.kind,
            "dimensions": json_safe_dimensions(self.dimensions),
        }


@dataclass(frozen=True)
class IssueEvidenceLink:
    """Link from a grouped issue back to raw session, event, or bundle evidence."""

    session_id: str | None = None
    timestamp_ns: int | None = None
    rank: int | None = None
    source_path: str | None = None
    source_kind: str | None = None
    event_type: str | None = None
    bundle_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        """Serialize the evidence link into a JSON-safe mapping."""

        return {
            "session_id": self.session_id,
            "timestamp_ns": self.timestamp_ns,
            "rank": self.rank,
            "source_path": self.source_path,
            "source_kind": self.source_kind,
            "event_type": self.event_type,
            "bundle_path": self.bundle_path,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class StormlogIssue:
    """Grouped issue suitable for CLI/TUI presentation and future persistence."""

    fingerprint: IssueFingerprint
    title: str
    severity: str
    hit_count: int
    first_seen_ns: int | None
    last_seen_ns: int | None
    affected_sessions: Sequence[str]
    representative_evidence: IssueEvidenceLink
    evidence: Sequence[IssueEvidenceLink]
    state: IssueState = ISSUE_STATE_OPEN
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", normalize_issue_state(self.state))
        if self.hit_count < 1:
            raise ValueError("hit_count must be >= 1")
        object.__setattr__(
            self,
            "affected_sessions",
            tuple(sorted({session for session in self.affected_sessions if session})),
        )
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "details", dict(self.details))

    @property
    def fingerprint_id(self) -> str:
        """Return the issue fingerprint id."""

        return self.fingerprint.fingerprint_id

    @property
    def kind(self) -> IssueKind:
        """Return the issue kind."""

        return self.fingerprint.kind

    def as_dict(self) -> dict[str, Any]:
        """Serialize the grouped issue into a JSON-safe mapping."""

        return {
            "fingerprint_id": self.fingerprint_id,
            "fingerprint": self.fingerprint.as_dict(),
            "kind": self.kind,
            "state": self.state,
            "severity": self.severity,
            "title": self.title,
            "hit_count": self.hit_count,
            "first_seen_ns": self.first_seen_ns,
            "last_seen_ns": self.last_seen_ns,
            "affected_sessions": list(self.affected_sessions),
            "representative_evidence": self.representative_evidence.as_dict(),
            "evidence": [link.as_dict() for link in self.evidence],
            "details": dict(self.details),
        }


def normalize_issue_state(state: str) -> IssueState:
    """Validate and normalize an issue state string."""

    normalized = state.strip().lower()
    if normalized not in _ISSUE_STATES:
        raise ValueError(f"unsupported issue state: {state}")
    return normalized  # type: ignore[return-value]


def normalize_dimensions(dimensions: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize fingerprint dimensions into a canonical JSON-safe mapping."""

    normalized: dict[str, Any] = {}
    for key, value in dimensions.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("fingerprint dimension keys must be non-empty strings")
        normalized[key] = _normalize_dimension_value(value)
    return dict(sorted(normalized.items()))


def freeze_dimensions(dimensions: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a recursively immutable canonical dimension mapping."""

    frozen = {
        key: _freeze_dimension_value(value)
        for key, value in sorted(dimensions.items(), key=lambda item: item[0])
    }
    return MappingProxyType(frozen)


def json_safe_dimensions(dimensions: Mapping[str, Any]) -> dict[str, Any]:
    """Return a mutable JSON-safe copy of canonical dimension material."""

    return {
        key: _json_safe_dimension_value(value)
        for key, value in sorted(dimensions.items(), key=lambda item: item[0])
    }


def normalize_text_dimension(value: object, *, default: str = "unknown") -> str:
    """Return a low-cardinality text token for fingerprint dimensions."""

    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    text = _RE_HEX_TOKEN.sub("<hex>", text)
    text = _RE_NUMERIC_TOKEN.sub("<num>", text)
    text = _RE_WHITESPACE.sub(" ", text)
    return text


def categorize_alert_context(context: object) -> str:
    """Return a stable alert category from event context text."""

    normalized = normalize_text_dimension(context)
    if "fragmentation" in normalized:
        return "high_fragmentation"
    if "oom" in normalized or "out of memory" in normalized:
        return "oom_alert"
    if normalized == "unknown":
        return "generic_alert"
    return normalized


def normalized_error_stem(error: object) -> str:
    """Return a stable low-cardinality error stem."""

    text = normalize_text_dimension(error)
    if text == "unknown":
        return text
    separators = (":", "\n")
    for separator in separators:
        if separator in text:
            text = text.split(separator, 1)[0].strip()
    return text[:120] or "unknown"


def _normalize_dimension_value(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, str):
        return normalize_text_dimension(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        normalized_items = [_normalize_dimension_value(item) for item in value]
        return sorted(
            normalized_items, key=lambda item: json.dumps(item, sort_keys=True)
        )
    if isinstance(value, Mapping):
        return normalize_dimensions(value)
    return normalize_text_dimension(value)


def _freeze_dimension_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return freeze_dimensions(value)
    if isinstance(value, list):
        return tuple(_freeze_dimension_value(item) for item in value)
    return value


def _json_safe_dimension_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return json_safe_dimensions(value)
    if isinstance(value, tuple):
        return [_json_safe_dimension_value(item) for item in value]
    return value


__all__ = [
    "ISSUE_FINGERPRINT_SCHEMA_VERSION",
    "ISSUE_STATE_OPEN",
    "IssueEvidenceLink",
    "IssueFingerprint",
    "IssueKind",
    "IssueState",
    "StormlogIssue",
    "categorize_alert_context",
    "freeze_dimensions",
    "json_safe_dimensions",
    "normalize_dimensions",
    "normalize_issue_state",
    "normalize_text_dimension",
    "normalized_error_stem",
]
