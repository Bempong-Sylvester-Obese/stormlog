"""Shared session identity and lifecycle helpers."""

from __future__ import annotations

import os
import socket
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Mapping, Protocol, TypeVar
from uuid import NAMESPACE_URL, uuid4, uuid5

SESSION_STATUS_RUNNING = "running"
SESSION_STATUS_COMPLETED = "completed"
SESSION_STATUS_INTERRUPTED = "interrupted"
SESSION_STATUS_INCOMPLETE = "incomplete"

SESSION_STATUSES = frozenset(
    {
        SESSION_STATUS_RUNNING,
        SESSION_STATUS_COMPLETED,
        SESSION_STATUS_INTERRUPTED,
        SESSION_STATUS_INCOMPLETE,
    }
)

_SESSION_SELECTION_PRIORITY = {
    SESSION_STATUS_COMPLETED: 0,
    SESSION_STATUS_INTERRUPTED: 1,
    SESSION_STATUS_INCOMPLETE: 2,
    SESSION_STATUS_RUNNING: 3,
}


@dataclass(frozen=True)
class SessionSummary:
    """Identity and lifecycle metadata for one capture session."""

    session_id: str
    status: str
    started_at_ns: int
    ended_at_ns: int | None
    host: str
    pid: int
    job_id: str | None = None
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    source: str = "stormlog.session"


class _HasSummary(Protocol):
    summary: SessionSummary


_SummaryT = TypeVar("_SummaryT", bound="_HasSummary")


def now_ns() -> int:
    """Return the current wall-clock timestamp in nanoseconds."""
    return time.time_ns()


def new_session_id() -> str:
    """Create a new opaque session identifier."""
    return str(uuid4())


def stable_legacy_session_id(*parts: object) -> str:
    """Build a deterministic synthetic session id for legacy artifacts."""
    payload = "::".join(str(part) for part in parts if part is not None)
    return f"legacy-{uuid5(NAMESPACE_URL, payload)}"


def create_session_summary(
    *,
    source: str,
    status: str = SESSION_STATUS_RUNNING,
    session_id: str | None = None,
    started_at_ns: int | None = None,
    ended_at_ns: int | None = None,
    host: str | None = None,
    pid: int | None = None,
    job_id: str | None = None,
    rank: int = 0,
    local_rank: int = 0,
    world_size: int = 1,
) -> SessionSummary:
    """Create a normalized session summary."""
    resolved_status = normalize_session_status(status)
    resolved_started = int(now_ns() if started_at_ns is None else started_at_ns)
    resolved_host = host or socket.gethostname()
    resolved_pid = int(os.getpid() if pid is None else pid)
    resolved_ended = None if ended_at_ns is None else int(ended_at_ns)
    return SessionSummary(
        session_id=session_id or new_session_id(),
        status=resolved_status,
        started_at_ns=resolved_started,
        ended_at_ns=resolved_ended,
        host=resolved_host,
        pid=resolved_pid,
        job_id=job_id,
        rank=int(rank),
        local_rank=int(local_rank),
        world_size=max(1, int(world_size)),
        source=source,
    )


def update_session_summary(
    summary: SessionSummary,
    *,
    status: str | None = None,
    ended_at_ns: int | None = None,
    source: str | None = None,
) -> SessionSummary:
    """Return a copy of a session summary with lifecycle updates applied."""
    updates: dict[str, Any] = {}
    if status is not None:
        updates["status"] = normalize_session_status(status)
    if ended_at_ns is not None:
        updates["ended_at_ns"] = int(ended_at_ns)
    if source is not None:
        updates["source"] = source
    return replace(summary, **updates)


def normalize_session_status(value: object) -> str:
    """Validate and normalize a session status string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("session status must be a non-empty string")
    normalized = value.strip().lower()
    if normalized not in SESSION_STATUSES:
        raise ValueError(f"Unsupported session status: {value}")
    return normalized


def session_summary_to_dict(summary: SessionSummary) -> dict[str, Any]:
    """Serialize a session summary into a plain JSON-safe mapping."""
    return asdict(summary)


def session_summary_from_dict(payload: Mapping[str, Any]) -> SessionSummary:
    """Deserialize a session summary mapping."""
    if not isinstance(payload, Mapping):
        raise ValueError("session summary payload must be an object")
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")
    source = payload.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a non-empty string")
    status = normalize_session_status(payload.get("status"))
    started_at_raw = payload.get("started_at_ns")
    if started_at_raw is None:
        raise ValueError("started_at_ns must be present")
    started_at_ns = int(started_at_raw)
    ended_raw = payload.get("ended_at_ns")
    ended_at_ns = None if ended_raw is None else int(ended_raw)
    host = payload.get("host")
    if not isinstance(host, str) or not host.strip():
        raise ValueError("host must be a non-empty string")
    return SessionSummary(
        session_id=session_id,
        status=status,
        started_at_ns=started_at_ns,
        ended_at_ns=ended_at_ns,
        host=host,
        pid=int(payload.get("pid", -1)),
        job_id=payload.get("job_id"),
        rank=int(payload.get("rank", 0)),
        local_rank=int(payload.get("local_rank", 0)),
        world_size=max(1, int(payload.get("world_size", 1))),
        source=source,
    )


def infer_session_summary_from_events(
    *,
    session_id: str,
    events: Iterable[Any],
    source: str,
    fallback_status: str = SESSION_STATUS_INCOMPLETE,
) -> SessionSummary:
    """Infer a session summary from event timestamps and lifecycle markers."""
    event_list = list(events)
    if event_list:
        sorted_events = sorted(
            event_list,
            key=lambda event: int(getattr(event, "timestamp_ns")),
        )
        first = sorted_events[0]
        ended_at_ns: int | None = None
        status = normalize_session_status(fallback_status)
        if any(getattr(event, "event_type", "") == "stop" for event in sorted_events):
            status = SESSION_STATUS_COMPLETED
            stop_events = [
                event
                for event in sorted_events
                if getattr(event, "event_type", "") == "stop"
            ]
            ended_at_ns = int(getattr(stop_events[-1], "timestamp_ns"))
        return create_session_summary(
            source=source,
            status=status,
            session_id=session_id,
            started_at_ns=int(getattr(first, "timestamp_ns")),
            ended_at_ns=ended_at_ns,
            host=str(getattr(first, "host", "unknown")),
            pid=int(getattr(first, "pid", -1)),
            job_id=getattr(first, "job_id", None),
            rank=int(getattr(first, "rank", 0)),
            local_rank=int(getattr(first, "local_rank", 0)),
            world_size=max(1, int(getattr(first, "world_size", 1))),
        )

    return create_session_summary(
        source=source,
        status=fallback_status,
        session_id=session_id,
    )


def sort_session_summaries(summaries: Iterable[SessionSummary]) -> list[SessionSummary]:
    """Sort sessions by selection priority and recency."""
    return sorted(
        summaries,
        key=lambda summary: (
            _SESSION_SELECTION_PRIORITY.get(summary.status, 99),
            -int(summary.ended_at_ns or summary.started_at_ns),
            -summary.started_at_ns,
            summary.session_id,
        ),
    )


def select_default_session(
    sessions: Iterable[_SummaryT],
) -> _SummaryT | None:
    """Pick the default session using lifecycle priority and recency."""
    session_list = list(sessions)
    if not session_list:
        return None
    ordered = sorted(
        session_list,
        key=lambda loaded: (
            _SESSION_SELECTION_PRIORITY.get(loaded.summary.status, 99),
            -int(loaded.summary.ended_at_ns or loaded.summary.started_at_ns),
            -loaded.summary.started_at_ns,
            loaded.summary.session_id,
        ),
    )
    return ordered[0]


__all__ = [
    "SESSION_STATUS_COMPLETED",
    "SESSION_STATUS_INCOMPLETE",
    "SESSION_STATUS_INTERRUPTED",
    "SESSION_STATUS_RUNNING",
    "SESSION_STATUSES",
    "SessionSummary",
    "create_session_summary",
    "infer_session_summary_from_events",
    "new_session_id",
    "normalize_session_status",
    "now_ns",
    "select_default_session",
    "session_summary_from_dict",
    "session_summary_to_dict",
    "sort_session_summaries",
    "stable_legacy_session_id",
    "update_session_summary",
]
