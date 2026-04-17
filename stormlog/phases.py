"""Structured phase telemetry helpers for long-running trackers and analysis."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

PHASE_ENTER_EVENT = "phase_enter"
PHASE_EXIT_EVENT = "phase_exit"
PHASE_SCOPE_METADATA_KEY = "phase_scope"
PHASE_SCOPE_ATTRIBUTES_KEY = "attributes"


def format_phase_path(path: Sequence[str]) -> str:
    """Return a human-readable phase path label."""
    return " / ".join(part for part in path if part)


def is_phase_boundary_event(event: Any) -> bool:
    """Return ``True`` when the event is a structured phase boundary."""
    event_type = _event_field(event, "event_type", "")
    if event_type not in {PHASE_ENTER_EVENT, PHASE_EXIT_EVENT}:
        return False
    return extract_phase_scope(event) is not None


@dataclass(frozen=True)
class PhaseAttribution:
    """Resolved workload phase attribution for an anomaly or report item."""

    phase_resolution: str
    phase_path: str | None = None
    phase_paths: list[str] = field(default_factory=list)
    scope_id: str | None = None
    thread_id: int | None = None
    thread_name: str | None = None


class PhaseHandle:
    """A closeable tracker phase handle returned by ``enter_phase()``."""

    def __init__(
        self,
        *,
        scope_id: str,
        name: str,
        path: tuple[str, ...],
        close_callback: Callable[[], None],
    ) -> None:
        self.scope_id = scope_id
        self.name = name
        self.path = path
        self._close_callback = close_callback
        self._closed = False

    @property
    def phase_path(self) -> str:
        """Return the formatted phase path."""
        return format_phase_path(self.path)

    @property
    def closed(self) -> bool:
        """Return ``True`` once the handle has been closed."""
        return self._closed

    def close(self) -> Any:
        """Close the phase handle once."""
        if self._closed:
            return None
        result = self._close_callback()
        self._closed = True
        return result

    def __enter__(self) -> "PhaseHandle":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


@dataclass(frozen=True)
class _PhaseBoundary:
    event_type: str
    context: str
    metadata: dict[str, Any]
    scope_id: str
    path: tuple[str, ...]


@dataclass(frozen=True)
class _ActivePhase:
    session_id: str
    rank: int
    thread_id: int
    thread_name: str
    scope_id: str
    parent_scope_id: str | None
    name: str
    path: tuple[str, ...]
    sequence: int
    attributes: dict[str, Any]


@dataclass(frozen=True)
class _PhaseInterval:
    session_id: str
    rank: int
    thread_id: int
    thread_name: str
    scope_id: str
    path: tuple[str, ...]
    start_ns: int
    end_ns: int
    sequence: int
    synthetic_end: bool = False

    @property
    def depth(self) -> int:
        return len(self.path)


class TrackerPhaseState:
    """Per-tracker phase nesting state and boundary payload generation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sequence = 0
        self._active_by_thread: dict[tuple[str, int, int], list[_ActivePhase]] = {}

    def reset(self) -> None:
        """Drop all active phase scopes for a new tracker session."""
        with self._lock:
            self._sequence = 0
            self._active_by_thread.clear()

    def enter_phase(
        self,
        *,
        session_id: str,
        rank: int,
        name: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> _PhaseBoundary:
        """Register one nested phase enter transition."""
        normalized_name = _normalize_phase_name(name)
        thread = threading.current_thread()
        thread_id = int(thread.ident or 0)
        thread_name = thread.name or f"thread-{thread_id}"
        attributes = dict(metadata or {})

        with self._lock:
            key = (session_id, rank, thread_id)
            stack = self._active_by_thread.setdefault(key, [])
            parent = stack[-1] if stack else None
            self._sequence += 1
            scope_id = f"{session_id}:{self._sequence}"
            path = (
                (*parent.path, normalized_name)
                if parent is not None
                else (normalized_name,)
            )
            active = _ActivePhase(
                session_id=session_id,
                rank=rank,
                thread_id=thread_id,
                thread_name=thread_name,
                scope_id=scope_id,
                parent_scope_id=parent.scope_id if parent is not None else None,
                name=normalized_name,
                path=path,
                sequence=self._sequence,
                attributes=attributes,
            )
            stack.append(active)

        boundary = _PhaseBoundary(
            event_type=PHASE_ENTER_EVENT,
            context=f"Phase entered: {format_phase_path(path)}",
            metadata={
                PHASE_SCOPE_METADATA_KEY: _phase_scope_payload(active, action="enter")
            },
            scope_id=scope_id,
            path=path,
        )
        return boundary

    def exit_phase(
        self,
        *,
        session_id: str,
        rank: int,
        scope_id: str,
        thread_id: int,
    ) -> _PhaseBoundary | None:
        """Register one nested phase exit transition."""
        with self._lock:
            key = (session_id, rank, thread_id)
            stack = self._active_by_thread.get(key)
            if not stack:
                return None

            active = stack[-1]
            if active.scope_id != scope_id:
                raise RuntimeError(
                    "Phase handles must be closed in strict LIFO order per thread."
                )

            stack.pop()
            if not stack:
                self._active_by_thread.pop(key, None)
            self._sequence += 1
            closed = _ActivePhase(
                session_id=active.session_id,
                rank=active.rank,
                thread_id=active.thread_id,
                thread_name=active.thread_name,
                scope_id=active.scope_id,
                parent_scope_id=active.parent_scope_id,
                name=active.name,
                path=active.path,
                sequence=self._sequence,
                attributes=active.attributes,
            )

        return _PhaseBoundary(
            event_type=PHASE_EXIT_EVENT,
            context=f"Phase exited: {format_phase_path(closed.path)}",
            metadata={
                PHASE_SCOPE_METADATA_KEY: _phase_scope_payload(closed, action="exit")
            },
            scope_id=closed.scope_id,
            path=closed.path,
        )


class PhaseTimelineResolver:
    """Replay phase boundaries and resolve the active phase path at a timestamp."""

    def __init__(
        self, intervals_by_group: Mapping[tuple[str, int], Sequence[_PhaseInterval]]
    ) -> None:
        self._intervals_by_group = {
            key: tuple(sorted(intervals, key=lambda item: (item.start_ns, item.end_ns)))
            for key, intervals in intervals_by_group.items()
        }

    @classmethod
    def from_events(cls, events: Sequence[Any]) -> "PhaseTimelineResolver":
        """Build a replay index from telemetry events."""
        session_end_by_group: dict[tuple[str, int], int] = {}
        boundaries: list[tuple[int, int, str, int, _NormalizedPhaseScope]] = []
        for event in events:
            session_id = _event_field(event, "session_id")
            timestamp_ns = _event_field(event, "timestamp_ns")
            if not isinstance(session_id, str) or not isinstance(timestamp_ns, int):
                continue
            rank = int(_event_field(event, "rank", 0))
            group_key = (session_id, rank)
            previous_end = session_end_by_group.get(group_key)
            if previous_end is None or timestamp_ns > previous_end:
                session_end_by_group[group_key] = timestamp_ns

            scope = extract_phase_scope(event)
            if scope is None:
                continue
            boundaries.append(
                (
                    timestamp_ns,
                    scope.sequence,
                    scope.scope_id,
                    rank,
                    scope,
                )
            )

        boundaries.sort(
            key=lambda item: (item[4].session_id, item[3], item[0], item[1], item[2])
        )

        active_by_thread: dict[tuple[str, int, int], list[_NormalizedPhaseScope]] = {}
        intervals_by_group: dict[tuple[str, int], list[_PhaseInterval]] = {}
        for timestamp_ns, _, _, rank, scope in boundaries:
            thread_key = (scope.session_id, rank, scope.thread_id)
            stack = active_by_thread.setdefault(thread_key, [])
            if scope.action == "enter":
                stack.append(scope)
                continue
            if not stack or stack[-1].scope_id != scope.scope_id:
                continue
            opened = stack.pop()
            if not stack:
                active_by_thread.pop(thread_key, None)
            intervals_by_group.setdefault((scope.session_id, rank), []).append(
                _PhaseInterval(
                    session_id=scope.session_id,
                    rank=rank,
                    thread_id=scope.thread_id,
                    thread_name=scope.thread_name,
                    scope_id=scope.scope_id,
                    path=opened.path,
                    start_ns=opened.timestamp_ns,
                    end_ns=timestamp_ns,
                    sequence=opened.sequence,
                )
            )

        for (session_id, rank, _thread_id), stack in active_by_thread.items():
            session_end_ns = session_end_by_group.get((session_id, rank))
            if session_end_ns is None:
                continue
            for scope in stack:
                intervals_by_group.setdefault((session_id, rank), []).append(
                    _PhaseInterval(
                        session_id=session_id,
                        rank=rank,
                        thread_id=scope.thread_id,
                        thread_name=scope.thread_name,
                        scope_id=scope.scope_id,
                        path=scope.path,
                        start_ns=scope.timestamp_ns,
                        end_ns=session_end_ns,
                        sequence=scope.sequence,
                        synthetic_end=True,
                    )
                )

        return cls(intervals_by_group)

    def resolve(
        self,
        *,
        timestamp_ns: int,
        session_id: str | None,
        rank: int | None = None,
    ) -> PhaseAttribution | None:
        """Resolve the active phase path at one timestamp."""
        thread_matches: list[_PhaseInterval] = []
        ambiguous_labels: set[str] = set()

        for (
            group_session_id,
            group_rank,
        ), intervals in self._intervals_by_group.items():
            if session_id is not None and group_session_id != session_id:
                continue
            if rank is not None and group_rank != rank:
                continue

            deepest_by_thread: dict[int, list[_PhaseInterval]] = {}
            for interval in intervals:
                if interval.start_ns <= timestamp_ns <= interval.end_ns:
                    deepest_by_thread.setdefault(interval.thread_id, []).append(
                        interval
                    )

            for thread_intervals in deepest_by_thread.values():
                max_depth = max(item.depth for item in thread_intervals)
                deepest = [item for item in thread_intervals if item.depth == max_depth]
                labels = {format_phase_path(item.path) for item in deepest}
                if len(labels) != 1:
                    ambiguous_labels.update(labels)
                    continue
                thread_matches.append(
                    max(deepest, key=lambda item: (item.sequence, item.scope_id))
                )

        if not thread_matches and not ambiguous_labels:
            return None

        if ambiguous_labels or len(thread_matches) != 1:
            phase_paths = sorted(
                ambiguous_labels
                | {format_phase_path(interval.path) for interval in thread_matches}
            )
            return PhaseAttribution(
                phase_resolution="ambiguous",
                phase_paths=phase_paths,
            )

        match = thread_matches[0]
        phase_path = format_phase_path(match.path)
        return PhaseAttribution(
            phase_resolution="unique",
            phase_path=phase_path,
            phase_paths=[phase_path],
            scope_id=match.scope_id,
            thread_id=match.thread_id,
            thread_name=match.thread_name,
        )

    def resolve_for_event(self, event: Any) -> PhaseAttribution | None:
        """Resolve phase attribution for one telemetry event-like object."""
        timestamp_ns = _event_field(event, "timestamp_ns")
        session_id = _event_field(event, "session_id")
        if not isinstance(timestamp_ns, int) or not isinstance(session_id, str):
            return None
        return self.resolve(
            timestamp_ns=timestamp_ns,
            session_id=session_id,
            rank=int(_event_field(event, "rank", 0)),
        )


@dataclass(frozen=True)
class _NormalizedPhaseScope:
    action: str
    name: str
    path: tuple[str, ...]
    depth: int
    scope_id: str
    parent_scope_id: str | None
    thread_id: int
    thread_name: str
    sequence: int
    session_id: str
    timestamp_ns: int
    attributes: dict[str, Any]


def extract_phase_scope(event: Any) -> _NormalizedPhaseScope | None:
    """Extract one normalized phase payload from an event-like object."""
    metadata = _event_field(event, "metadata", {})
    if not isinstance(metadata, Mapping):
        return None
    raw_scope = metadata.get(PHASE_SCOPE_METADATA_KEY)
    if not isinstance(raw_scope, Mapping):
        return None
    session_id = _event_field(event, "session_id")
    timestamp_ns = _event_field(event, "timestamp_ns")
    if not isinstance(session_id, str) or not isinstance(timestamp_ns, int):
        return None
    action = raw_scope.get("action")
    name = raw_scope.get("name")
    scope_id = raw_scope.get("scope_id")
    path = raw_scope.get("path")
    thread_name = raw_scope.get("thread_name")
    if (
        not isinstance(action, str)
        or action not in {"enter", "exit"}
        or not isinstance(name, str)
        or not name.strip()
        or not isinstance(scope_id, str)
        or not isinstance(thread_name, str)
        or not isinstance(path, list)
    ):
        return None
    normalized_path = tuple(
        str(part) for part in path if isinstance(part, str) and part
    )
    if not normalized_path:
        return None
    depth_value = raw_scope.get("depth")
    sequence_value = raw_scope.get("sequence")
    thread_id_value = raw_scope.get("thread_id")
    parent_scope_id_value = raw_scope.get("parent_scope_id")
    if not isinstance(depth_value, int) or depth_value != len(normalized_path):
        depth_value = len(normalized_path)
    if not isinstance(sequence_value, int):
        return None
    if not isinstance(thread_id_value, int):
        return None
    if parent_scope_id_value is not None and not isinstance(parent_scope_id_value, str):
        return None
    raw_attributes = raw_scope.get(PHASE_SCOPE_ATTRIBUTES_KEY, {})
    attributes = dict(raw_attributes) if isinstance(raw_attributes, Mapping) else {}
    return _NormalizedPhaseScope(
        action=action,
        name=name.strip(),
        path=normalized_path,
        depth=depth_value,
        scope_id=scope_id,
        parent_scope_id=parent_scope_id_value,
        thread_id=thread_id_value,
        thread_name=thread_name,
        sequence=sequence_value,
        session_id=session_id,
        timestamp_ns=timestamp_ns,
        attributes=attributes,
    )


def summarize_phase_attribution(attribution: PhaseAttribution | None) -> str | None:
    """Return a user-facing summary string for one phase attribution."""
    if attribution is None:
        return None
    return summarize_phase_resolution(
        phase_resolution=attribution.phase_resolution,
        phase_path=attribution.phase_path,
        phase_paths=attribution.phase_paths,
    )


def summarize_phase_resolution(
    *,
    phase_resolution: str | None,
    phase_path: str | None = None,
    phase_paths: Sequence[str] | None = None,
) -> str | None:
    """Render one phase resolution without hiding ambiguity semantics."""
    labels = [path for path in (phase_paths or ()) if path]
    if phase_resolution == "unique":
        if phase_path:
            return phase_path
        if len(labels) == 1:
            return labels[0]
        return None
    if phase_resolution == "ambiguous" and labels:
        return f"(ambiguous) {' | '.join(labels)}"
    return None


def merge_phase_attributions(
    first: PhaseAttribution | None,
    second: PhaseAttribution | None,
) -> PhaseAttribution | None:
    """Merge two attribution candidates without inventing a false unique path."""
    if first is None:
        return second
    if second is None:
        return first

    first_paths = _phase_paths(first)
    second_paths = _phase_paths(second)
    if not first_paths:
        return second
    if not second_paths:
        return first

    merged_paths = sorted(set(first_paths) | set(second_paths))
    if len(merged_paths) == 1:
        phase_path = merged_paths[0]
        if (
            first.phase_resolution == "unique"
            and second.phase_resolution == "unique"
            and first.phase_path == second.phase_path
            and first.scope_id == second.scope_id
            and first.thread_id == second.thread_id
        ):
            return first
        return PhaseAttribution(
            phase_resolution="ambiguous",
            phase_paths=[phase_path],
        )

    return PhaseAttribution(
        phase_resolution="ambiguous",
        phase_paths=merged_paths,
    )


def _normalize_phase_name(name: str) -> str:
    normalized = str(name).strip()
    if not normalized:
        raise ValueError("phase name must be a non-empty string")
    return normalized


def _phase_paths(attribution: PhaseAttribution) -> list[str]:
    if attribution.phase_paths:
        return [path for path in attribution.phase_paths if path]
    if attribution.phase_path:
        return [attribution.phase_path]
    return []


def _phase_scope_payload(active: _ActivePhase, *, action: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "action": action,
        "name": active.name,
        "path": list(active.path),
        "depth": len(active.path),
        "scope_id": active.scope_id,
        "parent_scope_id": active.parent_scope_id,
        "thread_id": active.thread_id,
        "thread_name": active.thread_name,
        "sequence": active.sequence,
    }
    if active.attributes:
        payload[PHASE_SCOPE_ATTRIBUTES_KEY] = dict(active.attributes)
    return payload


def _event_field(event: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(event, Mapping):
        return event.get(field_name, default)
    return getattr(event, field_name, default)


__all__ = [
    "PHASE_ENTER_EVENT",
    "PHASE_EXIT_EVENT",
    "PHASE_SCOPE_METADATA_KEY",
    "PhaseAttribution",
    "PhaseHandle",
    "PhaseTimelineResolver",
    "TrackerPhaseState",
    "extract_phase_scope",
    "format_phase_path",
    "is_phase_boundary_event",
    "merge_phase_attributions",
    "summarize_phase_attribution",
    "summarize_phase_resolution",
]
