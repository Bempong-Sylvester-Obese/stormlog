"""Runtime phase state tracking and boundary payload emission."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping

PHASE_ENTER_EVENT = "phase_enter"
PHASE_EXIT_EVENT = "phase_exit"
PHASE_SCOPE_METADATA_KEY = "phase_scope"
PHASE_SCOPE_ATTRIBUTES_KEY = "attributes"


class PhaseProtocolError(RuntimeError):
    """Raised when phase handles are closed incorrectly."""


@dataclass(frozen=True)
class PhaseBoundary:
    """Structured boundary payload emitted into tracker telemetry events."""

    event_type: str
    context: str
    metadata: dict[str, Any]
    scope_id: str
    path: tuple[str, ...]


@dataclass(frozen=True)
class PhaseToken:
    """Opaque runtime token used for strict phase exit semantics."""

    scope_id: str
    session_id: str
    rank: int
    thread_id: int
    name: str


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


class PhaseHandle:
    """A closeable tracker phase handle returned by ``enter_phase()``."""

    def __init__(
        self,
        *,
        scope_id: str,
        name: str,
        path: tuple[str, ...],
        close_callback: Callable[[], Any],
    ) -> None:
        self.scope_id = scope_id
        self.name = name
        self.path = path
        self._close_callback = close_callback
        self._closed = False

    @property
    def phase_path(self) -> str:
        """Return the formatted phase path."""
        return _format_phase_path(self.path)

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


class PhaseRecorder:
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

    def enter(
        self,
        *,
        session_id: str,
        rank: int,
        name: str,
        attrs: Mapping[str, Any] | None = None,
    ) -> tuple[PhaseToken, PhaseBoundary]:
        """Primary runtime API returning both a token and the emitted boundary."""
        boundary = self.enter_phase(
            session_id=session_id,
            rank=rank,
            name=name,
            metadata=attrs,
        )
        thread_id = int(threading.current_thread().ident or 0)
        return (
            PhaseToken(
                scope_id=boundary.scope_id,
                session_id=session_id,
                rank=rank,
                thread_id=thread_id,
                name=name,
            ),
            boundary,
        )

    def enter_phase(
        self,
        *,
        session_id: str,
        rank: int,
        name: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> PhaseBoundary:
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

        return PhaseBoundary(
            event_type=PHASE_ENTER_EVENT,
            context=f"Phase entered: {_format_phase_path(path)}",
            metadata={
                PHASE_SCOPE_METADATA_KEY: _phase_scope_payload(active, action="enter")
            },
            scope_id=scope_id,
            path=path,
        )

    def exit(self, token: PhaseToken) -> PhaseBoundary:
        """Primary runtime API for strict token-based exit."""
        return self.exit_phase(
            session_id=token.session_id,
            rank=token.rank,
            scope_id=token.scope_id,
            thread_id=token.thread_id,
        )

    def exit_phase(
        self,
        *,
        session_id: str,
        rank: int,
        scope_id: str,
        thread_id: int,
    ) -> PhaseBoundary:
        """Register one nested phase exit transition."""
        with self._lock:
            key = (session_id, rank, thread_id)
            stack = self._active_by_thread.get(key)
            if not stack:
                if self._scope_exists(
                    session_id=session_id, rank=rank, scope_id=scope_id
                ):
                    raise PhaseProtocolError(
                        "Phase handle was closed from a different thread than it was opened."
                    )
                raise PhaseProtocolError(
                    "No active phase stack exists for this thread."
                )

            active = stack[-1]
            if active.scope_id != scope_id:
                if any(item.scope_id == scope_id for item in stack):
                    raise PhaseProtocolError(
                        "Phase handles must be closed in strict LIFO order per thread."
                    )
                if self._scope_exists(
                    session_id=session_id, rank=rank, scope_id=scope_id
                ):
                    raise PhaseProtocolError(
                        "Phase handle was closed from a different thread than it was opened."
                    )
                raise PhaseProtocolError(
                    f"Unknown phase scope_id '{scope_id}' for active tracker state."
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

        return PhaseBoundary(
            event_type=PHASE_EXIT_EVENT,
            context=f"Phase exited: {_format_phase_path(closed.path)}",
            metadata={
                PHASE_SCOPE_METADATA_KEY: _phase_scope_payload(closed, action="exit")
            },
            scope_id=closed.scope_id,
            path=closed.path,
        )

    def _scope_exists(self, *, session_id: str, rank: int, scope_id: str) -> bool:
        for (
            active_session_id,
            active_rank,
            _,
        ), stack in self._active_by_thread.items():
            if active_session_id != session_id or active_rank != rank:
                continue
            if any(item.scope_id == scope_id for item in stack):
                return True
        return False


def _normalize_phase_name(name: str) -> str:
    normalized = str(name).strip()
    if not normalized:
        raise ValueError("phase name must be a non-empty string")
    return normalized


def _format_phase_path(path: tuple[str, ...]) -> str:
    return " / ".join(part for part in path if part)


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


__all__ = [
    "PHASE_ENTER_EVENT",
    "PHASE_EXIT_EVENT",
    "PHASE_SCOPE_METADATA_KEY",
    "PHASE_SCOPE_ATTRIBUTES_KEY",
    "PhaseBoundary",
    "PhaseHandle",
    "PhaseProtocolError",
    "PhaseRecorder",
    "PhaseToken",
]
