from __future__ import annotations

from dataclasses import dataclass

from stormlog.session import (
    SESSION_STATUS_COMPLETED,
    SESSION_STATUS_RUNNING,
    SessionSummary,
    _HasSummary,
    create_session_summary,
    select_default_session,
    update_session_summary,
)


@dataclass(frozen=True)
class _LoadedSession(_HasSummary):
    summary: SessionSummary


def test_update_session_summary_can_clear_terminal_timestamp() -> None:
    summary = create_session_summary(
        source="stormlog.test",
        status=SESSION_STATUS_COMPLETED,
        started_at_ns=10,
        ended_at_ns=20,
    )

    refreshed = update_session_summary(
        summary,
        status=SESSION_STATUS_RUNNING,
        ended_at_ns=None,
    )

    assert refreshed.status == SESSION_STATUS_RUNNING
    assert refreshed.ended_at_ns is None


def test_select_default_session_accepts_raw_session_summaries() -> None:
    running = create_session_summary(
        source="stormlog.test",
        status=SESSION_STATUS_RUNNING,
        session_id="running",
        started_at_ns=10,
    )
    completed = create_session_summary(
        source="stormlog.test",
        status=SESSION_STATUS_COMPLETED,
        session_id="completed",
        started_at_ns=9,
        ended_at_ns=30,
    )

    selected = select_default_session([running, completed])

    assert selected is completed


def test_select_default_session_accepts_loaded_wrappers() -> None:
    running = _LoadedSession(
        create_session_summary(
            source="stormlog.test",
            status=SESSION_STATUS_RUNNING,
            session_id="running-loaded",
            started_at_ns=10,
        )
    )
    completed = _LoadedSession(
        create_session_summary(
            source="stormlog.test",
            status=SESSION_STATUS_COMPLETED,
            session_id="completed-loaded",
            started_at_ns=9,
            ended_at_ns=30,
        )
    )

    selected = select_default_session([running, completed])

    assert selected is completed
