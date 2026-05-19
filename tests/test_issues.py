from __future__ import annotations

import pytest

from stormlog.issues import (
    IssueEvidenceLink,
    IssueFingerprint,
    StormlogIssue,
    categorize_alert_context,
    normalize_dimensions,
    normalize_issue_state,
    normalized_error_stem,
)


def test_fingerprint_id_is_deterministic_for_normalized_dimensions() -> None:
    first = IssueFingerprint(
        kind="oom",
        dimensions={
            "backend": "CUDA",
            "reason": "message_pattern:out of memory",
            "partial_fields": ["device_free", "device_total"],
        },
    )
    second = IssueFingerprint(
        kind="oom",
        dimensions={
            "partial_fields": ["device_total", "device_free"],
            "reason": "MESSAGE_PATTERN:OUT OF MEMORY",
            "backend": "cuda",
        },
    )

    assert first.fingerprint_id == second.fingerprint_id
    assert first.as_dict()["fingerprint_id"].startswith("issue:")


def test_normalize_dimensions_removes_high_cardinality_numeric_text() -> None:
    normalized = normalize_dimensions(
        {
            "category": "High fragmentation: 43.5%",
            "error": "CUDA error at 0x1234 after 125 retries",
        }
    )

    assert normalized["category"] == "high fragmentation: <num>"
    assert normalized["error"] == "cuda error at <hex> after <num> retries"


def test_issue_state_validation() -> None:
    assert normalize_issue_state(" RESOLVED ") == "resolved"

    with pytest.raises(ValueError, match="unsupported issue state"):
        normalize_issue_state("muted")


def test_alert_category_and_error_stem_helpers() -> None:
    assert categorize_alert_context("High fragmentation: 40.0%") == (
        "high_fragmentation"
    )
    assert normalized_error_stem("RuntimeError: CUDA failed at step 42") == (
        "runtimeerror"
    )


def test_stormlog_issue_serializes_evidence_and_sessions() -> None:
    fingerprint = IssueFingerprint(
        kind="collector_degradation",
        dimensions={"collector": "stormlog.cuda_tracker", "status": "degraded"},
    )
    evidence = IssueEvidenceLink(
        session_id="session-b",
        timestamp_ns=20,
        rank=1,
        source_path="/tmp/track.json",
        source_kind="telemetry_json",
        event_type="collector_degraded",
        metadata={"collector_health_status": "degraded"},
    )
    issue = StormlogIssue(
        fingerprint=fingerprint,
        title="Collector degraded",
        severity="warning",
        state="open",
        hit_count=2,
        first_seen_ns=10,
        last_seen_ns=20,
        affected_sessions=["session-b", "session-a", "session-a"],
        representative_evidence=evidence,
        evidence=[evidence],
        details={"collector": "stormlog.cuda_tracker"},
    )

    payload = issue.as_dict()

    assert payload["fingerprint_id"] == fingerprint.fingerprint_id
    assert payload["affected_sessions"] == ["session-a", "session-b"]
    assert payload["representative_evidence"]["event_type"] == "collector_degraded"
    assert payload["details"]["collector"] == "stormlog.cuda_tracker"
