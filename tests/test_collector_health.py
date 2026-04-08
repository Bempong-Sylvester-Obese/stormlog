from __future__ import annotations

from stormlog.collector_health import collector_retry_delay_seconds


def test_collector_retry_delay_caps_overflowing_backoff() -> None:
    delay = collector_retry_delay_seconds(
        10_000,
        initial_delay_s=1.0,
        factor=2.0,
        max_delay_s=30.0,
    )

    assert delay == 30.0


def test_collector_retry_delay_rejects_non_positive_cap() -> None:
    delay = collector_retry_delay_seconds(
        3,
        initial_delay_s=1.0,
        factor=2.0,
        max_delay_s=0.0,
    )

    assert delay == 0.0
