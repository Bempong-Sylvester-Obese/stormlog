from pathlib import Path

import pytest

from examples.cli import benchmark_harness


def _write_budget_file(path: Path, budgets: dict[str, float]) -> None:
    path.write_text(
        benchmark_harness.json.dumps(
            {
                "version": benchmark_harness.REPORT_VERSION,
                "budgets": budgets,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_benchmark_harness_tensorflow_cpu_runtime_integration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tensorflow")
    monkeypatch.setitem(benchmark_harness.PROFILE_EQUIVALENT_HOURS, "pr", 0.0001)
    monkeypatch.setattr(
        benchmark_harness,
        "DEFAULT_RETENTION_VALIDATION",
        {
            "flush_every_events": 8,
            "flush_every_seconds": 2.0,
            "rollover_max_bytes": 2048,
            "rollover_max_events": 16,
            "retention_max_files": 2,
            "retention_max_total_bytes": 8192,
            "sample_limit": 64,
        },
    )

    budgets_path = tmp_path / "budgets.json"
    _write_budget_file(
        budgets_path,
        {
            "tfmemprof_cpu.runtime_overhead_pct": 1_000_000.0,
            "tfmemprof_cpu.cpu_overhead_pct": 1_000_000.0,
            "tfmemprof_cpu.artifact_growth_bytes": 1_000_000.0,
            "tfmemprof_cpu.rss_growth_per_24h_equiv": 1_000_000_000.0,
            "tfmemprof_cpu.max_rss_delta_bytes": 1_000_000_000.0,
            "tfmemprof_cpu.final_retained_bytes": 1_000_000.0,
            "tfmemprof_cpu.final_retained_files": 100.0,
            "tfmemprof_cpu.collector_failure_event_count": 0.0,
            "tfmemprof_cpu.history_dropped_events": 10_000.0,
            "tfmemprof_cpu.history_dropped_samples": 10_000.0,
            "tfmemprof_cpu.history_dropped_alerts": 10_000.0,
            "tfmemprof_cpu.rollover_count": 10_000.0,
            "tfmemprof_cpu.pruned_segment_count": 10_000.0,
            "tfmemprof_cpu.pruned_bytes": 1_000_000.0,
        },
    )

    report = benchmark_harness.run_benchmark_harness(
        profile="pr",
        mode="all",
        gate_mode="budget",
        budgets_path=budgets_path,
        baseline_path=None,
        tolerances_path=None,
        artifact_root=tmp_path / "artifacts",
        output_path=tmp_path / "report.json",
        iterations=16,
        allocation_kb=16,
        runtime_names=["tfmemprof_cpu"],
    )

    runtime_report = report["runtimes"]["tfmemprof_cpu"]
    soak = runtime_report["soak"]

    assert report["passed"] is True
    assert runtime_report["status"] == "ok"
    assert soak["sample_count"] == 1
    assert soak["collector_health_status"] == "healthy"
    assert soak["retention_validation"]["passed"] is True
    assert Path(
        runtime_report["overhead"]["scenarios"]["tracked_default"]["output_path"]
    ).exists()
