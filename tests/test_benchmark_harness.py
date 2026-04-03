import json
from pathlib import Path

import pytest

from examples.cli import benchmark_harness


class _UnusedRuntimeSession(benchmark_harness.RuntimeSession):
    def start(self) -> None:
        return None

    def emit_sample(self, index: int) -> None:
        _ = index

    def finish(self) -> dict[str, object]:
        return {
            "stats": {},
            "event_count": 0,
            "collector_failure_event_count": 0,
            "output_path": "",
        }


def _write_budget_file(
    path: Path,
    budgets: dict[str, float],
    *,
    version: str = benchmark_harness.REPORT_VERSION,
) -> None:
    path.write_text(
        json.dumps(
            {
                "version": version,
                "budgets": budgets,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_baseline_file(
    path: Path,
    *,
    config: dict[str, object],
    metrics: dict[str, float],
    version: str = benchmark_harness.REPORT_VERSION,
) -> None:
    path.write_text(
        json.dumps(
            {
                "version": version,
                "config": config,
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_tolerance_file(
    path: Path,
    *,
    tolerances: dict[str, float],
    version: str = benchmark_harness.REPORT_VERSION,
) -> None:
    path.write_text(
        json.dumps(
            {
                "version": version,
                "tolerances": tolerances,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _runtime_report(
    *,
    default_interval: float,
    runtime_overhead_pct: float,
    cpu_overhead_pct: float,
    artifact_growth_bytes: float,
    rss_growth_per_24h_equiv: float,
    max_rss_delta_bytes: float,
    collector_failure_event_count: int = 0,
    history_dropped_events: int = 0,
    history_dropped_samples: int = 0,
    history_dropped_alerts: int = 0,
    rollover_count: int = 1,
    pruned_segment_count: int = 1,
    pruned_bytes: int = 512,
    final_retained_files: int = 2,
    final_retained_bytes: int = 2048,
    collector_health_status: str = "healthy",
) -> dict[str, object]:
    tracked_stats = {
        "collector_health_status": collector_health_status,
        "rollover_count": rollover_count,
        "pruned_segment_count": pruned_segment_count,
        "pruned_bytes": pruned_bytes,
        "final_retained_files": final_retained_files,
        "final_retained_bytes": final_retained_bytes,
        "history_retained_events": 24,
        "history_dropped_events": history_dropped_events,
        "history_retained_samples": 48,
        "history_dropped_samples": history_dropped_samples,
        "history_retained_alerts": 4,
        "history_dropped_alerts": history_dropped_alerts,
    }
    return {
        "status": "ok",
        "default_interval": default_interval,
        "overhead": {
            "scenarios": {
                "unprofiled": {
                    "wall_seconds": 0.1,
                    "cpu_seconds": 0.1,
                    "artifact_size_bytes": 0,
                },
                "tracked_default": {
                    "wall_seconds": 0.11,
                    "cpu_seconds": 0.105,
                    "artifact_size_bytes": int(artifact_growth_bytes),
                    "collector_failure_event_count": collector_failure_event_count,
                    "stats": tracked_stats,
                },
            },
            "metrics": {
                "runtime_overhead_pct": runtime_overhead_pct,
                "cpu_overhead_pct": cpu_overhead_pct,
                "artifact_growth_bytes": artifact_growth_bytes,
            },
        },
        "soak": {
            "sample_count": 64,
            "equivalent_seconds": 6.4,
            "equivalent_hours": 6.0,
            "wall_seconds": 0.02,
            "cpu_seconds": 0.01,
            "artifact_size_bytes": 4096,
            "rss_growth_per_24h_equiv": rss_growth_per_24h_equiv,
            "max_rss_delta_bytes": max_rss_delta_bytes,
            "collector_failure_event_count": collector_failure_event_count,
            "history_dropped_events": history_dropped_events,
            "history_dropped_samples": history_dropped_samples,
            "history_dropped_alerts": history_dropped_alerts,
            "rollover_count": rollover_count,
            "pruned_segment_count": pruned_segment_count,
            "pruned_bytes": pruned_bytes,
            "final_retained_files": final_retained_files,
            "final_retained_bytes": final_retained_bytes,
            "history_retained_events": 24,
            "history_retained_samples": 48,
            "history_retained_alerts": 4,
            "collector_health_status": collector_health_status,
            "retention_validation": {
                "sample_limit": 64,
                "stats": tracked_stats,
                "checks": {
                    "rollover_observed": True,
                    "pruning_observed": True,
                    "retained_files_bounded": True,
                    "retained_bytes_bounded": True,
                },
                "passed": True,
                "artifact_dir": "/tmp/fake",
                "output_path": "/tmp/fake/output.json",
            },
        },
    }


def _install_fake_runtimes(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_reports = {
        "gpumemprof_cpu": _runtime_report(
            default_interval=0.1,
            runtime_overhead_pct=4.0,
            cpu_overhead_pct=3.0,
            artifact_growth_bytes=1024.0,
            rss_growth_per_24h_equiv=4096.0,
            max_rss_delta_bytes=2048.0,
        ),
        "tfmemprof_cpu": _runtime_report(
            default_interval=1.0,
            runtime_overhead_pct=6.0,
            cpu_overhead_pct=5.0,
            artifact_growth_bytes=2048.0,
            rss_growth_per_24h_equiv=8192.0,
            max_rss_delta_bytes=4096.0,
        ),
    }

    def _fake_run_runtime_report(
        spec: benchmark_harness.RuntimeSpec,
        runtime_dir: Path,
        *,
        profile: str,
        mode: str,
        iterations: int,
        allocation_kb: int,
    ) -> dict[str, object]:
        _ = runtime_dir, profile, mode, iterations, allocation_kb
        return runtime_reports[spec.name]

    monkeypatch.setattr(
        benchmark_harness,
        "_RUNTIME_SPECS",
        {
            "gpumemprof_cpu": benchmark_harness.RuntimeSpec(
                name="gpumemprof_cpu",
                default_interval=0.1,
                factory=lambda artifact_dir, interval, sink_overrides: _UnusedRuntimeSession(),
            ),
            "tfmemprof_cpu": benchmark_harness.RuntimeSpec(
                name="tfmemprof_cpu",
                default_interval=1.0,
                factory=lambda artifact_dir, interval, sink_overrides: _UnusedRuntimeSession(),
            ),
        },
    )
    monkeypatch.setattr(
        benchmark_harness,
        "_run_runtime_report",
        _fake_run_runtime_report,
    )


def test_runtime_config_uses_equivalent_time_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(benchmark_harness.PROFILE_EQUIVALENT_HOURS, "pr", 1.0 / 3600.0)

    config = benchmark_harness._runtime_config(
        "pr",
        30,
        ["gpumemprof_cpu", "tfmemprof_cpu"],
    )

    assert config["gpumemprof_cpu"]["default_interval"] == 0.1
    assert config["gpumemprof_cpu"]["overhead_sample_count"] == 30
    assert config["gpumemprof_cpu"]["soak_sample_count"] == 10
    assert config["tfmemprof_cpu"]["default_interval"] == 1.0
    assert config["tfmemprof_cpu"]["overhead_sample_count"] == 3
    assert config["tfmemprof_cpu"]["soak_sample_count"] == 1


def test_benchmark_harness_writes_v04_report_with_expected_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_runtimes(monkeypatch)
    budgets_path = tmp_path / "budgets.json"
    _write_budget_file(
        budgets_path,
        {
            "gpumemprof_cpu.runtime_overhead_pct": 100.0,
            "gpumemprof_cpu.cpu_overhead_pct": 100.0,
            "gpumemprof_cpu.artifact_growth_bytes": 10_000.0,
            "gpumemprof_cpu.rss_growth_per_24h_equiv": 50_000.0,
            "gpumemprof_cpu.max_rss_delta_bytes": 50_000.0,
            "gpumemprof_cpu.final_retained_bytes": 50_000.0,
            "gpumemprof_cpu.final_retained_files": 10.0,
            "gpumemprof_cpu.collector_failure_event_count": 1.0,
            "gpumemprof_cpu.history_dropped_events": 100.0,
            "gpumemprof_cpu.history_dropped_samples": 100.0,
            "gpumemprof_cpu.history_dropped_alerts": 100.0,
            "gpumemprof_cpu.rollover_count": 10.0,
            "gpumemprof_cpu.pruned_segment_count": 10.0,
            "gpumemprof_cpu.pruned_bytes": 10_000.0,
            "tfmemprof_cpu.runtime_overhead_pct": 100.0,
            "tfmemprof_cpu.cpu_overhead_pct": 100.0,
            "tfmemprof_cpu.artifact_growth_bytes": 10_000.0,
            "tfmemprof_cpu.rss_growth_per_24h_equiv": 50_000.0,
            "tfmemprof_cpu.max_rss_delta_bytes": 50_000.0,
            "tfmemprof_cpu.final_retained_bytes": 50_000.0,
            "tfmemprof_cpu.final_retained_files": 10.0,
            "tfmemprof_cpu.collector_failure_event_count": 1.0,
            "tfmemprof_cpu.history_dropped_events": 100.0,
            "tfmemprof_cpu.history_dropped_samples": 100.0,
            "tfmemprof_cpu.history_dropped_alerts": 100.0,
            "tfmemprof_cpu.rollover_count": 10.0,
            "tfmemprof_cpu.pruned_segment_count": 10.0,
            "tfmemprof_cpu.pruned_bytes": 10_000.0,
        },
    )

    output_path = tmp_path / "report.json"
    artifact_root = tmp_path / "artifacts"

    report = benchmark_harness.run_benchmark_harness(
        profile="pr",
        mode="all",
        gate_mode="budget",
        budgets_path=budgets_path,
        baseline_path=None,
        tolerances_path=None,
        artifact_root=artifact_root,
        output_path=output_path,
    )

    assert output_path.exists()
    assert report["version"] == benchmark_harness.REPORT_VERSION
    assert report["passed"] is True
    assert report["gate_mode"] == "budget"
    assert set(report["runtimes"].keys()) == {"gpumemprof_cpu", "tfmemprof_cpu"}
    assert (
        report["runtimes"]["gpumemprof_cpu"]["soak"]["retention_validation"]["passed"]
        is True
    )
    assert report["metrics"]["tfmemprof_cpu.max_rss_delta_bytes"] == 4096.0
    assert (
        report["budget_checks"]["gpumemprof_cpu.runtime_overhead_pct"]["passed"] is True
    )
    assert report["failure_diagnostics"] == []


def test_benchmark_harness_budget_mode_fails_intentional_violation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_runtimes(monkeypatch)
    budgets_path = tmp_path / "strict_budgets.json"
    _write_budget_file(
        budgets_path,
        {
            "gpumemprof_cpu.runtime_overhead_pct": 1.0,
            "gpumemprof_cpu.cpu_overhead_pct": 1.0,
            "gpumemprof_cpu.artifact_growth_bytes": 10.0,
            "gpumemprof_cpu.rss_growth_per_24h_equiv": 10.0,
            "gpumemprof_cpu.max_rss_delta_bytes": 10.0,
            "gpumemprof_cpu.final_retained_bytes": 10.0,
            "gpumemprof_cpu.final_retained_files": 1.0,
            "gpumemprof_cpu.collector_failure_event_count": 0.0,
            "gpumemprof_cpu.history_dropped_events": 0.0,
            "gpumemprof_cpu.history_dropped_samples": 0.0,
            "gpumemprof_cpu.history_dropped_alerts": 0.0,
            "gpumemprof_cpu.rollover_count": 0.0,
            "gpumemprof_cpu.pruned_segment_count": 0.0,
            "gpumemprof_cpu.pruned_bytes": 0.0,
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
        runtime_names=["gpumemprof_cpu"],
    )

    assert report["passed"] is False
    assert (
        report["budget_checks"]["gpumemprof_cpu.runtime_overhead_pct"]["passed"]
        is False
    )
    assert any("collector=healthy" in item for item in report["failure_diagnostics"])
    assert any(
        "history_retained_events=24" in item for item in report["failure_diagnostics"]
    )


def test_evaluate_regressions_accepts_metric_improvements() -> None:
    metrics = {
        "gpumemprof_cpu.runtime_overhead_pct": 8.0,
        "gpumemprof_cpu.cpu_overhead_pct": 7.0,
    }
    baseline_metrics = {
        "gpumemprof_cpu.runtime_overhead_pct": 10.0,
        "gpumemprof_cpu.cpu_overhead_pct": 9.0,
    }
    tolerances = {
        "gpumemprof_cpu.runtime_overhead_pct": 0.5,
        "gpumemprof_cpu.cpu_overhead_pct": 0.5,
    }

    checks = benchmark_harness.evaluate_regressions(
        metrics,
        baseline_metrics,
        tolerances,
    )

    assert all(check["passed"] for check in checks.values())
    assert checks["gpumemprof_cpu.runtime_overhead_pct"]["delta"] == -2.0


def test_pct_overhead_preserves_signed_values() -> None:
    assert benchmark_harness._pct_overhead(100.0, 98.0) == pytest.approx(-2.0)
    assert benchmark_harness._pct_overhead(100.0, 103.5) == pytest.approx(3.5)


def test_normalize_comparison_config_preserves_integral_fields() -> None:
    config = benchmark_harness._normalize_comparison_config(
        {
            "profile": "pr",
            "mode": "all",
            "iterations": 10,
            "allocation_kb": 64,
            "profile_equivalent_hours": 6.0,
            "runtimes": {
                "gpumemprof_cpu": {
                    "default_interval": 0.1,
                    "overhead_sample_count": 10,
                    "soak_sample_count": 216000,
                }
            },
            "retention_validation": benchmark_harness.DEFAULT_RETENTION_VALIDATION,
        },
        label="Baseline file",
    )

    assert config["iterations"] == 10
    assert config["allocation_kb"] == 64
    assert isinstance(config["iterations"], int)
    assert isinstance(config["allocation_kb"], int)
    assert isinstance(
        config["runtimes"]["gpumemprof_cpu"]["overhead_sample_count"],
        int,
    )


def test_run_benchmark_harness_regression_mode_writes_comparison_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_runtimes(monkeypatch)
    output_path = tmp_path / "report.json"
    artifact_root = tmp_path / "artifacts"

    config = {
        "profile": "pr",
        "mode": "all",
        "iterations": 5000,
        "allocation_kb": 512,
        "profile_equivalent_hours": benchmark_harness.PROFILE_EQUIVALENT_HOURS["pr"],
        "runtimes": benchmark_harness._runtime_config(
            "pr",
            5000,
            ["gpumemprof_cpu", "tfmemprof_cpu"],
        ),
        "retention_validation": dict(benchmark_harness.DEFAULT_RETENTION_VALIDATION),
    }
    metrics = {
        "gpumemprof_cpu.runtime_overhead_pct": 4.0,
        "gpumemprof_cpu.cpu_overhead_pct": 3.0,
        "gpumemprof_cpu.artifact_growth_bytes": 1024.0,
        "gpumemprof_cpu.rss_growth_per_24h_equiv": 4096.0,
        "gpumemprof_cpu.max_rss_delta_bytes": 2048.0,
        "gpumemprof_cpu.final_retained_bytes": 2048.0,
        "gpumemprof_cpu.final_retained_files": 2.0,
        "gpumemprof_cpu.collector_failure_event_count": 0.0,
        "gpumemprof_cpu.history_dropped_events": 0.0,
        "gpumemprof_cpu.history_dropped_samples": 0.0,
        "gpumemprof_cpu.history_dropped_alerts": 0.0,
        "gpumemprof_cpu.rollover_count": 1.0,
        "gpumemprof_cpu.pruned_segment_count": 1.0,
        "gpumemprof_cpu.pruned_bytes": 512.0,
        "tfmemprof_cpu.runtime_overhead_pct": 6.0,
        "tfmemprof_cpu.cpu_overhead_pct": 5.0,
        "tfmemprof_cpu.artifact_growth_bytes": 2048.0,
        "tfmemprof_cpu.rss_growth_per_24h_equiv": 8192.0,
        "tfmemprof_cpu.max_rss_delta_bytes": 4096.0,
        "tfmemprof_cpu.final_retained_bytes": 2048.0,
        "tfmemprof_cpu.final_retained_files": 2.0,
        "tfmemprof_cpu.collector_failure_event_count": 0.0,
        "tfmemprof_cpu.history_dropped_events": 0.0,
        "tfmemprof_cpu.history_dropped_samples": 0.0,
        "tfmemprof_cpu.history_dropped_alerts": 0.0,
        "tfmemprof_cpu.rollover_count": 1.0,
        "tfmemprof_cpu.pruned_segment_count": 1.0,
        "tfmemprof_cpu.pruned_bytes": 512.0,
    }
    baseline_path = tmp_path / "baseline.json"
    tolerances_path = tmp_path / "tolerances.json"
    _write_baseline_file(baseline_path, config=config, metrics=metrics)
    _write_tolerance_file(
        tolerances_path,
        tolerances={metric_key: 50.0 for metric_key in metrics},
    )

    report = benchmark_harness.run_benchmark_harness(
        profile="pr",
        mode="all",
        gate_mode="regression",
        budgets_path=None,
        baseline_path=baseline_path,
        tolerances_path=tolerances_path,
        artifact_root=artifact_root,
        output_path=output_path,
    )

    assert output_path.exists()
    assert report["gate_mode"] == "regression"
    assert report["passed"] is True
    assert report["baseline"]["config"]["iterations"] == 5000
    assert report["regression_checks"]["tfmemprof_cpu.pruned_bytes"]["passed"] is True


def test_benchmark_harness_regression_mode_fails_intentional_regression(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_runtimes(monkeypatch)
    config = {
        "profile": "pr",
        "mode": "all",
        "iterations": 5000,
        "allocation_kb": 512,
        "profile_equivalent_hours": benchmark_harness.PROFILE_EQUIVALENT_HOURS["pr"],
        "runtimes": benchmark_harness._runtime_config("pr", 5000, ["gpumemprof_cpu"]),
        "retention_validation": dict(benchmark_harness.DEFAULT_RETENTION_VALIDATION),
    }
    baseline_metrics = {
        "gpumemprof_cpu.runtime_overhead_pct": 0.0,
        "gpumemprof_cpu.cpu_overhead_pct": 0.0,
        "gpumemprof_cpu.artifact_growth_bytes": 0.0,
        "gpumemprof_cpu.rss_growth_per_24h_equiv": 0.0,
        "gpumemprof_cpu.max_rss_delta_bytes": 0.0,
        "gpumemprof_cpu.final_retained_bytes": 0.0,
        "gpumemprof_cpu.final_retained_files": 0.0,
        "gpumemprof_cpu.collector_failure_event_count": 0.0,
        "gpumemprof_cpu.history_dropped_events": 0.0,
        "gpumemprof_cpu.history_dropped_samples": 0.0,
        "gpumemprof_cpu.history_dropped_alerts": 0.0,
        "gpumemprof_cpu.rollover_count": 0.0,
        "gpumemprof_cpu.pruned_segment_count": 0.0,
        "gpumemprof_cpu.pruned_bytes": 0.0,
    }
    baseline_path = tmp_path / "baseline.json"
    tolerances_path = tmp_path / "tolerances.json"
    _write_baseline_file(
        baseline_path,
        config=config,
        metrics=baseline_metrics,
    )
    _write_tolerance_file(
        tolerances_path,
        tolerances={metric_key: 0.0 for metric_key in baseline_metrics},
    )

    report = benchmark_harness.run_benchmark_harness(
        profile="pr",
        mode="all",
        gate_mode="regression",
        budgets_path=None,
        baseline_path=baseline_path,
        tolerances_path=tolerances_path,
        artifact_root=tmp_path / "artifacts",
        output_path=tmp_path / "report.json",
        runtime_names=["gpumemprof_cpu"],
    )

    assert report["passed"] is False
    assert (
        report["regression_checks"]["gpumemprof_cpu.runtime_overhead_pct"]["passed"]
        is False
    )
    assert any(
        "gpumemprof_cpu.runtime_overhead_pct" in item
        for item in report["failure_diagnostics"]
    )


def test_run_benchmark_harness_rejects_baseline_config_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_runtimes(monkeypatch)
    baseline_path = tmp_path / "baseline.json"
    tolerances_path = tmp_path / "tolerances.json"

    _write_baseline_file(
        baseline_path,
        config={
            "profile": "nightly",
            "mode": "all",
            "iterations": 99,
            "allocation_kb": 64,
            "profile_equivalent_hours": 24.0,
            "runtimes": benchmark_harness._runtime_config(
                "nightly",
                99,
                ["gpumemprof_cpu"],
            ),
            "retention_validation": dict(
                benchmark_harness.DEFAULT_RETENTION_VALIDATION
            ),
        },
        metrics={"gpumemprof_cpu.runtime_overhead_pct": 0.0},
    )
    _write_tolerance_file(
        tolerances_path,
        tolerances={"gpumemprof_cpu.runtime_overhead_pct": 10.0},
    )

    with pytest.raises(ValueError, match="Baseline config mismatch"):
        benchmark_harness.run_benchmark_harness(
            profile="pr",
            mode="all",
            gate_mode="regression",
            budgets_path=None,
            baseline_path=baseline_path,
            tolerances_path=tolerances_path,
            artifact_root=tmp_path / "artifacts",
            output_path=tmp_path / "report.json",
            runtime_names=["gpumemprof_cpu"],
        )


def test_load_regression_baseline_rejects_wrong_version(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    _write_baseline_file(
        baseline_path,
        config={
            "profile": "pr",
            "mode": "all",
            "iterations": 8,
            "allocation_kb": 64,
            "profile_equivalent_hours": 6.0,
            "runtimes": benchmark_harness._runtime_config(
                "pr",
                8,
                ["gpumemprof_cpu"],
            ),
            "retention_validation": dict(
                benchmark_harness.DEFAULT_RETENTION_VALIDATION
            ),
        },
        metrics={"gpumemprof_cpu.runtime_overhead_pct": 0.0},
        version="v0.3",
    )

    with pytest.raises(ValueError, match="Baseline file version"):
        benchmark_harness.load_regression_baseline(baseline_path)


def test_load_budget_thresholds_rejects_wrong_version(tmp_path: Path) -> None:
    budget_path = tmp_path / "budget.json"
    _write_budget_file(
        budget_path,
        {"gpumemprof_cpu.runtime_overhead_pct": 10.0},
        version="v0.3",
    )

    with pytest.raises(ValueError, match="Budget file version"):
        benchmark_harness.load_budget_thresholds(budget_path)


def test_load_regression_tolerances_requires_all_metrics(tmp_path: Path) -> None:
    tolerances_path = tmp_path / "tolerances.json"
    _write_tolerance_file(
        tolerances_path,
        tolerances={
            "gpumemprof_cpu.runtime_overhead_pct": 10.0,
        },
    )

    with pytest.raises(ValueError, match="missing metric keys"):
        benchmark_harness.evaluate_regressions(
            metrics={
                "gpumemprof_cpu.runtime_overhead_pct": 1.0,
                "gpumemprof_cpu.cpu_overhead_pct": 1.0,
            },
            baseline_metrics={
                "gpumemprof_cpu.runtime_overhead_pct": 0.0,
                "gpumemprof_cpu.cpu_overhead_pct": 0.0,
            },
            tolerances=benchmark_harness.load_regression_tolerances(tolerances_path),
        )


def test_failure_diagnostics_include_runtime_context_for_budget_failures() -> None:
    report = {
        "runtimes": {
            "gpumemprof_cpu": _runtime_report(
                default_interval=0.1,
                runtime_overhead_pct=8.0,
                cpu_overhead_pct=7.0,
                artifact_growth_bytes=1024.0,
                rss_growth_per_24h_equiv=4096.0,
                max_rss_delta_bytes=2048.0,
                history_dropped_events=5,
                history_dropped_samples=11,
                history_dropped_alerts=2,
                rollover_count=3,
                pruned_segment_count=2,
                pruned_bytes=1536,
                final_retained_files=2,
                final_retained_bytes=2048,
            )
        },
        "budget_checks": {
            "gpumemprof_cpu.runtime_overhead_pct": {
                "value": 8.0,
                "max_allowed": 1.0,
                "passed": False,
            }
        },
    }

    failures = benchmark_harness._failure_diagnostics(report)

    assert len(failures) == 1
    assert "gpumemprof_cpu.runtime_overhead_pct" in failures[0]
    assert "collector=healthy" in failures[0]
    assert "rollover_count=3" in failures[0]
    assert "pruned_segment_count=2" in failures[0]
    assert "history_dropped_samples=11" in failures[0]


def test_run_benchmark_harness_cpu_runtime_integration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setitem(benchmark_harness.PROFILE_EQUIVALENT_HOURS, "pr", 0.0001)
    monkeypatch.setattr(
        benchmark_harness,
        "DEFAULT_RETENTION_VALIDATION",
        {
            "flush_every_events": 10,
            "flush_every_seconds": 2.0,
            "rollover_max_bytes": 2048,
            "rollover_max_events": 20,
            "retention_max_files": 2,
            "retention_max_total_bytes": 8192,
            "sample_limit": 128,
        },
    )

    budgets_path = tmp_path / "budgets.json"
    _write_budget_file(
        budgets_path,
        {
            "gpumemprof_cpu.runtime_overhead_pct": 1_000_000.0,
            "gpumemprof_cpu.cpu_overhead_pct": 1_000_000.0,
            "gpumemprof_cpu.artifact_growth_bytes": 1_000_000.0,
            "gpumemprof_cpu.rss_growth_per_24h_equiv": 1_000_000_000.0,
            "gpumemprof_cpu.max_rss_delta_bytes": 1_000_000_000.0,
            "gpumemprof_cpu.final_retained_bytes": 1_000_000.0,
            "gpumemprof_cpu.final_retained_files": 100.0,
            "gpumemprof_cpu.collector_failure_event_count": 0.0,
            "gpumemprof_cpu.history_dropped_events": 10_000.0,
            "gpumemprof_cpu.history_dropped_samples": 10_000.0,
            "gpumemprof_cpu.history_dropped_alerts": 10_000.0,
            "gpumemprof_cpu.rollover_count": 10_000.0,
            "gpumemprof_cpu.pruned_segment_count": 10_000.0,
            "gpumemprof_cpu.pruned_bytes": 1_000_000.0,
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
        iterations=32,
        allocation_kb=32,
        runtime_names=["gpumemprof_cpu"],
    )

    soak = report["runtimes"]["gpumemprof_cpu"]["soak"]
    assert report["passed"] is True
    assert soak["sample_count"] == 4
    assert soak["retention_validation"]["passed"] is True
    assert soak["retention_validation"]["checks"]["rollover_observed"] is True
    assert soak["retention_validation"]["checks"]["pruning_observed"] is True
