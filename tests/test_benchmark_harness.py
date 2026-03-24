import json
from pathlib import Path

import pytest

from examples.cli import benchmark_harness


def _write_budget_file(path: Path, artifact_growth_max: float) -> None:
    path.write_text(
        json.dumps(
            {
                "version": "test",
                "budgets": {
                    "runtime_overhead_pct_max": 500.0,
                    "cpu_overhead_pct_max": 500.0,
                    "sampling_impact_pct_max": 500.0,
                    "artifact_growth_bytes_max": artifact_growth_max,
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_baseline_file(
    path: Path,
    *,
    iterations: int,
    allocation_kb: int,
    default_interval: float,
    lowfreq_interval: float,
    metrics: dict[str, float],
    version: str = benchmark_harness.REPORT_VERSION,
) -> None:
    path.write_text(
        json.dumps(
            {
                "version": version,
                "config": {
                    "iterations": iterations,
                    "allocation_kb": allocation_kb,
                    "default_interval": default_interval,
                    "lowfreq_interval": lowfreq_interval,
                },
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


def test_benchmark_harness_writes_report_with_expected_shape(tmp_path: Path) -> None:
    budgets_path = tmp_path / "budgets.json"
    _write_budget_file(budgets_path, artifact_growth_max=2_000_000.0)

    output_path = tmp_path / "report.json"
    artifact_root = tmp_path / "artifacts"

    report = benchmark_harness.run_benchmark_harness(
        iterations=8,
        allocation_kb=64,
        default_interval=0.05,
        lowfreq_interval=0.2,
        gate_mode="budget",
        budgets_path=budgets_path,
        baseline_path=None,
        tolerances_path=None,
        artifact_root=artifact_root,
        output_path=output_path,
    )

    assert output_path.exists()
    assert report["passed"] is True
    assert report["gate_mode"] == "budget"
    assert set(report["scenarios"].keys()) == {
        "unprofiled",
        "tracked_default",
        "tracked_lowfreq",
    }
    assert report["scenarios"]["tracked_default"]["event_count"] >= 2
    assert report["metrics"]["artifact_growth_bytes"] >= 0.0


def test_benchmark_harness_check_mode_fails_intentional_budget_violation(
    tmp_path: Path,
) -> None:
    strict_budgets_path = tmp_path / "strict_budgets.json"
    _write_budget_file(strict_budgets_path, artifact_growth_max=0.0)

    output_path = tmp_path / "strict_report.json"
    artifact_root = tmp_path / "strict_artifacts"

    exit_code = benchmark_harness.main(
        [
            "--iterations",
            "6",
            "--allocation-kb",
            "32",
            "--default-interval",
            "0.05",
            "--lowfreq-interval",
            "0.2",
            "--gate-mode",
            "budget",
            "--budgets",
            str(strict_budgets_path),
            "--artifact-root",
            str(artifact_root),
            "--output",
            str(output_path),
            "--check",
        ]
    )

    assert exit_code == 1
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["budget_checks"]["artifact_growth_bytes"]["passed"] is False


def test_evaluate_regressions_accepts_metric_improvements() -> None:
    metrics = {
        "runtime_overhead_pct": 8.0,
        "cpu_overhead_pct": 7.0,
        "sampling_impact_pct": 5.0,
        "artifact_growth_bytes": 1000.0,
    }
    baseline_metrics = {
        "runtime_overhead_pct": 10.0,
        "cpu_overhead_pct": 9.0,
        "sampling_impact_pct": 6.0,
        "artifact_growth_bytes": 1500.0,
    }
    tolerances = {
        "runtime_overhead_pct": 0.5,
        "cpu_overhead_pct": 0.5,
        "sampling_impact_pct": 0.5,
        "artifact_growth_bytes": 100.0,
    }

    checks = benchmark_harness.evaluate_regressions(
        metrics,
        baseline_metrics,
        tolerances,
    )

    assert all(check["passed"] for check in checks.values())
    assert checks["runtime_overhead_pct"]["delta"] == -2.0


def test_pct_overhead_preserves_signed_values() -> None:
    assert benchmark_harness._pct_overhead(100.0, 98.0) == pytest.approx(-2.0)
    assert benchmark_harness._pct_overhead(100.0, 103.5) == pytest.approx(3.5)


def test_run_benchmark_harness_regression_mode_writes_comparison_report(
    tmp_path: Path,
) -> None:
    iterations = 8
    allocation_kb = 64
    default_interval = 0.05
    lowfreq_interval = 0.2
    baseline_path = tmp_path / "baseline.json"
    tolerances_path = tmp_path / "tolerances.json"
    output_path = tmp_path / "report.json"
    artifact_root = tmp_path / "artifacts"

    _write_baseline_file(
        baseline_path,
        iterations=iterations,
        allocation_kb=allocation_kb,
        default_interval=default_interval,
        lowfreq_interval=lowfreq_interval,
        metrics={
            "runtime_overhead_pct": 0.0,
            "cpu_overhead_pct": 0.0,
            "sampling_impact_pct": 0.0,
            "artifact_growth_bytes": 0.0,
        },
    )
    _write_tolerance_file(
        tolerances_path,
        tolerances={
            "runtime_overhead_pct": 500.0,
            "cpu_overhead_pct": 500.0,
            "sampling_impact_pct": 500.0,
            "artifact_growth_bytes": 10_000.0,
        },
    )

    report = benchmark_harness.run_benchmark_harness(
        iterations=iterations,
        allocation_kb=allocation_kb,
        default_interval=default_interval,
        lowfreq_interval=lowfreq_interval,
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
    assert report["baseline"]["config"]["iterations"] == iterations
    assert report["regression_checks"]["artifact_growth_bytes"]["passed"] is True
    assert report["regression_checks"]["artifact_growth_bytes"]["delta"] >= 0.0


def test_benchmark_harness_regression_mode_fails_intentional_regression(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.json"
    tolerances_path = tmp_path / "tolerances.json"
    output_path = tmp_path / "report.json"
    artifact_root = tmp_path / "artifacts"

    _write_baseline_file(
        baseline_path,
        iterations=8,
        allocation_kb=64,
        default_interval=0.05,
        lowfreq_interval=0.2,
        metrics={
            "runtime_overhead_pct": 0.0,
            "cpu_overhead_pct": 0.0,
            "sampling_impact_pct": 0.0,
            "artifact_growth_bytes": 0.0,
        },
    )
    _write_tolerance_file(
        tolerances_path,
        tolerances={
            "runtime_overhead_pct": 500.0,
            "cpu_overhead_pct": 500.0,
            "sampling_impact_pct": 500.0,
            "artifact_growth_bytes": 0.0,
        },
    )

    exit_code = benchmark_harness.main(
        [
            "--gate-mode",
            "regression",
            "--iterations",
            "8",
            "--allocation-kb",
            "64",
            "--default-interval",
            "0.05",
            "--lowfreq-interval",
            "0.2",
            "--baseline",
            str(baseline_path),
            "--tolerances",
            str(tolerances_path),
            "--artifact-root",
            str(artifact_root),
            "--output",
            str(output_path),
            "--check",
        ]
    )

    assert exit_code == 1
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["regression_checks"]["artifact_growth_bytes"]["passed"] is False


def test_run_benchmark_harness_rejects_baseline_config_mismatch(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    tolerances_path = tmp_path / "tolerances.json"

    _write_baseline_file(
        baseline_path,
        iterations=99,
        allocation_kb=64,
        default_interval=0.05,
        lowfreq_interval=0.2,
        metrics={
            "runtime_overhead_pct": 0.0,
            "cpu_overhead_pct": 0.0,
            "sampling_impact_pct": 0.0,
            "artifact_growth_bytes": 0.0,
        },
    )
    _write_tolerance_file(
        tolerances_path,
        tolerances={
            "runtime_overhead_pct": 500.0,
            "cpu_overhead_pct": 500.0,
            "sampling_impact_pct": 500.0,
            "artifact_growth_bytes": 10_000.0,
        },
    )

    with pytest.raises(ValueError, match="Baseline config mismatch"):
        benchmark_harness.run_benchmark_harness(
            iterations=8,
            allocation_kb=64,
            default_interval=0.05,
            lowfreq_interval=0.2,
            gate_mode="regression",
            budgets_path=None,
            baseline_path=baseline_path,
            tolerances_path=tolerances_path,
            artifact_root=tmp_path / "artifacts",
            output_path=tmp_path / "report.json",
        )


def test_load_regression_baseline_rejects_wrong_version(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    _write_baseline_file(
        baseline_path,
        iterations=8,
        allocation_kb=64,
        default_interval=0.05,
        lowfreq_interval=0.2,
        metrics={
            "runtime_overhead_pct": 0.0,
            "cpu_overhead_pct": 0.0,
            "sampling_impact_pct": 0.0,
            "artifact_growth_bytes": 0.0,
        },
        version="v0.2",
    )

    with pytest.raises(ValueError, match="Baseline file version"):
        benchmark_harness.load_regression_baseline(baseline_path)


def test_load_regression_tolerances_requires_all_metrics(tmp_path: Path) -> None:
    tolerances_path = tmp_path / "tolerances.json"
    _write_tolerance_file(
        tolerances_path,
        tolerances={
            "runtime_overhead_pct": 10.0,
            "cpu_overhead_pct": 10.0,
        },
    )

    with pytest.raises(ValueError, match="missing metric keys"):
        benchmark_harness.load_regression_tolerances(tolerances_path)
